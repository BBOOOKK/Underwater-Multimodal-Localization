import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import bmtm, SO3


# =============================================================================
# 🔥 BLOCK A: 激活逻辑 (修复版)
# =============================================================================

class BaseLoss(torch.nn.Module):
    def __init__(self, min_N, max_N, dt):
        super().__init__()
        self.min_N = min_N
        self.max_N = max_N
        self.min_train_freq = 2 ** self.min_N
        self.dt = dt


class GyroLoss(BaseLoss):
    def __init__(self, w, min_N, max_N, dt, target, huber):
        super().__init__(min_N, max_N, dt)
        self.w = w
        self.huber = huber
        self.target_type = target
        self.sl = torch.nn.SmoothL1Loss()
        self.register_buffer('weight', torch.ones(1, 1, self.min_train_freq) / self.min_train_freq)

    def f_huber(self, rs):
        if rs.numel() == 0:
            return torch.tensor(0.0, device=rs.device, dtype=torch.float64, requires_grad=True)
        return self.w * self.sl(rs / self.huber, torch.zeros_like(rs)) * (self.huber ** 2)

    def _decimate(self, x, steps, op_type='rotation'):
        for _ in range(steps):
            if x.shape[0] % 2 != 0: x = x[:-1]
            if op_type == 'rotation':
                x = x[::2].bmm(x[1::2])
            else:
                x = SO3.qmul(x[::2], x[1::2])
        return x

    def _pad_sequence(self, xs, hat_xs, alignment_factor):
        N_batch, N_seq, dim = xs.shape
        remainder = N_seq % alignment_factor
        if remainder == 0: return xs, hat_xs
        pad_len = alignment_factor - remainder
        xs_padded = F.pad(xs.permute(0, 2, 1), (0, pad_len), "constant", 0).permute(0, 2, 1)
        hat_xs_padded = F.pad(hat_xs.permute(0, 2, 1), (0, pad_len), "constant", 0).permute(0, 2, 1)
        return xs_padded, hat_xs_padded

    def forward(self, xs, hat_xs):
        if isinstance(hat_xs, tuple): hat_xs = hat_xs[0]
        current_max_freq = 2 ** self.max_N
        xs, hat_xs = self._pad_sequence(xs, hat_xs, current_max_freq)
        N_batch = xs.shape[0]

        has_mask = xs.shape[-1] >= 4
        masks = None
        if has_mask:
            masks = xs[:, :, 3].unsqueeze(1)
            masks = F.conv1d(masks, self.weight, stride=self.min_train_freq).transpose(1, 2)
            masks = (masks >= 1.0).double().reshape(N_batch, -1, 1)

        Xs_input = xs[:, ::self.min_train_freq, :3].reshape(-1, 3).double()
        Xs = SO3.exp(Xs_input)
        hat_phi = self.dt * hat_xs.reshape(-1, 3).double()
        Omegas = SO3.exp(hat_phi)
        Omegas = self._decimate(Omegas, self.min_N, 'rotation')

        loss = torch.tensor(0.0, device=xs.device, dtype=torch.float64)

        for k in range(self.min_N, self.max_N + 1):
            res_mat = bmtm(Omegas, Xs)
            curr_rs = SO3.log(res_mat).view(N_batch, -1, 3)
            current_len = curr_rs.shape[1]
            if current_len > 0:
                curr_rs_flat = curr_rs.reshape(-1, 3)
                if has_mask:
                    curr_mask_flat = masks.reshape(-1, 1)
                    valid_indices = (curr_mask_flat.expand_as(curr_rs_flat) == 1)
                    curr_rs_flat = curr_rs_flat[valid_indices].reshape(-1, 3)
                scale = 1.0 if k == self.min_N else (2 ** (k - self.min_N))
                loss = loss + (self.f_huber(curr_rs_flat) / scale)
            if k < self.max_N:
                Omegas = Omegas[::2].bmm(Omegas[1::2])
                Xs = Xs[::2].bmm(Xs[1::2])
                if has_mask: masks = masks[:, ::2, :] * masks[:, 1::2, :]
        return loss.float()


class MoELoss(nn.Module):
    def __init__(self, task_loss_fn, aux_weight=0.01):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.aux_weight = aux_weight

    def forward(self, xs, model_output, target_expert=None):
        hat_xs, (gates, logits) = model_output
        task_loss = self.task_loss_fn(xs, hat_xs)

        # 🔥 强力监督逻辑 (Supervised Gating Logic)
        if target_expert is not None and gates is not None:
            # gates shape: (Batch, Seq, Experts)
            # 我们希望 target_expert 对应的概率最大化
            # Loss = -log(prob_target)

            # 取出目标专家的概率
            target_prob = gates[..., target_expert]
            # 加 epsilon 防止 log(0)
            gate_loss = -torch.log(target_prob + 1e-8).mean()

            # 权重设为 2.0，甚至比任务 Loss 更重要，因为选错专家=全盘皆输
            return task_loss + 2.0 * gate_loss

            # 无监督时的负载均衡
        if gates is not None:
            importance = gates.sum(dim=(0, 1))
            total_samples = gates.size(0) * gates.size(1)
            target = torch.full_like(importance, total_samples / gates.size(-1))
            balance_loss = F.mse_loss(importance, target)
            return task_loss + self.aux_weight * balance_loss

        return task_loss
