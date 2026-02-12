import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils import bbmv


# =============================================================================
# 🔥 最终修复版 Networks (智能归一化 + 防火墙机制)
# =============================================================================

# 1. BaseNet
class BaseNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, c0, dropout, ks, ds, momentum):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 注册 Buffer
        self.register_buffer('mean_u', torch.zeros(in_dim))
        self.register_buffer('std_u', torch.ones(in_dim))

        c1 = 2 * c0
        c2 = 2 * c1
        c3 = 2 * c2
        k0, k1, k2, k3 = ks
        d0, d1, d2 = ds
        p0 = (k0 - 1) + d0 * (k1 - 1) + d0 * d1 * (k2 - 1) + d0 * d1 * d2 * (k3 - 1)

        self.cnn = torch.nn.Sequential(
            torch.nn.ReplicationPad1d((p0, 0)),
            torch.nn.Conv1d(in_dim, c0, k0, dilation=1),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),

            torch.nn.Conv1d(c0, c1, k1, dilation=d0),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),

            torch.nn.Conv1d(c1, c2, k2, dilation=d0 * d1),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),

            torch.nn.Conv1d(c2, c3, k3, dilation=d0 * d1 * d2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),

            torch.nn.Conv1d(c3, out_dim, 1, dilation=1),
        )

    def set_normalized_factors(self, mean_u, std_u):
        # 单个专家训练时 (train_specialists.py)，允许 Dataset 传入真实的统计参数
        if mean_u is not None:
            self.mean_u.data = mean_u.to(self.mean_u.device)
        if std_u is not None:
            self.std_u.data = std_u.to(self.std_u.device)

    def forward(self, us):
        # 🔥 [修正] 智能归一化逻辑
        # 1. 如果 std_u 是真实计算出来的（通常 < 0.9，例如 0.05），说明是专家模式。
        #    此时使用 (us - mean) / std，这能有效去除 IMU 的 Bias！
        # 2. 如果 std_u 是默认值 1.0 (Router 或 Raw 模式)，则回退到 / 30.0。

        if self.std_u.mean() < 0.9:
            # 专家模式：去偏置 + 归一化
            u_norm = (us - self.mean_u) / (self.std_u + 1e-8)
        else:
            # 原始模式：简单缩放
            u_norm = us / 30.0

        u = u_norm.transpose(1, 2)
        y = self.cnn(u)
        return y.transpose(1, 2)


# 2. GyroNet (专家模型)
class GyroNet(BaseNet):
    def __init__(self, in_dim, out_dim, c0, dropout, ks, ds, momentum, gyro_std, **kwargs):
        super().__init__(in_dim, out_dim, c0, dropout, ks, ds, momentum)

        if out_dim != 3:
            raise ValueError("GyroNet out_dim must be 3")

        self.register_buffer('gyro_std', torch.tensor(gyro_std, dtype=torch.float32))
        self.register_buffer('Id3', torch.eye(3))

        self.gyro_Rot = torch.nn.Parameter(torch.randn(3, 3) * 1e-3)

    def forward(self, us):
        ys = super().forward(us)

        base_gyro = us[:, :, :3]
        calib_mat = self.Id3 + self.gyro_Rot

        Rots = calib_mat.view(1, 1, 3, 3).expand(us.shape[0], us.shape[1], 3, 3)
        Rot_us = bbmv(Rots, base_gyro)

        return self.gyro_std * ys + Rot_us


# 3. TemporalRouter (门控网络)
class TemporalRouter(nn.Module):
    def __init__(self, in_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts

        self.feature_net = nn.Sequential(
            nn.Conv1d(in_dim * 3, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.Conv1d(32, num_experts, kernel_size=1)
        )

    def forward(self, us):
        x_raw = us.transpose(1, 2)
        # Router 始终使用固定缩放，不受统计量影响
        x_norm = x_raw / 30.0

        feat_energy = torch.log(torch.abs(x_raw) + 1e-7)

        pad = 25
        avg = F.avg_pool1d(x_raw, kernel_size=51, stride=1, padding=pad)[:, :, :x_raw.shape[2]]
        diff = torch.abs(x_raw - avg)
        local_max_diff = F.max_pool1d(diff, kernel_size=51, stride=1, padding=pad)[:, :, :x_raw.shape[2]]
        feat_var = torch.log(local_max_diff + 1e-9)

        x_in = torch.cat([x_norm, feat_energy, feat_var], dim=1)

        logits = self.feature_net(x_in)
        gates = F.softmax(logits, dim=1)

        return gates.transpose(1, 2), logits.transpose(1, 2)


# 4. MoEGyroNet (混合网络)
class MoEGyroNet(nn.Module):
    def __init__(self, num_experts=4, top_k=None, **net_params):
        super().__init__()
        self.experts = nn.ModuleList([
            GyroNet(**net_params) for _ in range(num_experts)
        ])

        self.router = TemporalRouter(in_dim=net_params['in_dim'], num_experts=num_experts)

    def set_normalized_factors(self, mean_u, std_u):
        # 🔥🔥🔥 防火墙逻辑 (正确保留) 🔥🔥🔥
        # 禁止 Dataset 覆盖专家的参数
        pass

    def forward(self, us):
        gates, logits = self.router(us)

        expert_outputs = torch.stack([expert(us) for expert in self.experts], dim=1)

        gates_exp = gates.permute(0, 2, 1).unsqueeze(-1)
        final_output = torch.sum(expert_outputs * gates_exp, dim=1)

        return final_output, (gates, logits)