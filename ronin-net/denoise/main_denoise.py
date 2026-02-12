# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import src.learning as lr
import src.networks as sn
import src.losses as sl
import src.dataset as ds
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils import SO3, bmtm, pload, pdump, mkdir
import torch.nn as nn
import torch.nn.functional as F
import shutil

# ================= CONFIGURATION =================
ROOT_DIR = r"/ronin-net\denoise"
TRAIN_SOURCE_DIR = os.path.join(ROOT_DIR, "..", "data", "processed_data")
TEST_SOURCE_DIR = os.path.join(ROOT_DIR, "..", "data", "all_data")
CACHE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'src', 'data', 'UNDERWATER_CACHE')

SPECIALISTS_DIR = os.path.join(ROOT_DIR, "output", "best_experts")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "moe_fusion_results")
LOG_DIR = os.path.join(ROOT_DIR, "output", "moe_logs")

for d in [OUTPUT_DIR, LOG_DIR]:
    if not os.path.exists(d): os.makedirs(d)

TRAIN_SEQS = [
    '06', '07', '08', '09', 'suiyi1', 'suiyi2',
    'zhengfangxing1', 'zhengfangxing2', 'zhengfangxing3',
    'zhixian1', 'zhixian2',
    '02'
]
VAL_SEQS = ['10']
TEST_SEQS = ['01', '02', '03', '04', '05']

NET_PARAMS = {
    'in_dim': 6, 'out_dim': 3, 'c0': 32,
    'num_experts': 4, 'top_k': None,
    'dropout': 0.05,
    'ks': [11, 11, 11, 11], 'ds': [4, 4, 4], 'momentum': 0.1,
    'gyro_std': [0.017, 0.035, 0.087],
}


class SmartMoELearner(lr.GyroLearningBasedProcessing):
    def __init__(self, res_dir, tb_dir, net_params, address, dt):
        super().__init__(res_dir, tb_dir, sn.MoEGyroNet, net_params, address, dt)

    def sliding_window_inference(self, us, window_size=100000):
        # 全量推理
        B, L, D = us.shape
        if L <= window_size:
            with torch.no_grad():
                chunk_out_tuple = self.net(us)
                if isinstance(chunk_out_tuple, tuple):
                    return chunk_out_tuple[0]
                else:
                    return chunk_out_tuple

        pad_len = (window_size - (L % window_size)) % window_size
        if pad_len > 0:
            padding = torch.zeros(B, pad_len, D, device=us.device)
            us_padded = torch.cat([us, padding], dim=1)
        else:
            us_padded = us
        L_padded = us_padded.shape[1]
        chunks = us_padded.view(-1, window_size, D)
        with torch.no_grad():
            chunk_out_tuple = self.net(chunks)
            if isinstance(chunk_out_tuple, tuple):
                chunk_out = chunk_out_tuple[0]
            else:
                chunk_out = chunk_out_tuple
        full_out = chunk_out.view(1, L_padded, -1)
        return full_out[:, :L, :]

    def loop_test(self, dataset):
        self.net.eval()
        for i in tqdm(range(len(dataset)), desc="Testing"):
            seq = dataset.sequences[i]
            data_dict = dataset.load_seq(i)
            us = data_dict['us'].unsqueeze(0).to(self.device)

            with torch.no_grad():
                hat_xs = self.sliding_window_inference(us, window_size=100000)

            mkdir(self.address, seq)
            pdump({'hat_xs': hat_xs[0].cpu()}, self.address, seq, 'results.p')

    # 🔥 [已删除 display_test 覆盖]
    # 现在它会自动继承 learning.py 中的画图逻辑，解决绘图慢/不绘图的问题。


class GodModeMoELearner(SmartMoELearner):
    def __init__(self, res_dir, tb_dir, net_params, address, dt):
        super().__init__(res_dir, tb_dir, net_params, address, dt)
        if address is None:
            self.load_and_freeze_specialists()

    def load_and_freeze_specialists(self):
        expert_files = ['Expert_0_Static.pt', 'Expert_1_Dynamic.pt', 'Expert_2_Mixed.pt', 'Expert_3_Robust.pt']
        print(f"\n>>> 📥 Loading Specialists from {SPECIALISTS_DIR}...")
        for i, filename in enumerate(expert_files):
            path = os.path.join(SPECIALISTS_DIR, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ Missing expert file: {path}")
            try:
                state_dict = torch.load(path, map_location=self.device)
                self.net.experts[i].load_state_dict(state_dict)
                print(f"   ✅ Expert {i} loaded: {filename}")
            except Exception as e:
                new_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                self.net.experts[i].load_state_dict(new_dict)
                print(f"   ✅ Expert {i} loaded (Fixed Prefix): {filename}")

        print("❄️ Freezing Experts, Training Router ONLY...")
        for name, param in self.net.named_parameters():
            if 'router' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def loop_train(self, dataloader, optimizer, criterion, epoch):
        loss_epoch = 0
        self.net.train()
        pbar = tqdm(dataloader, desc=f"Ep {epoch} (Refined Tiered V4)", leave=False)

        for i, data in enumerate(pbar):
            us = data['u'].to(self.device)
            xs = data['x'].to(self.device)

            if hasattr(dataloader.dataset, 'add_noise'):
                us = dataloader.dataset.add_noise(us)

            optimizer.zero_grad()
            hat_xs, (gates, _) = self.net(us)

            task_loss = criterion.task_loss_fn(xs, hat_xs)

            # 🔥 [稳健的维度对齐]
            # 既然 Router 是 Global 的 [B, 4]，我们就把 Input 降维成 [B] 来匹配它
            raw_mag = torch.norm(us[..., :3], dim=2)
            seq_mag = raw_mag.mean(dim=1)  # Shape: (Batch_Size,)

            # 🔥 [精细化分流策略] Tiered Routing

            # 1. 静止区 (Static Zone): Mag < 0.11
            # 目标：Seq 02 (0.105) -> 强制 E0
            mask_static = seq_mag < 0.11

            # 2. 净化区 (Purge Zone): 0.11 <= Mag < 0.18
            # 目标：Seq 05 (0.132) -> E0 在此失效，必须封杀 E0
            mask_purge_e0 = (seq_mag >= 0.11) & (seq_mag < 0.18)

            # 3. 自由区 (Free Zone): 0.18 <= Mag < 0.30
            # 目标：Seq 01 (0.23), Seq 04 (0.23) -> 无约束，自由竞争

            # 4. 极端区 (Extreme Zone): Mag >= 0.30
            # 目标：保护大动态稳定性 -> 强制 E3
            mask_extreme = seq_mag >= 0.30

            # --- 计算 Gate Loss (严格匹配 [B] 维度) ---
            gate_loss = torch.tensor(0.0, device=self.device)

            # A. 强制 E0 (静止区)
            if mask_static.any():
                gate_loss += -torch.log(gates[mask_static, 0] + 1e-8).mean() * 4.0

            # B. 强制 E3 (极端区)
            if mask_extreme.any():
                gate_loss += -torch.log(gates[mask_extreme, 3] + 1e-8).mean() * 4.0

            # C. 封杀 E0 (净化区 - 救 Seq 05)
            # 使用平方惩罚，迫使 E0 的概率趋近于 0
            if mask_purge_e0.any():
                gate_loss += (gates[mask_purge_e0, 0] ** 2).mean() * 10.0

            # 4. 熵正则 (鼓励非黑即白)
            entropy = -torch.sum(gates * torch.log(gates + 1e-8), dim=-1).mean()
            entropy_loss = -0.05 * entropy

            total_loss = task_loss + gate_loss + entropy_loss
            total_loss.backward()
            optimizer.step()
            loss_epoch += total_loss.item()

            if i % 10 == 0:
                # 兼容 2D 和 3D 的打印逻辑
                if gates.ndim == 3:
                    g0 = gates[..., 0].mean().item()
                else:
                    g0 = gates[:, 0].mean().item()

                pbar.set_postfix({
                    'L': f'{total_loss.item():.2f}',
                    'E0_Gate': f'{g0:.3f}'
                })

        return loss_epoch / len(dataloader)


def fast_integrate_accelerated(gyro, q0, dt):
    device = gyro.device
    N = gyro.shape[0]
    if q0.dim() == 1: q0 = q0.view(1, 4)
    dqs = SO3.qexp(gyro * dt)
    qs = torch.zeros(N, 4, device=device)
    qs[0] = q0
    curr_q = q0
    dqs_list = list(torch.unbind(dqs))
    res_list = [curr_q]
    for i in range(N - 1):
        dq_next = dqs_list[i].view(1, 4)
        curr_q = SO3.qmul(curr_q, dq_next)
        res_list.append(curr_q)
    return torch.cat(res_list, 0)


def evaluate_quality(learner, dataset_class, dataset_params, test_seqs):
    print(f"\n{'=' * 80}")
    print(f"📊 MoE Fusion Comprehensive Evaluation")
    print(f"{'=' * 80}")
    print(f"{'Seq':<5} | {'Dur':<5} | {'RRE (Raw -> Net)':<20} | {'ATE (Raw -> Net)':<20} | {'ATE Imp':<8}")
    print("-" * 80)

    device = learner.device
    dt = 0.01
    STRIDE = 10

    test_params = dataset_params.copy()
    test_params['data_dir'] = TEST_SOURCE_DIR
    dataset = dataset_class(**test_params, mode='test')

    for i, seq in enumerate(dataset.sequences):
        if seq not in test_seqs: continue
        try:
            res_path = os.path.join(learner.address, seq, 'results.p')
            net_res = pload(res_path)
            hat_xs = net_res['hat_xs'].to(device).double()
            gt = dataset.load_gt(i)
            gt_qs = gt['qs'].to(device).double()
            data = dataset.load_seq(i)
            raw_gyro = data['us'][:, :3].to(device).double()
        except:
            continue

        N = min(hat_xs.shape[0], gt_qs.shape[0])
        N_crop = (N // STRIDE) * STRIDE
        if N_crop == 0: continue

        hat_xs = hat_xs[:N_crop]
        gt_qs = gt_qs[:N_crop]
        raw_gyro = raw_gyro[:N_crop]

        t_start = torch.arange(0, N_crop - 100, 100, device=device)
        if len(t_start) == 0:
            rre_net, rre_raw = 0.0, 0.0
        else:
            gyro_chunks = hat_xs[:len(t_start) * 100].view(-1, 100, 3)
            dRot_pred = SO3.exp(gyro_chunks.sum(dim=1) * dt)
            qs_start = gt_qs[t_start]
            qs_end = gt_qs[t_start + 100]
            dRot_gt = bmtm(SO3.from_quaternion(qs_start), SO3.from_quaternion(qs_end))
            rre_net = (SO3.log(bmtm(dRot_pred, dRot_gt)).norm(dim=1) * 180 / np.pi).mean().item()

            raw_chunks = raw_gyro[:len(t_start) * 100].view(-1, 100, 3)
            dRot_raw = SO3.exp(raw_chunks.sum(dim=1) * dt)
            rre_raw = (SO3.log(bmtm(dRot_raw, dRot_gt)).norm(dim=1) * 180 / np.pi).mean().item()

        dt_agg = dt * STRIDE
        q0 = gt_qs[0].view(1, 4).float()
        gt_qs_sub = gt_qs[::STRIDE]

        hat_xs_agg = hat_xs.view(-1, STRIDE, 3).mean(dim=1).float()
        q_pred = fast_integrate_accelerated(hat_xs_agg, q0, dt_agg).double()
        R_pred = SO3.from_quaternion(q_pred)
        R_gt = SO3.from_quaternion(gt_qs_sub)
        ate_net = torch.sqrt((SO3.log(bmtm(R_pred, R_gt)).norm(dim=1) ** 2).mean()).item() * 180 / np.pi

        raw_xs_agg = raw_gyro.view(-1, STRIDE, 3).mean(dim=1).float()
        q_raw = fast_integrate_accelerated(raw_xs_agg, q0, dt_agg).double()
        R_raw = SO3.from_quaternion(q_raw)
        ate_raw = torch.sqrt((SO3.log(bmtm(R_raw, R_gt)).norm(dim=1) ** 2).mean()).item() * 180 / np.pi

        ate_improv = (1 - ate_net / (ate_raw + 1e-8)) * 100

        rre_str = f"{rre_raw:.2f} -> {rre_net:.2f}"
        ate_str = f"{ate_raw:.1f} -> {ate_net:.1f}"

        print(f"{seq:<5} | {N * dt:<5.0f} | {rre_str:<20} | {ate_str:<20} | {ate_improv:>7.2f}%")
    print("-" * 80 + "\n")


def prepare_cache():
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 MoE Fusion (Refined Tiered Strategy) on {device}")
    prepare_cache()

    train_params = {
        'optimizer_class': torch.optim.Adam,
        'optimizer': {'lr': 1e-3, 'weight_decay': 1e-5},
        'loss_class': sl.MoELoss,
        'loss': {
            'task_loss_fn': sl.GyroLoss(w=10000, min_N=4, max_N=8, dt=0.01, target='rotation matrix', huber=0.01),
            'aux_weight': 0.0
        },
        'scheduler_class': torch.optim.lr_scheduler.StepLR,
        'scheduler': {'step_size': 15, 'gamma': 0.5},
        'dataloader': {'batch_size': 16, 'pin_memory': True, 'num_workers': 0, 'shuffle': True},
        'freq_val': 5,
        'n_epochs': 50,
        'res_dir': OUTPUT_DIR,
        'tb_dir': LOG_DIR
    }

    dataset_params = {
        'data_dir': TRAIN_SOURCE_DIR, 'predata_dir': CACHE_DIR,
        'train_seqs': TRAIN_SEQS, 'val_seqs': VAL_SEQS, 'test_seqs': TEST_SEQS,
        'N': 2048, 'min_train_freq': 16, 'max_train_freq': 32,
    }

    learner = GodModeMoELearner(OUTPUT_DIR, LOG_DIR, NET_PARAMS, None, 0.01)
    learner.train(ds.EUROCDataset, dataset_params, train_params)

    best_model_path = os.path.join(learner.address, 'weights.pt')
    print(f"🔍 Loading best model from: {best_model_path}")
    tester = GodModeMoELearner(OUTPUT_DIR, LOG_DIR, NET_PARAMS, best_model_path, 0.01)
    tester.test(ds.EUROCDataset, dataset_params.copy(), ['test'])
    evaluate_quality(tester, ds.EUROCDataset, dataset_params, TEST_SEQS)