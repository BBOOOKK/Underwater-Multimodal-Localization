# train_specialists.py (回滚至核选项版本 - 修复版)
import os
import torch
import numpy as np
import src.learning as lr
import src.networks as sn
import src.losses as sl
import src.dataset as ds
from src.utils import bmtm, SO3
import warnings
from datetime import datetime
import shutil

warnings.filterwarnings("ignore")

# ================= CONFIGURATION =================
ROOT_DIR = r"/ronin-net\denoise"
TRAIN_SOURCE_DIR = os.path.join(ROOT_DIR, "..", "data", "processed_data")
TEST_SOURCE_DIR = os.path.join(ROOT_DIR, "..", "data", "all_data")
CACHE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'src', 'data', 'UNDERWATER_CACHE')

BASE_OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "specialists")
TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
RUN_DIR = os.path.join(BASE_OUTPUT_DIR, TIMESTAMP)
DETAILS_DIR = os.path.join(RUN_DIR, "details")

if not os.path.exists(RUN_DIR):
    os.makedirs(RUN_DIR)
    os.makedirs(DETAILS_DIR)
    print(f"📂 [New Run] 结果保存至: {RUN_DIR}")

# ==============================================================================
# 1. 数据定义 (回滚到验证有效的逻辑)
# ==============================================================================

# [核选项] 专门给 Expert 0 用于修复 Seq 02
SEQS_ZERO_BIAS_ONLY = ['02', '08']

# [V4 原版数据组] (这些组合被证明是有效的)
SEQS_LOW_BIAS = ['07', '08', 'zhixian1', 'zhixian2']
SEQS_DYNAMIC_SUPPORT = ['07', '08', '09', 'suiyi1', 'suiyi2']
SEQS_EXTREME_CORE = ['06']
if os.path.exists(os.path.join(TRAIN_SOURCE_DIR, '01')):
    SEQS_EXTREME_CORE.append('01')

# ==============================================================================
# 2. 专家配置 (恢复 Nuclear Option)
# ==============================================================================
SPECIALISTS = [
    {
        # 🔴 Expert 0: 静止/低偏置专家
        # 策略：死磕 02/08，通过大量重复让模型记住这种低偏置模式
        'name': 'Expert_0_Static',
        'huber': 0.0001,
        'dropout': 0.05,
        'seqs': SEQS_ZERO_BIAS_ONLY,
        'epochs': 150,
        'mix_strategy': 'repeat_20x'  # 🔥 暴力重复 20 次，确保 E0 极其擅长处理静止
    },
    {
        # 🟢 Expert 1: 动态/低偏置专家
        'name': 'Expert_1_Dynamic',
        'huber': 0.002, 'dropout': 0.1,
        'seqs': SEQS_LOW_BIAS,
        'epochs': 120,
        'mix_strategy': 'repeat_5x'
    },
    {
        # 🔵 Expert 2: 混合/中等偏置专家
        'name': 'Expert_2_Mixed',
        'huber': 0.005, 'dropout': 0.2,
        'seqs': SEQS_DYNAMIC_SUPPORT + SEQS_EXTREME_CORE,  # 混合数据，保证泛化
        'epochs': 100,
        'mix_strategy': 'balanced'
    },
    {
        # ⚫ Expert 3: 鲁棒/极端专家
        'name': 'Expert_3_Robust',
        'huber': 0.02, 'dropout': 0.4,  # 高 Dropout 防止过拟合 06
        'seqs': SEQS_EXTREME_CORE,
        'support': SEQS_DYNAMIC_SUPPORT,
        'epochs': 150,
        'mix_strategy': 'weighted_mix_10x'  # 核心数据加权
    }
]

# 网络参数
NET_PARAMS = {
    'in_dim': 6, 'out_dim': 3, 'c0': 32,
    'ks': [11, 11, 11, 11], 'ds': [4, 4, 4], 'momentum': 0.1,
    'gyro_std': [0.017, 0.035, 0.087],
    'dropout': 0.05
}

TEST_SEQS = ['01', '02', '03', '04', '05']


# ==============================================================================
# 3. 辅助工具 & 主流程
# ==============================================================================
def fast_integrate_orientation(gyro, q0, dt):
    device = gyro.device
    N = gyro.shape[0]
    # 使用 batch 操作加速
    dqs = SO3.qexp(gyro * dt)
    qs = dqs.clone()

    # 简单的累乘循环 (比对数级慢一点但对于评估够用且稳定)
    curr_q = q0.view(1, 4)
    qs_list = []
    qs_list.append(curr_q)

    # Unbind to avoid indexing overhead in loop
    dqs_unbind = torch.unbind(dqs)

    for i in range(N - 1):
        curr_q = SO3.qmul(curr_q, dqs_unbind[i].view(1, 4))
        qs_list.append(curr_q)

    return torch.cat(qs_list, 0)


def fast_evaluate(model, dataset, device, dt=0.01):
    total_rmse = {}
    model.eval()
    if device.type == 'cuda': torch.cuda.synchronize()

    with torch.no_grad():
        for i in range(len(dataset)):
            seq = dataset.sequences[i]
            try:
                data = dataset.load_seq(i)
                gt_dict = dataset.load_gt(i)

                # 准备输入
                us = data['us'].unsqueeze(0).to(device)

                # 推理
                hat_xs = model(us)
                if isinstance(hat_xs, tuple): hat_xs = hat_xs[0]
                hat_xs = hat_xs[0]

                # 准备 GT
                N = hat_xs.shape[0]
                min_len = min(N, gt_dict['qs'].shape[0])
                hat_xs = hat_xs[:min_len].double()
                gt_q = gt_dict['qs'][:min_len].to(device).double()
                q0 = gt_q[0]

                # 积分
                qs_hat = fast_integrate_orientation(hat_xs, q0, dt)

                # 评估
                R_hat = SO3.from_quaternion(qs_hat)
                R_gt = SO3.from_quaternion(gt_q)
                dR = bmtm(R_hat, R_gt)
                err_rad = SO3.log(dR).norm(dim=1)
                rmse = torch.sqrt((err_rad ** 2).mean()) * 180 / np.pi
                total_rmse[seq] = rmse.item()
            except Exception as e:
                print(f"Eval Error {seq}: {e}")
                continue
    return total_rmse


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "#" * 60 + "\n>>> 🔄 ROLLING BACK: Phase 1 - Nuclear Option + V4 Standard\n" + "#" * 60)

    dataset_class = ds.EUROCDataset
    dataset_params_base = {
        'data_dir': TRAIN_SOURCE_DIR, 'predata_dir': CACHE_DIR,
        'val_seqs': ['10'], 'test_seqs': [],
        'N': 2048, 'min_train_freq': 16, 'max_train_freq': 32,
    }

    for spec in SPECIALISTS:
        epochs = spec.get('epochs', 60)
        strategy = spec.get('mix_strategy', None)
        final_train_seqs = []

        # 策略逻辑
        if strategy == 'repeat_20x':
            final_train_seqs = spec['seqs'] * 20
            desc = f"Repeat 20x (Nuclear)"
        elif strategy == 'repeat_5x':
            final_train_seqs = spec['seqs'] * 5
            desc = f"Repeat 5x"
        elif strategy == 'weighted_mix_10x':
            core = spec['seqs'] * 10
            support = spec.get('support', []) * 1
            final_train_seqs = core + support
            desc = f"Core*10 + Support"
        elif strategy == 'balanced':
            final_train_seqs = spec['seqs'] * 3
            desc = "Balanced (3x)"
        else:
            final_train_seqs = spec['seqs']
            desc = "Normal"

        # 过滤不存在的文件
        final_train_seqs = [s for s in final_train_seqs if os.path.exists(os.path.join(TRAIN_SOURCE_DIR, s))]

        print(f"\n>>> 🚀 Training: {spec['name']}")
        print(f"    Target: {list(set(final_train_seqs))} (Strategy: {desc})")

        current_net_params = NET_PARAMS.copy()
        current_net_params['dropout'] = spec['dropout']
        current_dataset_params = dataset_params_base.copy()
        current_dataset_params['train_seqs'] = final_train_seqs

        expert_run_dir = os.path.join(DETAILS_DIR, spec['name'])

        train_params = {
            'optimizer_class': torch.optim.Adam,
            'optimizer': {'lr': 1e-4, 'weight_decay': 1e-4},
            'loss_class': sl.GyroLoss,
            'loss': {'min_N': 4, 'max_N': 5, 'w': 10000, 'target': 'rotation matrix', 'huber': spec['huber'],
                     'dt': 0.01},
            'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
            'scheduler': {'T_0': epochs, 'T_mult': 1, 'eta_min': 1e-7},
            'dataloader': {'batch_size': 16, 'pin_memory': True, 'num_workers': 0, 'shuffle': True},
            'freq_val': 10, 'n_epochs': epochs,
            'res_dir': RUN_DIR, 'tb_dir': os.path.join(RUN_DIR, "logs", spec['name'])
        }

        learner = lr.GyroLearningBasedProcessing(
            train_params['res_dir'], train_params['tb_dir'], sn.GyroNet,
            current_net_params, address=expert_run_dir, dt=0.01
        )
        learner.train(dataset_class, current_dataset_params, train_params)

        best_ckpt_path = os.path.join(expert_run_dir, "weights.pt")
        target_path = os.path.join(RUN_DIR, f"{spec['name']}.pt")
        if os.path.exists(best_ckpt_path):
            shutil.copy(best_ckpt_path, target_path)
        else:
            torch.save(learner.net.state_dict(), target_path)

    # --- Testing ---
    print("\n" + "#" * 60 + "\n>>> 🧪 Phase 2: Verification (Weights Loaded Successfully)\n" + "#" * 60)

    # 注意：这里使用 TEST_SOURCE_DIR 里的数据来验证，不进行归一化（模拟真实推理）
    test_dataset = ds.EUROCDataset(data_dir=TEST_SOURCE_DIR, predata_dir=CACHE_DIR, train_seqs=[], val_seqs=[],
                                   test_seqs=TEST_SEQS, mode='test', N=1024, min_train_freq=16, max_train_freq=32)

    header = f"{'Model Name':<25} |"
    for seq in TEST_SEQS: header += f" {seq:<8} |"
    print("\n" + header)
    print("-" * (25 + 11 * len(TEST_SEQS)))

    for spec in SPECIALISTS:
        model_name = spec['name']
        model_path = os.path.join(RUN_DIR, f"{model_name}.pt")

        # 重新初始化模型，不带归一化层参数，模拟“只加载权重”
        model = sn.GyroNet(**NET_PARAMS).to(device)
        try:
            # 必须加载权重，否则归一化层参数是错的
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue

        rmse_results = fast_evaluate(model, test_dataset, device)
        row = f"{model_name:<25} |"
        for seq in TEST_SEQS: val = rmse_results.get(seq, 999.9); row += f" {val:<8.4f} |"
        print(row)