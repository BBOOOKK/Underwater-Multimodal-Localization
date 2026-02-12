# select_best_specialists.py (修复维度错误版)
import os
import torch
import shutil
import numpy as np
import src.networks as sn
import src.dataset as ds
from src.utils import bmtm, SO3
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. 配置
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = CURRENT_DIR

SEARCH_DIRS = [
    os.path.join(ROOT_DIR, "output", "specialists"),
    os.path.join(ROOT_DIR, "output", "denoise_results"),
    os.path.join(ROOT_DIR, "output", "moe_fusion_results")
]

BEST_DIR = os.path.join(ROOT_DIR, "output", "best_experts")
TEST_DATA_DIR = os.path.join(ROOT_DIR, "..", "data", "all_data")
CACHE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'src', 'data', 'UNDERWATER_CACHE')

TEST_SEQS = ['01', '02', '03', '04', '05']

NET_PARAMS = {
    'in_dim': 6, 'out_dim': 3, 'c0': 32,
    'ks': [11, 11, 11, 11], 'ds': [4, 4, 4], 'momentum': 0.1,
    'gyro_std': [0.017, 0.035, 0.087],
    'dropout': 0.0
}


# ==============================================================================
# 2. 核心工具
# ==============================================================================
def fast_integrate_orientation(gyro, q0, dt):
    device = gyro.device
    N = gyro.shape[0]
    dqs = SO3.qexp(gyro * dt)
    qs = dqs.clone()
    N_log = int(np.ceil(np.log2(N)))
    for i in range(N_log):
        k = 2 ** i
        if k < N: qs[k:] = SO3.qnorm(SO3.qmul(qs[:-k], qs[k:]))
    q0_broadcast = q0.view(1, 4).expand(N, 4)
    return SO3.qmul(q0_broadcast, qs)


def inspect_checkpoint(model_path):
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        keys = list(state_dict.keys())
        if any('experts.' in k for k in keys) or any('router.' in k for k in keys):
            return False, "Is MoE Router"
        if not any('cnn.' in k for k in keys):
            return False, "Not GyroNet"
        if 'std_u' in state_dict:
            std_val = state_dict['std_u'].mean().item()
            if std_val > 0.9:
                return True, "RAW"
            else:
                return True, "STAT"
        return True, "UNKNOWN"
    except:
        return False, "Corrupted"


def evaluate_model(model_path, dataset, device):
    is_valid, msg = inspect_checkpoint(model_path)
    if not is_valid: return None
    try:
        model = sn.GyroNet(**NET_PARAMS).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
    except:
        return None

    scores = {}
    total_err = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            seq = dataset.sequences[i]
            data = dataset.load_seq(i)
            gt_dict = dataset.load_gt(i)
            us = data['us'].unsqueeze(0).to(device)  # Shape: (1, L, 6)

            # 🔥 [修复逻辑] 获取输出并去除 Batch 维度
            output = model(us)
            if isinstance(output, tuple):
                hat_xs = output[0]
            else:
                hat_xs = output

            # 🔥 关键修正：(1, L, 3) -> (L, 3)
            # 如果不 squeeze，fast_integrate_orientation 里的 einsum 会报错
            hat_xs = hat_xs.squeeze(0)

            N = hat_xs.shape[0]
            min_len = min(N, gt_dict['qs'].shape[0])
            hat_xs = hat_xs[:min_len].double()
            gt_q = gt_dict['qs'][:min_len].to(device).double()
            q0 = gt_q[0]

            qs_hat = fast_integrate_orientation(hat_xs, q0, 0.01)
            R_hat = SO3.from_quaternion(qs_hat)
            R_gt = SO3.from_quaternion(gt_q)
            dR = bmtm(R_hat, R_gt)
            err_rad = SO3.log(dR).norm(dim=1)
            rmse = torch.sqrt((err_rad ** 2).mean()) * 180 / np.pi
            scores[seq] = rmse.item()
            total_err += rmse.item()
    scores['avg'] = total_err / len(dataset)
    scores['is_stat_ok'] = (msg == "STAT")
    return scores


# ==============================================================================
# 3. 主程序
# ==============================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_files = []
    print("🔍 正在扫描以下目录寻找候选模型:")
    for s_dir in SEARCH_DIRS:
        print(f"  -> {s_dir}")
        if not os.path.exists(s_dir): continue
        for root, dirs, files in os.walk(s_dir):
            for file in files:
                if file.endswith(".pt") and "checkpoint" not in file:
                    model_files.append(os.path.join(root, file))

    print(f"\n🔍 总共找到 {len(model_files)} 个候选模型，开始评估...\n")
    dataset = ds.EUROCDataset(
        data_dir=TEST_DATA_DIR, predata_dir=CACHE_DIR,
        train_seqs=[], val_seqs=[], test_seqs=TEST_SEQS,
        mode='test', N=1024, min_train_freq=16, max_train_freq=32
    )

    results = []
    print(f"{'Model Name':<40} | {'Type':<6} | {'01':<7} | {'02':<7} | {'03':<7} | {'04':<7} | {'05':<7} | {'Avg':<7}")
    print("-" * 110)

    for m_path in model_files:
        scores = evaluate_model(m_path, dataset, device)
        if scores:
            parent = os.path.basename(os.path.dirname(m_path))
            fname = os.path.basename(m_path)
            if len(parent) > 20: parent = parent[:10] + "..." + parent[-8:]
            disp_name = f"{parent}/{fname}"[-40:]
            tag = "✅STAT" if scores['is_stat_ok'] else "⚠️RAW"
            results.append({'path': m_path, 'name': disp_name, 'scores': scores, 'timestamp': os.path.getmtime(m_path)})
            print(
                f"{disp_name:<40} | {tag:<6} | {scores.get('01', 99):<7.2f} | {scores.get('02', 99):<7.2f} | {scores.get('03', 99):<7.2f} | {scores.get('04', 99):<7.2f} | {scores.get('05', 99):<7.2f} | {scores['avg']:<7.2f}")
    print("-" * 110)


    def find_best(metric_key):
        candidates = [r for r in results if r['scores']['is_stat_ok']]
        if not candidates: candidates = results
        if not candidates: return None
        return sorted(candidates, key=lambda x: (x['scores'].get(metric_key, 9999), -x['timestamp']))[0]


    if not results: exit()

    print("\n🏆 === 最终选拔 (按 4 级路由逻辑适配) ===")

    # Expert 0 (Static): 选 02 最好的
    best_static = find_best('02')

    # Expert 1 (Stable): 选 04 最好的
    best_stable = find_best('04')

    # Expert 2 (Mixed): 选 01 最好的
    best_mixed = find_best('01')

    # Expert 3 (Extreme): 选 Avg 最好的
    best_robust = find_best('avg')

    target_map = {}
    if best_static:
        print(f"🥇 Expert 0 (Static/Seq02) : {best_static['name']} (Err: {best_static['scores']['02']:.2f})")
        target_map['Expert_0_Static.pt'] = best_static['path']

    if best_stable:
        print(f"🥇 Expert 1 (Stable/Seq04) : {best_stable['name']} (Err: {best_stable['scores']['04']:.2f})")
        target_map['Expert_1_Dynamic.pt'] = best_stable['path']

    if best_mixed:
        print(f"🥇 Expert 2 (Mixed/Seq01)  : {best_mixed['name']} (Err: {best_mixed['scores']['01']:.2f})")
        target_map['Expert_2_Mixed.pt'] = best_mixed['path']

    if best_robust:
        print(f"🥇 Expert 3 (Robust/Avg)   : {best_robust['name']} (Avg: {best_robust['scores']['avg']:.2f})")
        target_map['Expert_3_Robust.pt'] = best_robust['path']

    if not os.path.exists(BEST_DIR): os.makedirs(BEST_DIR)
    print(f"\n💾 正在保存最强阵容...")
    for new_name, src_path in target_map.items():
        shutil.copy(src_path, os.path.join(BEST_DIR, new_name))
        print(f"  -> {new_name}")

    print("\n✅ 选拔完成！")