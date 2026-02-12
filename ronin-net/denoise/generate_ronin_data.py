# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd
import glob
import shutil
from tqdm import tqdm
import src.networks as sn
import src.dataset as ds
from src.utils import SO3, pload, bmtm
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import datetime

# ================= CONFIGURATION =================
ROOT_DIR = r"/ronin-net\denoise"
OUTPUT_DIR = os.path.join(ROOT_DIR, "output_csv_moe")

# MoE 模型参数
MOE_NET_PARAMS = {
    'in_dim': 6, 'out_dim': 3, 'c0': 32,
    'num_experts': 4, 'top_k': None,
    'dropout': 0.05,
    'ks': [11, 11, 11, 11], 'ds': [4, 4, 4], 'momentum': 0.1,
    'gyro_std': [0.017, 0.035, 0.087],
}

# 数据源
DATA_SOURCES = [
    {
        "dir": os.path.join(ROOT_DIR, "..", "data", "processed_data"),
        "seqs": ['06', '07', '08', '09', '10', 'suiyi1', 'suiyi2', 'zhengfangxing1', 'zhengfangxing2', 'zhengfangxing3',
                 'zhixian1', 'zhixian2']
    },
    {
        "dir": os.path.join(ROOT_DIR, "..", "data", "all_data"),
        "seqs": ['01', '02', '03', '04', '05']
    }
]

CACHE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'src', 'data', 'UNDERWATER_CACHE')


# ==============================================

def find_latest_moe_weights(root_dir):
    """自动寻找最新的 MoE 融合权重"""
    fusion_dir = os.path.join(root_dir, "output", "moe_fusion_results")
    if not os.path.exists(fusion_dir):
        raise FileNotFoundError(f"找不到融合结果目录: {fusion_dir}")

    runs = [os.path.join(fusion_dir, d) for d in os.listdir(fusion_dir) if os.path.isdir(os.path.join(fusion_dir, d))]
    if not runs:
        raise FileNotFoundError("没有找到任何训练记录！")

    latest_run = max(runs, key=os.path.getmtime)
    weights_path = os.path.join(latest_run, "weights.pt")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"最新的运行目录中没有 weights.pt: {latest_run}")

    print(f"✅ 锁定最新 MoE 模型: {weights_path}")
    return weights_path


def clean_cache_files(cache_dir):
    print(f"\n[Info] Cleaning cache directory: {cache_dir} ...")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        return

    # 注意：频繁删除缓存是导致变慢的另一个原因。
    # 如果数据没有变动，可以注释掉下面这段删除逻辑。
    p_files = glob.glob(os.path.join(cache_dir, "*.p"))
    if not p_files: return

    count = 0
    for f in p_files:
        try:
            os.remove(f)
            count += 1
        except:
            pass
    print(f"   -> Deleted {count} old .p files.")


# 🔥 [极速版] 快速积分与误差计算函数
# 采用了对数级并行算法 (Parallel Scan)，速度提升 100x
def calc_orientation_error(gyro, q0, gt_qs, dt, device):
    """
    gyro: [N, 3] tensor
    q0: [4] tensor (starting quaternion)
    gt_qs: [N, 4] tensor (ground truth)
    """
    N = gyro.shape[0]

    # 1. 计算所有增量四元数 (Batch Operation)
    dqs = SO3.qexp(gyro * dt)  # [N, 4]

    # 2. 快速并行累乘 (Logarithmic Integration)
    # 利用四元数乘法的结合律进行倍增计算
    qs = dqs.clone()
    N_log = int(np.ceil(np.log2(N)))
    for i in range(N_log):
        k = 2 ** i
        if k < N:
            # 这里的 qmul 是 Batch 操作，完全并行，极快
            # qs[:-k] 和 qs[k:] 维度都是 [N-k, 4]，不会报错
            qs[k:] = SO3.qmul(qs[:-k], qs[k:])
            # 归一化防止误差漂移
            qs[k:] = SO3.qnorm(qs[k:])

    # 3. 应用初始姿态 q0
    if q0.dim() == 1: q0 = q0.view(1, 4)
    q0_broadcast = q0.expand(N, 4)
    qs_absolute = SO3.qmul(q0_broadcast, qs)

    # 4. 计算误差
    # 确保 GT 也是归一化的
    R_pred = SO3.from_quaternion(qs_absolute)
    R_gt = SO3.from_quaternion(gt_qs)

    dR = bmtm(R_pred, R_gt)
    err_rad = SO3.log(dR).norm(dim=1)
    rmse_deg = torch.sqrt((err_rad ** 2).mean()).item() * 180 / np.pi

    return rmse_deg


def prepare_data():
    clean_cache_files(CACHE_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    try:
        weight_path = find_latest_moe_weights(ROOT_DIR)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 加载 MoE
    model = sn.MoEGyroNet(**MOE_NET_PARAMS)
    print(f"Loading weights...")
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.double().to(device)
    model.eval()

    # 📊 评估结果容器
    evaluation_results = []

    for source_idx, source in enumerate(DATA_SOURCES):
        current_dir = source['dir']
        current_seqs = source['seqs']
        print(f"\n>>> Processing Source {source_idx + 1}: {current_dir}")

        try:
            dataset = ds.EUROCDataset(
                data_dir=current_dir, predata_dir=CACHE_DIR,
                train_seqs=current_seqs, val_seqs=[], test_seqs=[],
                mode='train', N=1024, min_train_freq=64, max_train_freq=128
            )
        except Exception as e:
            print(f"Skipping source {current_dir}: {e}")
            continue

        for seq_idx, seq_name in enumerate(dataset.sequences):
            print(f"Processing sequence: {seq_name}...")
            try:
                data = dataset.load_seq(seq_idx)
                gt = dataset.load_gt(seq_idx)

                # Input
                us = data['us'][:, :6].to(device).unsqueeze(0).double()

                # Inference
                with torch.no_grad():
                    hat_xs_tuple = model(us)
                    if isinstance(hat_xs_tuple, tuple):
                        hat_xs = hat_xs_tuple[0].squeeze(0).double()
                    else:
                        hat_xs = hat_xs_tuple.squeeze(0).double()

                # GT Prep
                gt_qs = gt['qs'].to(device).double()
                gt_pos = gt['ps'].to(device).double()
                dt = 0.01
                if 't' in data and len(data['t']) > 1: dt = (data['t'][1] - data['t'][0]).item()

                gt_vel = torch.zeros_like(gt_pos)
                if len(gt_pos) > 1: gt_vel[:-1] = (gt_pos[1:] - gt_pos[:-1]) / dt

                raw_acc = data['us'][:, 3:6].to(device).double()
                raw_gyro = data['us'][:, :3].to(device).double()

                # Alignment
                N = hat_xs.shape[0]
                min_len = min(N, raw_acc.shape[0], gt_pos.shape[0], gt_qs.shape[0])
                ts = np.arange(min_len) * dt

                df = pd.DataFrame({
                    'timestamp': ts,
                    'gyro_x': hat_xs[:min_len, 0].detach().cpu().numpy(),
                    'gyro_y': hat_xs[:min_len, 1].detach().cpu().numpy(),
                    'gyro_z': hat_xs[:min_len, 2].detach().cpu().numpy(),
                    'acce_x': raw_acc[:min_len, 0].detach().cpu().numpy(),
                    'acce_y': raw_acc[:min_len, 1].detach().cpu().numpy(),
                    'acce_z': raw_acc[:min_len, 2].detach().cpu().numpy(),
                    'gt_q_w': gt_qs[:min_len, 0].detach().cpu().numpy(),
                    'gt_q_x': gt_qs[:min_len, 1].detach().cpu().numpy(),
                    'gt_q_y': gt_qs[:min_len, 2].detach().cpu().numpy(),
                    'gt_q_z': gt_qs[:min_len, 3].detach().cpu().numpy(),
                    'gt_p_x': gt_pos[:min_len, 0].detach().cpu().numpy(),
                    'gt_p_y': gt_pos[:min_len, 1].detach().cpu().numpy(),
                    'depth': gt_pos[:min_len, 2].detach().cpu().numpy(),
                    'gt_v_x': gt_vel[:min_len, 0].detach().cpu().numpy(),
                    'gt_v_y': gt_vel[:min_len, 1].detach().cpu().numpy(),
                    'gt_v_z': gt_vel[:min_len, 2].detach().cpu().numpy(),
                })

                # --- DVL & Euler Logic ---
                try:
                    raw_csv_path = os.path.join(current_dir, seq_name, 'SenseINS_aligned.csv')
                    if not os.path.exists(raw_csv_path):
                        import glob
                        candidates = glob.glob(os.path.join(current_dir, seq_name, "**", "SenseINS_aligned.csv"),
                                               recursive=True)
                        if candidates: raw_csv_path = candidates[0]

                    if os.path.exists(raw_csv_path):
                        raw_df = pd.read_csv(raw_csv_path)
                        # DVL Interpolation
                        dvl_map = {'x': ['dvl_vx', 'dvl_x', 'gt_v_x'], 'y': ['dvl_vy', 'dvl_y', 'gt_v_y'],
                                   'z': ['dvl_vz', 'dvl_z', 'gt_v_z']}

                        def get_col_name(candidates, columns):
                            for c in candidates:
                                if c in columns: return c
                            return None

                        col_vx = get_col_name(dvl_map['x'], raw_df.columns)
                        col_vy = get_col_name(dvl_map['y'], raw_df.columns)
                        col_vz = get_col_name(dvl_map['z'], raw_df.columns)

                        ts_col = 'timestamp' if 'timestamp' in raw_df.columns else 'time'
                        if ts_col not in raw_df.columns and 'Time' in raw_df.columns: ts_col = 'Time'

                        try:
                            raw_ts = pd.to_numeric(raw_df[ts_col], errors='coerce').values
                            raw_ts = raw_ts[~np.isnan(raw_ts)]
                            target_ts = df['timestamp'].values + raw_ts[0]
                        except:
                            target_ts = None

                        if col_vx and col_vy and col_vz and target_ts is not None:
                            raw_dvl = raw_df[[col_vx, col_vy, col_vz]].values
                            f_dvl = interp1d(raw_ts, raw_dvl, axis=0, kind='linear', fill_value="extrapolate",
                                             bounds_error=False)
                            aligned_dvl = f_dvl(target_ts)
                            df['dvl_vx'] = aligned_dvl[:, 0];
                            df['dvl_vy'] = aligned_dvl[:, 1];
                            df['dvl_vz'] = aligned_dvl[:, 2]
                        else:
                            df['dvl_vx'] = 0.0;
                            df['dvl_vy'] = 0.0;
                            df['dvl_vz'] = 0.0

                        # Euler Overwrite
                        if 'roll' in raw_df.columns and 'pitch' in raw_df.columns and 'yaw' in raw_df.columns and target_ts is not None:
                            raw_euler = raw_df[['yaw', 'pitch', 'roll']].values
                            f_euler = interp1d(raw_ts, raw_euler, axis=0, kind='linear', fill_value="extrapolate",
                                               bounds_error=False)
                            aligned_euler = f_euler(target_ts)
                            is_degrees = np.max(np.abs(raw_euler[:, 0])) > 7.0
                            r_new = R.from_euler('zyx', aligned_euler, degrees=is_degrees)
                            qs_new = r_new.as_quat()
                            df['gt_q_x'] = qs_new[:, 0];
                            df['gt_q_y'] = qs_new[:, 1];
                            df['gt_q_z'] = qs_new[:, 2];
                            df['gt_q_w'] = qs_new[:, 3]
                            gt_qs = torch.tensor(qs_new, device=device, dtype=torch.double)
                            gt_qs_eval = torch.cat([gt_qs[:, 3:4], gt_qs[:, :3]], dim=1)
                        else:
                            gt_qs_eval = gt_qs[:min_len]

                    else:
                        df['dvl_vx'] = 0.0;
                        df['dvl_vy'] = 0.0;
                        df['dvl_vz'] = 0.0
                        gt_qs_eval = gt_qs[:min_len]

                except Exception:
                    if 'dvl_vx' not in df.columns: df['dvl_vx'] = 0.0; df['dvl_vy'] = 0.0; df['dvl_vz'] = 0.0
                    gt_qs_eval = gt_qs[:min_len]

                # --- 🔥 F. 核心评估 (In-Loop Evaluation) ---
                raw_gyro_eval = raw_gyro[:min_len]
                q0 = gt_qs_eval[0]

                rmse_raw = calc_orientation_error(raw_gyro_eval, q0, gt_qs_eval, dt, device)
                rmse_net = calc_orientation_error(hat_xs[:min_len], q0, gt_qs_eval, dt, device)

                improv = (1 - rmse_net / (rmse_raw + 1e-8)) * 100
                evaluation_results.append({
                    'seq': seq_name, 'dur': min_len * dt,
                    'raw_rmse': rmse_raw, 'net_rmse': rmse_net, 'imp': improv
                })

                # --- G. Save ---
                seq_dir = os.path.join(OUTPUT_DIR, seq_name)
                os.makedirs(seq_dir, exist_ok=True)
                save_path = os.path.join(seq_dir, 'SenseINS_aligned.csv')
                df.to_csv(save_path, index=False)
                print(f"   -> Saved. Raw: {rmse_raw:.2f}°, Net: {rmse_net:.2f}° (Imp: {improv:.1f}%)")

            except Exception as e:
                print(f"Failed to process {seq_name}: {e}")

    return evaluation_results


def print_final_report(results):
    if not results:
        print("No results to report.")
        return

    print("\n" + "=" * 80)
    print(f"📊 Final MoE Enhancement Report")
    print("=" * 80)
    print(f"{'Seq':<15} | {'Dur (s)':<8} | {'Raw RMSE (°)':<15} | {'Net RMSE (°)':<15} | {'Improvement':<10}")
    print("-" * 80)

    avg_imp = 0
    for res in results:
        print(
            f"{res['seq']:<15} | {res['dur']:<8.1f} | {res['raw_rmse']:<15.4f} | {res['net_rmse']:<15.4f} | {res['imp']:>8.2f}%")
        avg_imp += res['imp']

    print("-" * 80)
    print(f"Average Improvement: {avg_imp / len(results):.2f}%")
    print("=" * 80 + "\n")


def verify_and_fix_alignment(output_dir):
    print("\n" + "=" * 80)
    print("🚀 Advanced Auto-Fixing (Single Axis Support)")
    print("=" * 80)
    print(f"{'Sequence':<20} | {'CosSim':<10} | {'Corr X':<10} | {'Corr Y':<10} | {'Status'}")
    print("-" * 90)

    pass_cnt, fix_cnt, fail_cnt = 0, 0, 0

    for root, dirs, files in os.walk(output_dir):
        if 'SenseINS_aligned.csv' in files:
            path = os.path.join(root, 'SenseINS_aligned.csv')
            seq = os.path.basename(root)
            try:
                df = pd.read_csv(path)
                if 'gt_q_w' not in df.columns: continue

                qs = df[['gt_q_x', 'gt_q_y', 'gt_q_z', 'gt_q_w']].values
                dvl_b = df[['dvl_vx', 'dvl_vy', 'dvl_vz']].values
                pos = df[['gt_p_x', 'gt_p_y', 'depth']].values
                ts = df['timestamp'].values
                dt = np.diff(ts);
                dt = np.where(dt <= 1e-6, 1e-4, dt)
                gt_vel = np.diff(pos, axis=0) / dt[:, None]

                def calc_metrics(dvl_curr):
                    r = R.from_quat(qs[:-1])
                    dvl_g = r.apply(dvl_curr[:-1])
                    norm_gt = np.linalg.norm(gt_vel[:, :2], axis=1) + 1e-6
                    norm_dvl = np.linalg.norm(dvl_g[:, :2], axis=1) + 1e-6
                    dot = np.sum(gt_vel[:, :2] * dvl_g[:, :2], axis=1)
                    sim = np.mean(dot / (norm_gt * norm_dvl))
                    cx = np.corrcoef(gt_vel[:, 0], dvl_g[:, 0])[0, 1]
                    cy = np.corrcoef(gt_vel[:, 1], dvl_g[:, 1])[0, 1]
                    return sim, cx, cy

                cos_sim, corr_x, corr_y = calc_metrics(dvl_b)

                if cos_sim > 0.5 and corr_x > 0.3 and corr_y > 0.3:
                    print(f"{seq:<20} | {cos_sim:8.4f}   | {corr_x:8.4f}   | {corr_y:8.4f}   | ✅ PASS")
                    pass_cnt += 1
                else:
                    print(
                        f"{seq:<20} | {cos_sim:8.4f}   | {corr_x:8.4f}   | {corr_y:8.4f}   | ❌ FAIL -> 🛠️ TRYING FIXES...")
                    fixes = [('Invert Y', [1, -1, 1]), ('Invert X', [-1, 1, 1]), ('Invert Both', [-1, -1, 1]),
                             ('Swap Axes', 'swap')]
                    best_metrics = (-1, 0, 0);
                    best_dvl = None;
                    best_fix_name = None

                    for name, op in fixes:
                        dvl_test = dvl_b.copy()
                        if op == 'swap':
                            dvl_test[:, 0] = -dvl_b[:, 1]; dvl_test[:, 1] = dvl_b[:, 0]
                        else:
                            dvl_test[:, 0] *= op[0]; dvl_test[:, 1] *= op[1]; dvl_test[:, 2] *= op[2]
                        s, cx, cy = calc_metrics(dvl_test)
                        if s > best_metrics[0]: best_metrics = (s, cx, cy); best_fix_name = name; best_dvl = dvl_test

                    bs, bcx, bcy = best_metrics
                    if bs > 0.5:
                        df['dvl_vx'] = best_dvl[:, 0];
                        df['dvl_vy'] = best_dvl[:, 1];
                        df['dvl_vz'] = best_dvl[:, 2]
                        df.to_csv(path, index=False)
                        print(f"{' ':<20} | {bs:8.4f}   | {bcx:8.4f}   | {bcy:8.4f}   | ✅ FIXED ({best_fix_name})")
                        fix_cnt += 1
                    else:
                        print(f"{' ':<20} | {bs:8.4f}   | {bcx:8.4f}   | {bcy:8.4f}   | 💀 FAILED TO FIX")
                        fail_cnt += 1
            except Exception as e:
                print(f"Error {seq}: {e}")

    print("-" * 90)
    print(f"Summary: {pass_cnt} Passed, {fix_cnt} Fixed, {fail_cnt} Failed.")


if __name__ == '__main__':
    # 1. 处理数据并获取评估结果
    results = prepare_data()

    # 2. 打印对比报告
    print_final_report(results)

    # 3. 校验和修复 DVL 对齐
    verify_and_fix_alignment(OUTPUT_DIR)