import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from src.utils import pload
from src.lie_algebra import SO3

# ==============================================================================
# 配置区域
# ==============================================================================
RES_DIR = r'E:\results\EUROC'
DATA_DIR = os.path.join('src', 'data', 'MY_DATA')
TARGET_ADDRESS = "last"
TEST_SEQS = ['7']
DT = 0.01  # 100Hz

# ROE 评估的时间间隔 (单位: 秒)
ROE_INTERVALS_SEC = [1, 5, 10, 20, 30]


# ==============================================================================
# 核心工具函数
# ==============================================================================

def get_latest_result_dir(base_dir):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"找不到结果目录: {base_dir}")
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        raise FileNotFoundError("结果目录下没有任何训练记录")
    dirs.sort(key=lambda x: os.path.getmtime(os.path.join(base_dir, x)))
    return os.path.join(base_dir, dirs[-1])


def compute_angle_diff(R_a, R_b):
    """
    计算两组旋转矩阵之间的角度差 (batch)
    theta = | arccos((tr(R_a^T * R_b) - 1) / 2) |
    """
    if R_a.device != R_b.device:
        R_b = R_b.to(R_a.device)

    R_diff = torch.bmm(R_a.transpose(1, 2), R_b)

    tr = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_theta = (tr - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta) * (180 / np.pi)
    return theta


def compute_roe(R_est, R_gt, interval_frames):
    """
    计算相对姿态误差 (ROE) - Relative Orientation Error
    论文公式: || log( delta_R_gt^T * delta_R_est ) ||
    即计算: (R_i^T * R_{i+k}) 与 (R_hat_i^T * R_hat_{i+k}) 之间的误差
    """
    N = R_est.shape[0]
    if interval_frames >= N:
        return np.array([])  # 间隔太长，无法计算

    # 1. 提取起点和终点
    # R_start: 0 到 N-k
    # R_end:   k 到 N
    R_est_start = R_est[:-interval_frames]
    R_est_end = R_est[interval_frames:]

    R_gt_start = R_gt[:-interval_frames]
    R_gt_end = R_gt[interval_frames:]

    # 2. 计算相对旋转 delta_R = R_start^T * R_end
    # delta_R_est: (N-k, 3, 3)
    delta_R_est = torch.bmm(R_est_start.transpose(1, 2), R_est_end)
    delta_R_gt = torch.bmm(R_gt_start.transpose(1, 2), R_gt_end)

    # 3. 计算两者之间的误差角度
    # error = angle(delta_R_est, delta_R_gt)
    roe_errors = compute_angle_diff(delta_R_est, delta_R_gt)

    return roe_errors.cpu().numpy()


def integrate_trajectory(us, dt, init_rot):
    N = us.shape[0]
    device = us.device
    d_rot_vec = us * dt
    d_qs = SO3.qexp(d_rot_vec)

    r = R.from_matrix(init_rot.cpu().numpy())
    q_curr = torch.tensor(r.as_quat(), device=device, dtype=us.dtype)[[3, 0, 1, 2]].unsqueeze(0)

    qs_list = []
    curr = q_curr
    for i in range(N):
        curr = SO3.qmul(curr, d_qs[i:i + 1])
        curr = SO3.qnorm(curr)
        qs_list.append(curr)

    qs_all = torch.cat(qs_list, dim=0)
    return SO3.from_quaternion(qs_all, ordering='wxyz')


def get_euler_unwrapped(rot_matrices):
    r = R.from_matrix(rot_matrices.cpu().numpy())
    euler_rad = r.as_euler('zyx', degrees=False)
    euler_unwrapped = np.unwrap(euler_rad, axis=0)
    return np.rad2deg(euler_unwrapped)[:, [2, 1, 0]]  # [Roll, Pitch, Yaw]


# ==============================================================================
# 主逻辑
# ==============================================================================

def evaluate_network():
    if TARGET_ADDRESS == "last":
        result_path = get_latest_result_dir(RES_DIR)
    else:
        result_path = os.path.join(RES_DIR, TARGET_ADDRESS)

    print("=" * 60)
    print(f"🚀 开始全方位评估 (AOE & ROE)")
    print(f"📂 结果路径: {result_path}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for seq in TEST_SEQS:
        print(f"\n>>> 处理序列: {seq}")

        # --- 加载数据 ---
        res_file = os.path.join(result_path, seq, 'results.p')
        raw_path = os.path.join(DATA_DIR, f"{seq}.p")
        gt_path = os.path.join(DATA_DIR, f"{seq}_gt.p")

        if not (os.path.exists(res_file) and os.path.exists(raw_path) and os.path.exists(gt_path)):
            print("❌ 文件缺失，跳过。")
            continue

        res_data = pload(res_file)
        raw_data = pload(raw_path)
        gt_data = pload(gt_path)

        net_gyro = res_data['hat_xs'].to(device).double()
        raw_gyro = raw_data['us'][:, :3].to(device).double()
        gt_qs = gt_data['qs'].to(device).double()
        gt_rots = SO3.from_quaternion(gt_qs, ordering='wxyz')

        # 对齐长度
        min_len = min(net_gyro.shape[0], raw_gyro.shape[0], gt_rots.shape[0])
        net_gyro, raw_gyro, gt_rots = net_gyro[:min_len], raw_gyro[:min_len], gt_rots[:min_len]

        # 积分
        print("    正在进行姿态积分...")
        init_rot = gt_rots[0]
        traj_raw = integrate_trajectory(raw_gyro, DT, init_rot)
        traj_net = integrate_trajectory(net_gyro, DT, init_rot)

        # =================================================================
        # 1. 计算 AOE (Absolute Orientation Error)
        # =================================================================
        err_raw_aoe = compute_angle_diff(traj_raw, gt_rots).cpu().numpy()
        err_net_aoe = compute_angle_diff(traj_net, gt_rots).cpu().numpy()

        ate_raw = np.sqrt(np.mean(err_raw_aoe ** 2))
        ate_net = np.sqrt(np.mean(err_net_aoe ** 2))
        imp_aoe = (ate_raw - ate_net) / ate_raw * 100

        print(f"    [AOE 结果] RMSE: Raw={ate_raw:.2f}°, Net={ate_net:.2f}° (提升 {imp_aoe:.1f}%)")

        # =================================================================
        # 2. 计算 ROE (Relative Orientation Error)
        # =================================================================
        print("    正在计算 ROE (相对误差)...")
        roe_data_net = []
        roe_data_raw = []
        roe_labels = []

        for sec in ROE_INTERVALS_SEC:
            k = int(sec / DT)  # 帧数
            if k >= min_len: continue

            # 计算该间隔下的所有误差样本
            errs_net = compute_roe(traj_net, gt_rots, k)
            errs_raw = compute_roe(traj_raw, gt_rots, k)

            if len(errs_net) > 0:
                roe_data_net.append(errs_net)
                roe_data_raw.append(errs_raw)
                roe_labels.append(f"{sec}s")

        # =================================================================
        # 3. 绘图展示
        # =================================================================

        # --- 图1: AOE 随时间变化 ---
        time_axis = np.arange(min_len) * DT
        plt.figure(figsize=(10, 5))
        plt.plot(time_axis, err_raw_aoe, 'gray', alpha=0.5, label='Raw IMU')
        plt.plot(time_axis, err_net_aoe, 'r-', linewidth=1.5, label='GyroNet')
        plt.title(f'Seq {seq}: Absolute Orientation Error (AOE)\nRMSE: {ate_net:.2f}° (Imp: {imp_aoe:.1f}%)')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (deg)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(result_path, seq, 'eval_AOE.png'))
        plt.close()

        # --- 图2: ROE 箱线图 (核心图表) ---
        plt.figure(figsize=(10, 6))
        # 绘制 Raw IMU (灰色虚线箱体)
        bp1 = plt.boxplot(roe_data_raw, positions=np.arange(len(roe_labels)) * 2.0 - 0.4, widths=0.6,
                          patch_artist=True, boxprops=dict(facecolor='lightgray', alpha=0.5), showfliers=False)
        # 绘制 Net IMU (红色实线箱体)
        bp2 = plt.boxplot(roe_data_net, positions=np.arange(len(roe_labels)) * 2.0 + 0.4, widths=0.6,
                          patch_artist=True, boxprops=dict(facecolor='pink', alpha=0.8), showfliers=False)

        plt.xticks(np.arange(len(roe_labels)) * 2.0, roe_labels)
        plt.title(f'Seq {seq}: Relative Orientation Error (ROE) Distribution')
        plt.xlabel('Time Interval (Delta t)')
        plt.ylabel('Relative Error (deg)')
        plt.grid(True, alpha=0.3, axis='y')

        # 自定义图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', lw=4, label='Raw IMU', alpha=0.5),
            Line2D([0], [0], color='pink', lw=4, label='GyroNet')
        ]
        plt.legend(handles=legend_elements)

        plt.savefig(os.path.join(result_path, seq, 'eval_ROE_boxplot.png'))
        plt.close()

        # --- 图3: 轨迹对比 (无跳变) ---
        euler_gt = get_euler_unwrapped(gt_rots)
        euler_net = get_euler_unwrapped(traj_net)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ['Roll', 'Pitch', 'Yaw']
        for i in range(3):
            axes[i].plot(time_axis, euler_gt[:, i], 'k--', label='GT')
            axes[i].plot(time_axis, euler_net[:, i], 'b-', label='Net')
            axes[i].set_ylabel(f'{labels[i]} (deg)')
            axes[i].grid(True, alpha=0.3)
            if i == 0: axes[i].legend()
        plt.xlabel('Time (s)')
        plt.suptitle(f'Seq {seq}: Unwrapped Trajectory')
        plt.savefig(os.path.join(result_path, seq, 'eval_traj_unwrapped.png'))
        plt.close()

        print(f"    ✅ 图片已生成: AOE, ROE_boxplot, Trajectory")


if __name__ == "__main__":
    evaluate_network()