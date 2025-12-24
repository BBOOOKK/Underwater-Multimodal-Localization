import os
import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys

# 引入项目模块
# 确保 src 在 python 路径中
sys.path.append(os.getcwd())
from src.utils import pload, pdump, bmtm
from src.lie_algebra import SO3
import src.networks as sn

# ================= 配置区域 =================
# 1. 输入文件路径 (请确保文件名正确)
INPUT_CSV = "E:/水下导航资料/xiachi/1/aligned_dataset.csv"

# 2. 模型结果目录
RES_DIR = r'E:\results\EUROC'
# 设置为 "last" 自动找最新的，或者指定具体文件夹名
TARGET_ADDRESS = "2025_12_16_22_31_28"

# 3. 输出数据保存位置
DATA_OUTPUT_DIR = os.path.join("src", "data", "MY_DATA")
SEQ_NAME = "custom_test"  # 给这个新数据起个名字

# 4. 参数设置 (必须与训练一致)
DT = 0.01
MIN_TRAIN_FREQ = 16
CONVERT_DEG_TO_RAD = False


# ===========================================

def get_latest_result_dir(base_dir):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"找不到结果目录: {base_dir}")
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        raise FileNotFoundError("结果目录下没有任何训练记录")
    dirs.sort()
    return os.path.join(base_dir, dirs[-1])


def process_csv_to_pickle():
    """将 CSV 转换为 .p 文件"""
    print(f">>> 正在处理数据: {INPUT_CSV}")

    if not os.path.exists(INPUT_CSV):
        print(f"❌ 错误: 找不到文件 {INPUT_CSV}，请确认它在项目根目录下。")
        return False

    df = pd.read_csv(INPUT_CSV)

    try:
        # 1. 提取传感器数据
        gx = df['gx'].values
        gy = df['gy'].values
        gz = df['gz'].values
        ax = df['ax'].values
        ay = df['ay'].values
        az = df['az'].values
        ts = df['timestamp'].values

        # 2. 提取真值数据
        gt_px = df['gt_px'].values
        gt_py = df['gt_py'].values
        gt_pz = df['gt_pz'].values
        gt_qx = df['gt_qx'].values
        gt_qy = df['gt_qy'].values
        gt_qz = df['gt_qz'].values
        gt_qw = df['gt_qw'].values

    except KeyError as e:
        print(f"❌ 列名错误: {e}")
        return False

    # 单位转换
    if CONVERT_DEG_TO_RAD:
        gx, gy, gz = gx * np.pi / 180, gy * np.pi / 180, gz * np.pi / 180

    # ---------------------------
    # 构造 Tensor
    # ---------------------------
    # us: (N, 6) [gx, gy, gz, ax, ay, az]
    gyro = torch.tensor(np.stack([gx, gy, gz], axis=1)).float()
    acc = torch.tensor(np.stack([ax, ay, az], axis=1)).float()
    us = torch.cat([gyro, acc], dim=1)

    # qs: (N, 4) [w, x, y, z]
    # 原始数据是 [x, y, z, w] -> 调整为 [w, x, y, z]
    q_xyzw = torch.tensor(np.stack([gt_qx, gt_qy, gt_qz, gt_qw], axis=1)).double()
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    q_wxyz = SO3.qnorm(q_wxyz)

    # 计算 xs (相对旋转增量, 用于 loss 或 debug, 虽然推理不需要但保持格式一致)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    R_all = SO3.from_quaternion(q_wxyz.to(device), ordering='wxyz')

    mtf = MIN_TRAIN_FREQ
    if R_all.shape[0] <= mtf:
        print("❌ 数据太短")
        return False

    R_i = R_all[:-mtf]
    R_j = R_all[mtf:]
    dRot = bmtm(R_i, R_j)
    dRot = SO3.dnormalize(dRot)
    xs = SO3.log(dRot).cpu().float()

    # ---------------------------
    # 保存文件
    # ---------------------------
    if not os.path.exists(DATA_OUTPUT_DIR):
        os.makedirs(DATA_OUTPUT_DIR)

    # 保存主数据
    train_dict = {'us': us, 'xs': xs, 't': ts}
    save_path = os.path.join(DATA_OUTPUT_DIR, f"{SEQ_NAME}.p")
    with open(save_path, 'wb') as f:
        pickle.dump(train_dict, f)

    # 保存真值数据
    gt_dict = {
        'ts': ts,
        'qs': q_wxyz.float().cpu(),
        'ps': torch.tensor(np.stack([gt_px, gt_py, gt_pz], axis=1)).float()
    }
    gt_save_path = os.path.join(DATA_OUTPUT_DIR, f"{SEQ_NAME}_gt.p")
    with open(gt_save_path, 'wb') as f:
        pickle.dump(gt_dict, f)

    print(f"✅ 数据已转换并保存至: {save_path}")
    return True


def compute_angle_error(R_est, R_gt):
    """计算角度误差"""
    R_diff = torch.bmm(R_est, R_gt.transpose(1, 2))
    tr = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_theta = (tr - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta) * (180 / np.pi)
    return theta


def integrate_trajectory(us, dt, init_rot):
    """积分轨迹"""
    N = us.shape[0]
    device = us.device
    d_rot_vec = us * dt
    d_qs = SO3.qexp(d_rot_vec)

    # 初始四元数 (wxyz)
    r = R.from_matrix(init_rot.cpu().numpy())
    q_init = torch.tensor(r.as_quat(), device=device, dtype=us.dtype).unsqueeze(0)
    q_init = q_init[:, [3, 0, 1, 2]]  # xyzw -> wxyz

    d_qs_wxyz = torch.cat([d_qs[:, 0:1], d_qs[:, 1:4]], dim=1)

    qs_all = torch.zeros(N, 4, device=device)
    curr = q_init

    # 简单的累乘积分
    for i in range(N):
        curr = SO3.qmul(curr, d_qs_wxyz[i:i + 1])
        curr = SO3.qnorm(curr)
        qs_all[i] = curr

    return SO3.from_quaternion(qs_all, ordering='wxyz')


def run_test():
    # 1. 处理数据
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    if not process_csv_to_pickle():
        return

    # 2. 加载模型
    if TARGET_ADDRESS == "last":
        model_dir = get_latest_result_dir(RES_DIR)
    else:
        model_dir = os.path.join(RES_DIR, TARGET_ADDRESS)

    print(f">>> 加载模型: {model_dir}")

    # 加载网络参数
    net_params = pload(model_dir, 'net_params.p')
    # 实例化网络
    net = sn.GyroNet(**net_params)
    # 加载权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.load(os.path.join(model_dir, 'weights.pt'), map_location=device)
    net.load_state_dict(weights)
    net.to(device)
    net.eval()

    # 3. 读取数据
    data_path = os.path.join(DATA_OUTPUT_DIR, f"{SEQ_NAME}.p")
    gt_path = os.path.join(DATA_OUTPUT_DIR, f"{SEQ_NAME}_gt.p")

    raw_data = pload(data_path)
    gt_data = pload(gt_path)

    us = raw_data['us'].to(device).unsqueeze(0)  # (1, N, 6)
    gt_qs = gt_data['qs'].to(device)  # (N, 4)

    # 4. 推理
    print(">>> 正在进行推理...")
    with torch.no_grad():
        hat_xs = net(us)  # (1, N, 3) 输出修正后的角速度(或增量，取决于网络定义)
        # GyroNet 直接输出去噪后的角速度
        net_gyro = hat_xs[0]

    raw_gyro = raw_data['us'][:, :3].to(device)

    # 截取对齐长度
    min_len = min(net_gyro.shape[0], gt_qs.shape[0])
    net_gyro = net_gyro[:min_len]
    raw_gyro = raw_gyro[:min_len]
    gt_qs = gt_qs[:min_len]

    # 5. 积分评估
    print(">>> 正在进行积分评估...")
    gt_rots = SO3.from_quaternion(gt_qs, ordering='wxyz')
    init_rot = gt_rots[0]

    traj_raw = integrate_trajectory(raw_gyro, DT, init_rot)
    traj_net = integrate_trajectory(net_gyro, DT, init_rot)

    # 6. 计算误差
    err_raw = compute_angle_error(traj_raw, gt_rots).cpu().numpy()
    err_net = compute_angle_error(traj_net, gt_rots).cpu().numpy()

    ate_raw = np.sqrt(np.mean(err_raw ** 2))
    ate_net = np.sqrt(np.mean(err_net ** 2))
    end_raw = err_raw[-1]
    end_net = err_net[-1]

    duration = (min_len * DT) / 60.0  # min
    drift_raw = end_raw / duration
    drift_net = end_net / duration
    imp = (ate_raw - ate_net) / ate_raw * 100

    print("\n" + "=" * 60)
    print(f"测试文件: {INPUT_CSV} (时长 {duration:.2f} min)")
    print("=" * 60)
    print(f"{'指标':<20} | {'原始 (Raw)':<15} | {'去噪后 (Net)':<15} | {'提升'}")
    print("-" * 60)
    print(f"{'ATE RMSE (deg)':<20} | {ate_raw:<15.4f} | {ate_net:<15.4f} | {imp:.2f}%")
    print(f"{'End Error (deg)':<20} | {end_raw:<15.4f} | {end_net:<15.4f} |")
    print(f"{'Drift (deg/min)':<20} | {drift_raw:<15.4f} | {drift_net:<15.4f} |")
    print("=" * 60)

    # 7. 画图
    time_axis = np.arange(min_len) * DT / 60.0
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, err_raw, 'r-', label=f'Raw (ATE={ate_raw:.2f})', alpha=0.5)
    plt.plot(time_axis, err_net, 'b-', label=f'Net (ATE={ate_net:.2f})', linewidth=2)
    plt.title(f"Test Result: {INPUT_CSV}\nImprovement: {imp:.1f}%")
    plt.xlabel("Time (min)")
    plt.ylabel("Error (deg)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    save_img = "test_custom_result.png"
    plt.savefig(save_img)
    print(f"\n结果图已保存: {save_img}")
    plt.show()


if __name__ == "__main__":
    run_test()