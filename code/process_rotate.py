"""
ROV 航位推算数据处理脚本 (IMU + DVL + Ground Truth)
功能：
1. 读取 IMU、DVL 及真值(GT) 数据并进行时间戳对齐 (UTC+8)。
2. 清洗 DVL 异常跳变值 (Error Code Removal)。
3. 基于传感器安装矩阵进行坐标系转换与航位推算。
4. 自动/手动对齐初始航向角，生成平稳的轨迹图并保存结果。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import os

# ==============================================================================
# 1. 全局配置区域 (Configuration)
# ==============================================================================

SUB_DIR = "01"  # 实验数据集子目录名称
BASE_DIR = r"D:/A_underwater_study/data/0730shuichi"
INPUT_DIR = os.path.join(BASE_DIR, SUB_DIR)

# 输入文件路径配置
FILE_IMU = os.path.join(INPUT_DIR, "imu.csv")
FILE_DVL = os.path.join(INPUT_DIR, "dvl.csv")
FILE_GT = os.path.join(INPUT_DIR, "seg1.csv")  # 这里在当前数据集中也需要改

# 输出文件路径配置
FILE_OUT_CSV = os.path.join(BASE_DIR, f"processed_results_cut/{SUB_DIR}.csv")
FILE_OUT_IMG = os.path.join(BASE_DIR, f"processed_results_cut/{SUB_DIR}.png")

# --- 算法参数设置 ---
MANUAL_OFFSET = -20  # 手动航向角偏移量 (deg)，用于手动修正系统偏差
ENABLE_AUTO_ALIGN = True  # 是否启用基于真值自动计算初始航向偏差

# 传感器安装矩阵 (Alignment Matrix)
# 用于将 IMU 原始坐标系对齐到 ROV 机体坐标系 (Body Frame)
ALIGN_MATRIX = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, -1]
])


# ==============================================================================
# 2. 数据读取与预处理 (Data Loading & Cleaning)
# ==============================================================================

def load_data():
    """
    从指定路径加载传感器数据，并执行初步的清洗与时间戳转换。
    Returns:
        tuple: (df_imu, df_dvl, df_gt) 处理后的 DataFrame
    """
    print(f"[*] 正在读取数据集: {SUB_DIR} ...")

    # --- IMU 数据处理 ---
    cols_imu = ["time", "roll", "pitch", "yaw", "accX", "accY", "accZ", "gyrX", "gyrY", "gyrZ"]
    try:
        df_imu = pd.read_csv(FILE_IMU, usecols=lambda c: c in cols_imu)
    except:
        df_imu = pd.read_excel(FILE_IMU.replace('.csv', '.xlsx'))

    df_imu["time"] = pd.to_datetime(df_imu["time"])
    # 时间戳转换：将北京时间转为 Unix 时间戳 (减去 8 小时偏移)
    df_imu["timestamp"] = df_imu["time"].astype('int64') / 1e9 - 28800
    df_imu = df_imu.sort_values("timestamp").dropna()

    # --- DVL 数据处理 (包含异常码过滤) ---
    cols_dvl = ["time", "Vx", "Vy", "Vz", "d"]
    try:
        df_dvl = pd.read_csv(FILE_DVL, usecols=lambda c: c in cols_dvl)
        df_dvl["time"] = pd.to_datetime(df_dvl["time"])
        df_dvl["timestamp"] = df_dvl["time"].astype('int64') / 1e9 - 28800

        # DVL 物理阈值清洗：ROV 运动通常不超过 5m/s，过滤掉 88.888 等错误码
        MAX_VELOCITY = 5.0
        for c in ["Vx", "Vy", "Vz"]:
            mask_error = df_dvl[c].abs() > MAX_VELOCITY
            if mask_error.sum() > 0:
                print(f"    [!] 清洗: 检出 {mask_error.sum()} 个异常速度点 ({c} 轴), 已设为无效值。")
            df_dvl.loc[mask_error, c] = np.nan

        # 深度/底距清洗：排除超出量程的数据 (如 88.888)
        if "d" in df_dvl.columns:
            mask_d_error = (df_dvl["d"] > 100) | (df_dvl["d"] < 0)
            df_dvl.loc[mask_d_error, ["Vx", "Vy", "Vz", "d"]] = np.nan

        # 线性插值补齐：填充短时间内的信号丢失，防止积分中断
        LIMIT_COUNT = 5  # 连续丢失超过 5 个点则不再补全
        for c in ["Vx", "Vy", "Vz", "d"]:
            df_dvl[c] = df_dvl[c].interpolate(method='linear', limit=LIMIT_COUNT, limit_direction='both')
            df_dvl[c] = df_dvl[c].fillna(0.0)  # 无法补全的部分设为 0

        df_dvl = df_dvl.dropna(subset=["timestamp"])
    except Exception as e:
        print(f"[X] DVL 读取失败: {e}")
        df_dvl = pd.DataFrame(columns=["timestamp", "Vx", "Vy", "Vz"])

    # --- GT 真值数据处理 ---
    try:
        df_gt = pd.read_csv(FILE_GT)
        rename = {'x': 'px', 'y': 'py', 'z': 'pz', 'position_x': 'px', 'position_y': 'py'}
        df_gt.rename(columns=rename, inplace=True)
        df_gt = df_gt.dropna(subset=['px', 'py'])
        if 'time' in df_gt.columns:
            df_gt["time"] = pd.to_datetime(df_gt["time"])
            df_gt["timestamp"] = df_gt["time"].astype('int64') / 1e9 - 28800
        df_gt = df_gt.sort_values("timestamp")
    except:
        df_gt = pd.DataFrame(columns=["timestamp", "px", "py"])

    return df_imu, df_dvl, df_gt


# ==============================================================================
# 3. 核心计算模块 (Core Processing)
# ==============================================================================

def calculate_auto_heading_offset(est_x, est_y, gt_x, gt_y):
    """
    计算估计轨迹与真值轨迹之间的主航向角度偏差。
    利用轨迹终点相对于起点的向量夹角进行计算。
    """
    valid_len = min(len(est_x), len(gt_x))
    if valid_len < 10: return 0.0

    end_idx = valid_len - 1
    avg_window = max(1, int(valid_len * 0.1))  # 取最后 10% 的点求均值，提高稳定性

    # 计算真值航向向量角度
    dx_gt = np.mean(gt_x[end_idx - avg_window: end_idx]) - gt_x[0]
    dy_gt = np.mean(gt_y[end_idx - avg_window: end_idx]) - gt_y[0]
    angle_gt = np.arctan2(dy_gt, dx_gt)

    # 计算估计航向向量角度
    dx_est = np.mean(est_x[end_idx - avg_window: end_idx]) - est_x[0]
    dy_est = np.mean(est_y[end_idx - avg_window: end_idx]) - est_y[0]
    angle_est = np.arctan2(dy_est, dx_est)

    return np.degrees(angle_gt - angle_est)


def process_and_plot(df_imu, df_dvl, df_gt):
    """
    核心流水线：时间对齐 -> 轨迹推算 -> 坐标旋转 -> 结果可视化。
    """
    print("[*] 正在执行航位推算与重叠截断...")

    # --- 步骤 1: 时间戳重叠截断 (Overlap Truncation) ---
    # 寻找所有数据源的公共时间交集，确保积分起始时间一致
    start_times = [df_imu["timestamp"].min(), df_dvl["timestamp"].min()]
    end_times = [df_imu["timestamp"].max(), df_dvl["timestamp"].max()]

    has_gt = not df_gt.empty
    if has_gt:
        start_times.append(df_gt["timestamp"].min())
        end_times.append(df_gt["timestamp"].max())

    t_start, t_end = max(start_times) + 0.1, min(end_times) - 0.1

    if t_end <= t_start:
        print("[X] 错误: 数据源之间没有重叠的时间段，请检查时间戳。")
        return

    # 截断数据
    df = df_imu[(df_imu["timestamp"] >= t_start) & (df_imu["timestamp"] <= t_end)].copy()
    df_dvl = df_dvl[(df_dvl["timestamp"] >= t_start) & (df_dvl["timestamp"] <= t_end)].copy()
    if has_gt:
        df_gt = df_gt[(df_gt["timestamp"] >= t_start) & (df_gt["timestamp"] <= t_end)].copy()

    # --- 步骤 2: 数据对齐与插值 ---
    # 将 DVL 速度插值到 IMU 的高频时间轴上
    f_dvl = interp1d(df_dvl["timestamp"], df_dvl[["Vx", "Vy", "Vz"]],
                     axis=0, bounds_error=False, fill_value=0.0)
    dvl_val = f_dvl(df["timestamp"])
    df["dvl_vx"], df["dvl_vy"], df["dvl_vz"] = dvl_val[:, 0], dvl_val[:, 1], dvl_val[:, 2]

    # 【重要】坐标系修正：根据 DVL 安装方向调整 Vy (如 RTK 与 DVL 定义相反需取负)
    df["dvl_vy"] = -df["dvl_vy"]

    # --- 步骤 3: 航位推算 (Dead Reckoning) ---
    # 使用旋转矩阵将 DVL 速度从机体坐标系转到世界坐标系并积分
    R_sensor = R.from_euler('xyz', df[["roll", "pitch", "yaw"]].values, degrees=True)
    R_align = R.from_matrix(ALIGN_MATRIX)
    R_body = R_sensor * R_align  # 融合安装偏差后的机体姿态

    vel_body = df[["dvl_vx", "dvl_vy", "dvl_vz"]].values
    dt = np.diff(df["timestamp"], prepend=df["timestamp"].iloc[0])
    dt[0] = np.mean(dt[1:])  # 修复首帧 dt 异常

    pos_est = np.zeros((len(df), 3))
    for i in range(1, len(df)):
        # P_n = P_n-1 + R * V_body * dt
        pos_est[i] = pos_est[i - 1] + R_body[i].apply(vel_body[i]) * dt[i]

    est_x, est_y = pos_est[:, 0], pos_est[:, 1]

    # --- 步骤 4: 真值插值与对齐 ---
    gt_x, gt_y = np.nan, np.nan
    if has_gt:
        f_gt = interp1d(df_gt["timestamp"], df_gt[["px", "py"]],
                        axis=0, bounds_error=False, fill_value=np.nan)
        gt_interp = f_gt(df["timestamp"])
        # 真值归零化：以交集第一帧为坐标原点
        valid_mask = ~np.isnan(gt_interp[:, 0])
        if np.sum(valid_mask) > 0:
            gt_x = gt_interp[:, 0] - gt_interp[valid_mask, 0][0]
            gt_y = gt_interp[:, 1] - gt_interp[valid_mask, 1][0]

    # --- 步骤 5: 初始航向角偏差修正 ---
    rotation_used = 0.0
    if MANUAL_OFFSET != 0:
        rotation_used = MANUAL_OFFSET
    elif ENABLE_AUTO_ALIGN and has_gt and np.sum(~np.isnan(gt_x)) > 10:
        rotation_used = calculate_auto_heading_offset(est_x, est_y, gt_x[~np.isnan(gt_x)], gt_y[~np.isnan(gt_y)])

    # 应用旋转修正
    est_x, est_y = est_x - est_x[0], est_y - est_y[0]
    if rotation_used != 0:
        rad = np.radians(rotation_used)
        c, s = np.cos(rad), np.sin(rad)
        est_x, est_y = c * est_x - s * est_y, s * est_x + c * est_y

    # ==============================================================================
    # 4. 结果整理与可视化 (Export & Visualization)
    # ==============================================================================

    # 更新数据表以便保存
    df["est_px_aligned"], df["est_py_aligned"] = est_x, est_y
    if has_gt:
        df["gt_px"], df["gt_py"] = gt_x, gt_y
        # 清洗掉真值尚未开始记录的行
        df = df.dropna(subset=["gt_px", "gt_py"])
        # 二次归零，确保 CSV 起点为 (0,0)
        for c in ["est_px_aligned", "est_py_aligned", "gt_px", "gt_py"]:
            df[c] = df[c] - df[c].iloc[0]

    # 保存数据
    save_cols = ["timestamp", "est_px_aligned", "est_py_aligned", "gt_px", "gt_py",
                 "dvl_vx", "dvl_vy", "dvl_vz", "roll", "pitch", "yaw"]
    df[[c for c in save_cols if c in df.columns]].to_csv(FILE_OUT_CSV, index=False)
    print(f"[√] 结果已导出至 CSV: {FILE_OUT_CSV}")

    # 绘制轨迹对比图
    plt.figure(figsize=(10, 10))
    if has_gt:
        plt.plot(df["gt_px"], df["gt_py"], 'r-', lw=2, label='Ground Truth (RTK)', alpha=0.7)
    plt.plot(df["est_px_aligned"], df["est_py_aligned"], 'b--', lw=2, label='Estimated (IMU+DVL)')
    plt.plot(0, 0, 'k*', ms=15, label='Start Point')

    plt.title(f"Trajectory Comparison: {SUB_DIR}\nRotation Correction: {rotation_used:.2f}°")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.savefig(FILE_OUT_IMG, dpi=300)
    print(f"[√] 轨迹图已保存: {FILE_OUT_IMG}")
    plt.show()


if __name__ == "__main__":
    try:
        imu_raw, dvl_raw, gt_raw = load_data()
        process_and_plot(imu_raw, dvl_raw, gt_raw)
    except Exception as e:
        print(f"[X] 程序运行出错: {e}")