import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import glob
import os

# 配置你的 CSV 路径 (请确认路径无误)
RAW_CSV_DIR = r"/ronin-net\data\processed_data"


def check_axis():
    print(f"🔍 正在搜索目录: {RAW_CSV_DIR}")
    # 递归搜索
    search_pattern = os.path.join(RAW_CSV_DIR, "**", "*.csv")
    csv_files = sorted(glob.glob(search_pattern, recursive=True))

    if not csv_files:
        print("❌ 没找到 CSV 文件")
        return

    # 只检查第一个文件
    target_file = csv_files[0]
    print(f"👀 正在检查文件: {os.path.basename(target_file)}")

    try:
        # 读取 CSV
        df = pd.read_csv(target_file)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # ==================================================================
    # 1. 智能计算采样时间 dt (修复 TypeError 问题的核心部分)
    # ==================================================================
    dt_mean = 0.01  # 默认值
    found_dt = False

    # 优先尝试 'time' 列 (通常是秒)
    if 'time' in df.columns:
        try:
            # 🔥 [关键修复] 强制转换为数字，非数字变成 NaN
            t = pd.to_numeric(df['time'], errors='coerce')
            t = t.dropna().values  # 去掉空值

            if len(t) > 1:
                diffs = np.diff(t)
                dt_temp = np.mean(diffs)
                if dt_temp > 0:
                    dt_mean = dt_temp
                    found_dt = True
                    print(f"⏳ [Time列] 检测到 dt = {dt_mean:.5f} s")
        except Exception as e:
            print(f"⚠️ 'time' 列解析失败: {e}")

    # 如果 'time' 失败，尝试 'timestamp' (可能是纳秒/毫秒)
    if not found_dt and 'timestamp' in df.columns:
        try:
            t = pd.to_numeric(df['timestamp'], errors='coerce')
            t = t.dropna().values
            if len(t) > 1:
                diffs = np.mean(np.diff(t))
                # 简单判断单位
                if diffs > 1e8:  # 纳秒 (10^9)
                    dt_mean = diffs / 1e9
                    print(f"⏳ [Timestamp列] 检测到纳秒单位，转换后 dt = {dt_mean:.5f} s")
                elif diffs > 1e5:  # 微秒 (10^6)
                    dt_mean = diffs / 1e6
                    print(f"⏳ [Timestamp列] 检测到微秒单位，转换后 dt = {dt_mean:.5f} s")
                elif diffs > 10:  # 毫秒 (10^3)
                    dt_mean = diffs / 1e3
                    print(f"⏳ [Timestamp列] 检测到毫秒单位，转换后 dt = {dt_mean:.5f} s")
                else:  # 秒
                    dt_mean = diffs
                    print(f"⏳ [Timestamp列] 检测到秒单位，dt = {dt_mean:.5f} s")
                found_dt = True
        except Exception:
            pass

    if not found_dt:
        print(f"⚠️ 未能自动识别时间列，使用默认 dt = {dt_mean} s")

    # ==================================================================
    # 2. 读取数据与计算真值
    # ==================================================================
    try:
        # 读取原始数据 (Raw)
        # 注意：这里按默认顺序读取，不做交换，不做取反，先看原始对比！
        my_gx = pd.to_numeric(df['gyro_x'], errors='coerce').values
        my_gy = pd.to_numeric(df['gyro_y'], errors='coerce').values
        my_gz = pd.to_numeric(df['gyro_z'], errors='coerce').values

        # 读取真值姿态 (GT)
        roll = pd.to_numeric(df['roll'], errors='coerce').values
        pitch = pd.to_numeric(df['pitch'], errors='coerce').values
        yaw = pd.to_numeric(df['yaw'], errors='coerce').values

        # 清洗数据：如果有 NaN，这行会报错，所以我们需要对齐长度
        # 为了简单，直接截断到最短有效长度
        valid_len = min(len(my_gx), len(roll))
        my_gx, my_gy, my_gz = my_gx[:valid_len], my_gy[:valid_len], my_gz[:valid_len]
        roll, pitch, yaw = roll[:valid_len], pitch[:valid_len], yaw[:valid_len]

    except KeyError as e:
        print(f"❌ 列名错误: {e}")
        print(f"   当前CSV列名: {list(df.columns)}")
        return

    # 计算真值角速度
    print("🔄 正在计算真值角速度 (Ground Truth Angular Velocity)...")

    # 注意：通常 CSV 里的 roll/pitch/yaw 是 角度(deg)，需要 degrees=True
    # 如果出来的黑线特别大，说明 CSV 里其实是弧度，改 degrees=False
    try:
        r_obj = R.from_euler('zyx', np.stack([yaw, pitch, roll], axis=1), degrees=True)
        R_mats = r_obj.as_matrix()

        gt_w = []
        for i in range(len(R_mats) - 1):
            # w = log(R(t)^T * R(t+1)) / dt
            dR = R_mats[i].T @ R_mats[i + 1]
            r_vec = R.from_matrix(dR).as_rotvec()
            w = r_vec / dt_mean
            gt_w.append(w)

        gt_w = np.array(gt_w)
        # 补齐最后一个点
        gt_w = np.vstack([gt_w, gt_w[-1]])

    except Exception as e:
        print(f"❌ 真值计算失败: {e}")
        return

    # ==================================================================
    # 3. 绘图对比
    # ==================================================================
    print("🎨 正在绘图...")
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # X 轴对比
    axs[0].plot(gt_w[:, 0], 'k', label='GT_X (Truth)', linewidth=2)
    axs[0].plot(my_gx, 'r--', label='Gyro_X (Raw)', alpha=0.7)
    axs[0].set_title(f"X Axis Check (Roll Axis) - dt={dt_mean:.4f}")
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    # Y 轴对比
    axs[1].plot(gt_w[:, 1], 'k', label='GT_Y (Truth)', linewidth=2)
    axs[1].plot(my_gy, 'r--', label='Gyro_Y (Raw)', alpha=0.7)
    axs[1].set_title("Y Axis Check (Pitch Axis)")
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    # Z 轴对比
    axs[2].plot(gt_w[:, 2], 'k', label='GT_Z (Truth)', linewidth=2)
    axs[2].plot(my_gz, 'r--', label='Gyro_Z (Raw)', alpha=0.7)
    axs[2].set_title("Z Axis Check (Yaw Axis)")
    axs[2].legend(loc='upper right')
    axs[2].grid(True)

    plt.tight_layout()
    print("✅ 图片绘制完成！")
    plt.show()


if __name__ == "__main__":
    check_axis()