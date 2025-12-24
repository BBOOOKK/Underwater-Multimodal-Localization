import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

BASE_PATH = r"E:\水下导航资料\xiachi"


def plot_check(csv_path, save_path):
    print(f"正在绘图: {csv_path}")
    try:
        df = pd.read_csv(csv_path)

        # 1. 计算 DVL 速度模长
        speed_dvl = np.sqrt(df['dvl_vx'] ** 2 + df['dvl_vy'] ** 2)

        # 2. 获取 GT (AVP) 速度模长
        # 优先使用 AVP 直接提供的速度 (gt_vx)
        if 'gt_vx' in df.columns:
            speed_gt = np.sqrt(df['gt_vx'] ** 2 + df['gt_vy'] ** 2)
            label_gt = "GT Speed (From AVP)"
        else:
            # 如果 AVP 没提供速度，才使用位置微分
            dt = np.diff(df['timestamp'])
            dt[dt == 0] = 0.001
            vx = np.diff(df['gt_px']) / dt
            vy = np.diff(df['gt_py']) / dt
            vx = np.append(vx, vx[-1])
            vy = np.append(vy, vy[-1])
            speed_gt = np.sqrt(vx ** 2 + vy ** 2)
            label_gt = "GT Speed (Derived)"

        # 3. 绘图
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], speed_dvl, 'b-', label='DVL Speed', alpha=0.6)
        plt.plot(df['timestamp'], speed_gt, 'r--', label=label_gt, alpha=0.8, linewidth=1.5)

        plt.title(f"Speed Check: {os.path.basename(os.path.dirname(csv_path))}")
        plt.xlabel("Time")
        plt.ylabel("Speed (m/s)")
        plt.legend()
        plt.grid(True)

        plt.savefig(save_path)
        plt.close()
        print(f"  图片保存至: {save_path}")

    except Exception as e:
        print(f"绘图出错: {e}")


def main():
    for i in range(1, 7):
        folder = os.path.join(BASE_PATH, str(i))
        csv_file = os.path.join(folder, "aligned_dataset.csv")
        if os.path.exists(csv_file):
            plot_check(csv_file, os.path.join(folder, "speed_check.png"))


if __name__ == "__main__":
    main()