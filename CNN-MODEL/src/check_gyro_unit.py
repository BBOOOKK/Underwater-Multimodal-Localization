import pandas as pd
import numpy as np
import os
import glob

# 你的数据根目录
BASE_PATH = r"E:\水下导航资料\xiachi"


def check_unit(folder_id):
    csv_path = os.path.join(BASE_PATH, str(folder_id), "aligned_dataset.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    # 提取 gx, gy, gz
    # 注意：aligned_dataset.csv 的列名通常是 gx, gy, gz
    if 'gx' not in df.columns:
        print(f"[{folder_id}] 警告：没找到 gx 列，跳过")
        return

    gyro_data = df[['gx', 'gy', 'gz']].values
    max_val = np.max(np.abs(gyro_data))

    print(f"[{folder_id}] 陀螺仪最大绝对值: {max_val:.4f}")

    if max_val > 15:
        print(f"    ⚠️  看起来像是【度 (deg/s)】！需要转换！ (15 rad/s ≈ 860 deg/s，通常水下机器人没这么快)")
    elif max_val < 10:
        print(f"    ✅ 看起来像是【弧度 (rad/s)】。无需转换。")
    else:
        print(f"    ❓ 数值在 10~15 之间，比较模糊，请手动确认。")


print("正在检查陀螺仪单位...")
for i in range(1, 7):
    check_unit(i)