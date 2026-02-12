# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np

# 🔥 [关键] 这里填入你的真实数据路径
DATA_PATHS = [
    r"E:\电脑管家迁移文件\xwechat_files\wxid_9ong8xwpxo7f22_51c6\msg\file\2026-01\jxt(1)\jxt\data\processed_data",
    r"E:\电脑管家迁移文件\xwechat_files\wxid_9ong8xwpxo7f22_51c6\msg\file\2026-01\jxt(1)\jxt\data\all_data"
]


def analyze_single_file(file_path):
    try:
        # 1. 读取 CSV
        df = pd.read_csv(file_path)

        # 2. 智能匹配列名 (兼容不同命名习惯)
        col_map = {
            'x': ['gyro_x', 'gryX', 'wx', 'GyrX'],
            'y': ['gyro_y', 'gryY', 'wy', 'GyrY'],
            'z': ['gyro_z', 'gryZ', 'wz', 'GyrZ']
        }

        gx, gy, gz = None, None, None
        for cand in col_map['x']:
            if cand in df.columns: gx = cand; break
        for cand in col_map['y']:
            if cand in df.columns: gy = cand; break
        for cand in col_map['z']:
            if cand in df.columns: gz = cand; break

        if not (gx and gy and gz):
            return None  # 没找到陀螺仪数据

        # 强制转为数值型
        for c in [gx, gy, gz]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=[gx, gy, gz])

        data = df[[gx, gy, gz]].values

        # 3. 自动单位检测 (Deg vs Rad)
        # 如果最大值 > 15.0 (约 860度/秒)，通常认为是角度制，转为弧度
        max_val = np.max(np.abs(data))
        unit_info = "Rad (Raw)"
        if max_val > 15.0:
            data = np.deg2rad(data)
            unit_info = "Deg -> Rad (Auto-converted)"

        # 4. 寻找静止片段 (Stationary Analysis)
        # 使用滑动窗口标准差来寻找“相对静止”的时刻
        window = 100  # 假设 100Hz，对应 1秒
        rolling_std = pd.DataFrame(data).rolling(window=window).std().mean(axis=1)

        # 阈值：取所有数据中波动最小的前 20% 作为静止参考
        threshold = rolling_std.quantile(0.2)
        static_mask = rolling_std < threshold
        static_data = data[static_mask]

        if len(static_data) < 100:
            return None  # 找不到足够的静止数据

        # === 核心指标计算 ===

        # A. 底噪 (White Noise Std)
        # 取三轴中最大的那个标准差，作为保守估计
        noise_std = np.max(np.std(static_data, axis=0))

        # B. 静态偏置 (Static Bias Magnitude)
        # 静止数据的均值绝对值
        bias_mag = np.max(np.abs(np.mean(static_data, axis=0)))

        # C. 漂移量 (Drift / Instability)
        # 通过计算静止片段均值的变化范围来估算
        # 如果是 Random Walk，均值会随时间游走
        rolling_mean = pd.DataFrame(static_data).rolling(window=200).mean().dropna()
        if len(rolling_mean) > 0:
            # 漂移 = 最大均值 - 最小均值
            drift_range = np.max(rolling_mean.max() - rolling_mean.min())
        else:
            drift_range = 0.0

        return {
            'File': os.path.basename(file_path),
            'Folder': os.path.basename(os.path.dirname(file_path)),
            'Std': noise_std,
            'Bias': bias_mag,
            'Drift': drift_range
        }

    except Exception as e:
        # print(f"[ERR] Error reading {file_path}: {e}")
        return None


def main():
    print(f"🔍 开始扫描数据目录...\n")

    all_csv_files = []
    for d_path in DATA_PATHS:
        if os.path.exists(d_path):
            print(f"  -> Scanning: {d_path}")
            # 递归查找所有 csv
            files = glob.glob(os.path.join(d_path, "**", "*.csv"), recursive=True) + \
                    glob.glob(os.path.join(d_path, "*.csv"))
            all_csv_files.extend(files)
        else:
            print(f"  ❌ Path not found: {d_path}")

    # 去重
    all_csv_files = list(set(all_csv_files))
    print(f"\nFound {len(all_csv_files)} potential CSV files.")

    results = []
    for f in all_csv_files:
        res = analyze_single_file(f)
        if res:
            results.append(res)
            # print(f"✅ Analyzed: {res['File']:<25} | Std: {res['Std']:.4f} | Bias: {res['Bias']:.4f}")

    if not results:
        print("\n❌ 未找到有效的 IMU 数据文件。")
        return

    # === 生成汇总报告 ===
    df_res = pd.DataFrame(results)

    # 按照 Bias 大小排序，方便查看
    df_res = df_res.sort_values(by='Bias', ascending=False)

    print("\n" + "=" * 100)
    print(f"📊 最终数据分析报告 (共 {len(df_res)} 个有效文件)")
    print("=" * 100)
    # 打印前 20 个重要文件
    print(df_res[['File', 'Folder', 'Std', 'Bias', 'Drift']].head(20).to_string(index=False))
    print("-" * 100)

    # === 智能参数推荐 ===

    rec_std_min = df_res['Std'].min() * 0.5
    rec_std_max = df_res['Std'].max() * 1.5

    rec_bias_min = df_res['Bias'].min() * 0.5
    rec_bias_max = df_res['Bias'].max() * 2.0

    max_drift = df_res['Drift'].max()

    # 估算 drift_speed (步长)
    # 假设 drift 是在约 1000 步内累积出来的
    rec_drift_speed = max_drift / 1000.0

    print("\n💡 【推荐 Dataset 参数设置】(请复制给 AI)")
    print(f"1. imu_std (白噪声范围):  [{rec_std_min:.5f}, {rec_std_max:.5f}]")
    print(f"2. imu_b0  (Bias 范围):   [{rec_bias_min:.5f}, {rec_bias_max:.5f}]")
    print(f"3. drift_speed (漂移步长): {rec_drift_speed:.6f}")
    print("=" * 100)


if __name__ == "__main__":
    main()