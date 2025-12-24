import pandas as pd
import numpy as np
import re
from datetime import datetime
import os
import glob
from scipy.spatial.transform import Rotation as R

# ==================== 配置区域 ====================
BASE_PATH = r"E:\水下导航资料\xiachi"  # 根目录
DVL_TIME_OFFSET = 0

# 定义可能的列名映射 (优先级: 左边优先)
# 格式: 目标列 -> [候选列名列表]
IMU_COLUMN_MAPPING = {
    'ax': ['accX', 'ax', 'acc_x'],
    'ay': ['accY', 'ay', 'acc_y', 'axxY'],  # 兼容 axxY
    'az': ['accZ', 'az', 'acc_z'],
    'gx': ['gyrX', 'gx', 'gyr_x', 'gryX'],  # 兼容 gryX
    'gy': ['gyrY', 'gy', 'gyr_y', 'gryY'],
    'gz': ['gyrZ', 'gz', 'gyr_z', 'gryZ']
}


# =================================================

def parse_time_string(time_str):
    try:
        time_str = str(time_str).strip()
        try:
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()
    except:
        return None


def find_col(df, candidates):
    """在DataFrame中模糊查找列名 (忽略大小写)"""
    df_cols_lower = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in df_cols_lower:
            # 返回真实的列名
            return df.columns[df_cols_lower.index(cand.lower())]
    return None


def load_imu(file_path):
    print(f"  [读取 IMU] {os.path.basename(file_path)}")
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        df.columns = [str(c).strip() for c in df.columns]

        # 1. 找时间列
        time_col = next((c for c in df.columns if 'time' in c.lower()), None)
        if not time_col:
            time_col = df.columns[0]

        df['timestamp'] = df[time_col].apply(parse_time_string)

        df_clean = pd.DataFrame()
        df_clean['timestamp'] = df['timestamp']

        # 2. 智能匹配传感器列
        print("    -> 正在匹配列名...")
        for target_key, candidate_list in IMU_COLUMN_MAPPING.items():
            found_col = find_col(df, candidate_list)
            if found_col:
                # print(f"       {target_key} -> {found_col}") # 调试用
                df_clean[target_key] = df[found_col]
            else:
                print(f"       ⚠️ 警告: 未找到 {target_key} 数据，尝试列表: {candidate_list}")
                df_clean[target_key] = 0.0

        return df_clean.dropna(subset=['timestamp']).sort_values('timestamp')
    except Exception as e:
        print(f"    ❌ IMU读取失败: {e}")
        return None


def load_dvl(file_path):
    print(f"  [读取 DVL] {os.path.basename(file_path)}")
    data_list = []
    current_ts = None
    time_pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+")

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line: continue

            if time_pattern.match(line):
                current_ts = parse_time_string(line)
                continue

            # :BS, Vx, Vy, Vz (mm/s)
            if line.startswith(':BS') and current_ts is not None:
                parts = line.split(',')
                try:
                    vx = float(parts[1]) / 1000.0
                    vy = float(parts[2]) / 1000.0
                    vz = float(parts[3]) / 1000.0
                    data_list.append([current_ts, vx, vy, vz])
                except:
                    continue

        if not data_list: return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=['timestamp', 'dvl_vx', 'dvl_vy', 'dvl_vz'])
        df['timestamp'] = df['timestamp'] + DVL_TIME_OFFSET
        return df.sort_values('timestamp')
    except:
        return pd.DataFrame()


def load_avp_as_gt(file_path):
    print(f"  [读取 AVP] {os.path.basename(file_path)}")
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        df.columns = [str(c).strip().lower() for c in df.columns]

        time_col = next((c for c in df.columns if 'time' in c), None)
        if time_col:
            df['timestamp'] = df[time_col].apply(parse_time_string)
        else:
            return None

        # 姿态
        if all(c in df.columns for c in ['roll', 'pitch', 'yaw']):
            euler = df[['roll', 'pitch', 'yaw']].values
            is_degrees = np.max(np.abs(euler)) > 6.3
            r = R.from_euler('xyz', euler, degrees=is_degrees)
            quats = r.as_quat()
        else:
            return None

        df_gt = pd.DataFrame()
        df_gt['timestamp'] = df['timestamp']

        # 位置归一化
        if 'x' in df.columns:
            df_gt['gt_px'] = df['x'] - df['x'].iloc[0]
            df_gt['gt_py'] = df['y'] - df['y'].iloc[0]
            df_gt['gt_pz'] = df['z'] - df['z'].iloc[0]
        else:
            df_gt['gt_px'] = 0;
            df_gt['gt_py'] = 0;
            df_gt['gt_pz'] = 0

        df_gt['gt_qx'] = quats[:, 0]
        df_gt['gt_qy'] = quats[:, 1]
        df_gt['gt_qz'] = quats[:, 2]
        df_gt['gt_qw'] = quats[:, 3]

        if 'vx' in df.columns:
            df_gt['gt_vx'] = df['vx']
            df_gt['gt_vy'] = df['vy']
            df_gt['gt_vz'] = df['vz']

        return df_gt.dropna().sort_values('timestamp')

    except Exception as e:
        print(f"    ❌ AVP解析错误: {e}")
        return None


def align_all(df_imu, df_dvl, df_gt):
    df_final = df_imu.copy()
    t_target = df_final['timestamp'].values

    # 对齐 DVL
    if not df_dvl.empty:
        for col in ['dvl_vx', 'dvl_vy', 'dvl_vz']:
            df_final[col] = np.interp(t_target, df_dvl['timestamp'], df_dvl[col])
    else:
        for col in ['dvl_vx', 'dvl_vy', 'dvl_vz']:
            df_final[col] = 0.0

    # 对齐 GT
    if df_gt is not None:
        cols_to_interp = ['gt_px', 'gt_py', 'gt_pz', 'gt_qx', 'gt_qy', 'gt_qz', 'gt_qw']
        if 'gt_vx' in df_gt.columns:
            cols_to_interp.extend(['gt_vx', 'gt_vy', 'gt_vz'])

        for col in cols_to_interp:
            df_final[col] = np.interp(t_target, df_gt['timestamp'], df_gt[col])

    return df_final


def process_folder(folder_path):
    print(f"\n>>> 处理文件夹: {folder_path}")
    all_files = glob.glob(os.path.join(folder_path, "*"))

    file_avp = next((f for f in all_files if 'AVP' in os.path.basename(f) and f.endswith(('.csv', '.xlsx'))), None)

    file_imu = None
    for f in all_files:
        if f.endswith(('.csv', '.xlsx')):
            fname = os.path.basename(f).upper()
            if 'AVP' not in fname and 'POSE' not in fname and 'ALIGNED' not in fname:
                try:
                    if f.endswith('.csv'):
                        check_df = pd.read_csv(f, nrows=0)
                    else:
                        check_df = pd.read_excel(f, nrows=0)
                    # 简单检查是否有加速度计列
                    cols = [c.lower() for c in check_df.columns]
                    if any(x in cols for x in ['accx', 'ax', 'acc_x']):
                        file_imu = f
                        break
                except:
                    continue

    file_dvl = next((f for f in all_files if f.endswith('.dat')), None)

    if not file_imu:
        print("❌ 未找到 Raw IMU 文件")
        return
    if not file_avp:
        print("❌ 未找到 AVP 文件")
        return

    df_imu = load_imu(file_imu)
    df_avp = load_avp_as_gt(file_avp)
    df_dvl = load_dvl(file_dvl) if file_dvl else pd.DataFrame()

    if df_imu is None or df_avp is None: return

    df_aligned = align_all(df_imu, df_dvl, df_avp)

    out_path = os.path.join(folder_path, "aligned_dataset.csv")
    df_aligned.to_csv(out_path, index=False)
    print(f"✅ 生成完毕: {out_path}")


def main():
    for i in range(1, 7):
        path = os.path.join(BASE_PATH, str(i))
        if os.path.exists(path):
            process_folder(path)


if __name__ == "__main__":
    main()