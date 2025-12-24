import os
import glob
import pickle
import torch
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import src.learning as lr
import src.networks as sn
import src.losses as sl
import src.dataset as ds
from src.lie_algebra import SO3

# ==============================================================================
# 路径配置
# ==============================================================================
base_dir = os.path.dirname(os.path.realpath(__file__))

# 原始 CSV 文件所在的目录
RAW_CSV_DIR = r"E:\水下导航资料\processed_data"

# 输出 .p 文件的目录
data_dir = os.path.join(base_dir, 'src', 'data', 'MY_DATA')

# 结果保存路径
res_dir = r'E:\results\EUROC'
tb_dir = r'E:\results\runs\EUROC'

if not os.path.exists(res_dir): os.makedirs(res_dir)
if not os.path.exists(tb_dir): os.makedirs(tb_dir)

# 强制不加载旧模型
address = None


# ==============================================================================
# 自动化数据清洗与生成
# ==============================================================================
def bmtm(A, B):
    return torch.bmm(A.transpose(1, 2), B)


def regenerate_data_from_csv(raw_dir, out_dir, min_train_freq=16):
    """
    1. 删除 out_dir 下所有 .p 文件
    2. 读取 raw_dir 下的 CSV
    3. 应用轴向修正并生成新的 .p 文件
    """
    print("\n" + "=" * 60)
    print("🧹 [自动流程] 正在清理旧缓存并重新生成数据...")
    print("=" * 60)

    # 1. 确保输出目录存在
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 2. 暴力删除旧文件 (包括 nf.p)
    old_files = glob.glob(os.path.join(out_dir, "*.p"))
    for f in old_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"删除失败 {f}: {e}")
    print(f"   -> 已删除 {len(old_files)} 个旧文件 (包含 nf.p)")

    # 3. 查找 CSV
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"在 {raw_dir} 没找到任何 .csv 文件！")

    print(f"   -> 发现 {len(csv_files)} 个 CSV 文件，开始转换...")

    for i, csv_path in enumerate(csv_files):
        seq_id = str(i + 1)  # 生成 1, 2, 3...
        try:
            df = pd.read_csv(csv_path)

            # --- A. 读取原始数据 ---
            # 注意：这里根据你的列名
            raw_gx = df['gryX'].values
            raw_gy = df['gryY'].values
            raw_gz = df['gryZ'].values

            # 加速度 (兼容 axxY 拼写错误)
            if 'axxY' in df.columns:
                raw_ay = df['axxY'].values
            else:
                raw_ay = df['accY'].values
            raw_ax = df['accX'].values
            raw_az = df['accZ'].values

            # --- B. 🚨 核心步骤：应用轴向修正 🚨 ---

            # 修正 Gyro
            gx = raw_gy
            gy = -raw_gx
            gz = raw_gz

            # 修正 Acc (假设安装是刚体，变换相同)
            ax = raw_ay
            ay = -raw_ax
            az = raw_az

            # --- C. 处理 GT (用于生成标签 xs) ---
            roll = df['roll_body'].values
            pitch = df['pitch_body'].values
            yaw = df['yaw_body'].values

            # 欧拉角 -> 四元数 -> 旋转矩阵
            r_obj = R.from_euler('zyx', np.stack([yaw, pitch, roll], axis=1), degrees=False)
            gt_quats = r_obj.as_quat()  # [x, y, z, w]

            # 转换为 Tensor
            gyro_t = torch.tensor(np.stack([gx, gy, gz], axis=1)).double()
            acc_t = torch.tensor(np.stack([ax, ay, az], axis=1)).double()
            us = torch.cat([gyro_t, acc_t], dim=1).float()  # (N, 6)

            # 处理 GT 旋转
            q_xyzw = torch.tensor(gt_quats).double()
            q_wxyz = q_xyzw[:, [3, 0, 1, 2]]  # 调整为 wxyz
            q_wxyz = SO3.qnorm(q_wxyz)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            R_all = SO3.from_quaternion(q_wxyz.to(device), ordering='wxyz')

            # 计算相对旋转 (Label xs)
            if R_all.shape[0] <= min_train_freq:
                print(f"   ⚠️ 跳过 {seq_id} (数据太短)")
                continue

            R_i = R_all[:-min_train_freq]
            R_j = R_all[min_train_freq:]
            dRot = bmtm(R_i, R_j)
            dRot = SO3.dnormalize(dRot)
            xs = SO3.log(dRot).cpu().float()

            # --- D. 保存训练数据 ({id}.p) ---
            save_path = os.path.join(out_dir, f"{seq_id}.p")
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'us': us,
                    'xs': xs,
                    't': df['timestamp'].values
                }, f)

            # --- E. 保存真值数据 ({id}_gt.p) ---
            # 主要是为了评估程序 evaluate_results.py 使用
            gt_path = os.path.join(out_dir, f"{seq_id}_gt.p")

            # 提取位置
            if 'est_px' in df.columns:
                pos = np.stack([df['est_px'], df['est_py'], df['est_pz']], axis=1)
            else:
                pos = np.zeros((len(df), 3))

            with open(gt_path, 'wb') as f:
                pickle.dump({
                    'ts': df['timestamp'].values,
                    'qs': q_wxyz.cpu().float(),  # [w,x,y,z]
                    'ps': torch.tensor(pos).float()
                }, f)

            print(f"   ✅ 生成完成: {seq_id}.p (应用了轴向修正)")

        except Exception as e:
            print(f"   ❌ 处理 {csv_path} 失败: {e}")

    print("=" * 60 + "\n")


################################################################################
# Network parameters
################################################################################
net_class = sn.GyroNet
net_params = {
    'in_dim': 6,
    'out_dim': 3,
    'c0': 16,
    'dropout': 0.2,  # 建议 0.2
    'ks': [7, 7, 7, 7],
    'ds': [4, 4, 4],
    'momentum': 0.1,
    'gyro_std': [1 * np.pi / 180, 2 * np.pi / 180, 5 * np.pi / 180],
}

################################################################################
# Dataset parameters
################################################################################
dataset_class = ds.EUROCDataset
dataset_params = {
    'data_dir': data_dir,
    'predata_dir': data_dir,
    'train_seqs': ['1', '2', '3', '4', '5'],
    'val_seqs': ['6'],
    'test_seqs': ['7'],
    'N': 1024,
    'min_train_freq': 16,
    'max_train_freq': 32,
}

################################################################################
# Training parameters
################################################################################
min_N = int(np.log2(dataset_params['min_train_freq']))
max_N = int(np.log2(dataset_params['max_train_freq']))

train_params = {
    'optimizer_class': torch.optim.Adam,
    'optimizer': {
        'lr': 0.01,  # 如果 loss 震荡，可以尝试 0.001
        'weight_decay': 1e-2,  # 适当增加 weight_decay 有助于防止过拟合
        'amsgrad': False,
    },
    'loss_class': sl.GyroLoss,
    'loss': {
        'min_N': min_N,
        'max_N': max_N,
        'w': 1e5,
        'target': 'rotation matrix',
        'huber': 0.005,
        'dt': 0.01,
    },
    'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'scheduler': {
        'T_0': 600,
        'T_mult': 2,
        'eta_min': 1e-3,
    },
    'dataloader': {
        'batch_size': 8,
        'pin_memory': False,
        'num_workers': 0,
        'shuffle': True,
    },
    'freq_val': 10,
    'n_epochs': 300,
    'res_dir': res_dir,
    'tb_dir': tb_dir,
}

################################################################################
# Main Execution
################################################################################
if __name__ == '__main__':
    # ==========================================================================
    # 0. 关键步骤：每次训练前强制清理并重新生成数据
    # ==========================================================================
    regenerate_data_from_csv(RAW_CSV_DIR, data_dir, dataset_params['min_train_freq'])

    # ==========================================================================
    # 1. 训练
    # ==========================================================================
    print(">>> 开始训练...")
    # address=None 确保重新初始化网络
    learning_process = lr.GyroLearningBasedProcessing(
        train_params['res_dir'],
        train_params['tb_dir'],
        net_class,
        net_params,
        address=None,
        dt=train_params['loss']['dt']
    )
    learning_process.train(dataset_class, dataset_params, train_params)

    # ==========================================================================
    # 2. 测试
    # ==========================================================================
    print("\n>>> 开始测试...")
    learning_process = lr.GyroLearningBasedProcessing(
        train_params['res_dir'],
        train_params['tb_dir'],
        net_class,
        net_params,
        address="last",
        dt=train_params['loss']['dt']
    )

    print(f"    正在测试集 {dataset_params['test_seqs']} 上进行评估...")
    learning_process.test(dataset_class, dataset_params, ['test'])

    print(f"\n>>> 全部完成！请运行 evaluate_results.py 查看详细图表。")