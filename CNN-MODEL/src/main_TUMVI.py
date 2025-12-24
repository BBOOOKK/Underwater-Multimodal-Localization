import os
import torch
import src.learning as lr
import src.networks as sn
import src.losses as sl
import src.dataset as ds
import numpy as np

# 获取当前脚本路径
base_dir = os.path.dirname(os.path.realpath(__file__))

# [修改点1] 指向你的 E 盘数据路径
data_dir = r'E:\data\MY_DATA'

# 测试时使用最新的训练结果
address = 'last'

################################################################################
# Network parameters (网络参数 - TUM 版本的参数通常针对噪声更小的 IMU)
################################################################################
net_class = sn.GyroNet
net_params = {
    'in_dim': 6,
    'out_dim': 3,
    'c0': 16,
    'dropout': 0.1,
    'ks': [7, 7, 7, 7],
    'ds': [4, 4, 4],
    'momentum': 0.1,
    # 注意：这里的噪声参数比 EuRoC 版小很多 (0.2 vs 1.0)，
    # 如果你的 IMU 比较便宜（噪声大），可能需要改回 1.0 或 2.0
    'gyro_std': [0.2 * np.pi / 180, 0.2 * np.pi / 180, 0.2 * np.pi / 180],
}

################################################################################
# Dataset parameters (数据集参数)
################################################################################
# [关键修改] 使用我们修改过的 EUROCDataset 类，因为它支持读取自定义的 .p 文件
dataset_class = ds.EUROCDataset

dataset_params = {
    # 数据路径
    'data_dir': data_dir,
    'predata_dir': data_dir,

    # [修改点2] 序列名改为你生成的文件名
    'train_seqs': [
        'train',
    ],
    'val_seqs': [
        'val',
    ],
    'test_seqs': [
        'test',
    ],

    # 训练轨迹长度
    'N': 32 * 500,
    'min_train_freq': 16,
    'max_train_freq': 32,
}

################################################################################
# Training parameters (训练参数)
################################################################################
train_params = {
    'optimizer_class': torch.optim.Adam,
    'optimizer': {
        'lr': 0.01,
        'weight_decay': 1e-1,
        'amsgrad': False,
    },
    'loss_class': sl.GyroLoss,
    'loss': {
        'min_N': int(np.log2(dataset_params['min_train_freq'])),
        'max_N': int(np.log2(dataset_params['max_train_freq'])),
        'w': 1e6,

        # [修改点3] 改为 'rotation matrix'
        # 原代码是 'rotation matrix mask'，因为 TUM 数据集真值有间断。
        # 你的数据是连续的，不需要 mask，否则会报错缺数据维度。
        'target': 'rotation matrix',

        'huber': 0.005,
        'dt': 0.005,  # [注意] 请确认你的采样率是否为 200Hz (0.005s)
    },
    'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'scheduler': {
        'T_0': 600,
        'T_mult': 2,
        'eta_min': 1e-3,
    },
    'dataloader': {
        'batch_size': 10,
        'pin_memory': False,
        'num_workers': 0,  # Windows 必须为 0
        'shuffle': True,  # 训练时建议打乱
    },
    # 验证频率
    'freq_val': 10,
    # 总 Epoch 数
    'n_epochs': 100,

    # [修改点4] 结果保存路径 (保存到 TUM 文件夹下以示区分)
    'res_dir': r'E:\results\TUM',
    'tb_dir': r'E:\results\runs\TUM',
}

################################################################################
# Train on training data set (主程序)
################################################################################
if __name__ == '__main__':
    # 1. 开启训练
    print("开始使用 TUM 配置进行训练...")
    learning_process = lr.GyroLearningBasedProcessing(
        train_params['res_dir'],
        train_params['tb_dir'],
        net_class,
        net_params,
        None,
        train_params['loss']['dt']
    )
    learning_process.train(dataset_class, dataset_params, train_params)

    # 2. 开启测试
    print("\n开始测试...")
    learning_process = lr.GyroLearningBasedProcessing(
        train_params['res_dir'],
        train_params['tb_dir'],
        net_class,
        net_params,
        address=address,
        dt=train_params['loss']['dt']
    )
    learning_process.test(dataset_class, dataset_params, ['test'])