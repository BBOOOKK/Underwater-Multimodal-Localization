import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ================= 配置参数 =================
# 使用多个CSV文件进行训练
CSV_FILES = [
    '/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/processed_data/suiyi1.csv',
    '/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/processed_data/suiyi2.csv',
    '/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/processed_data/zhengfangxing1.csv',
    '/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/processed_data/zhengfangxing2.csv',
    '/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/processed_data/zhengfangxing3.csv',
    '/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/processed_data/zhixian1.csv',
    '/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/processed_data/zhixian2.csv'
]
WINDOW_SIZE = 50          # 窗口大小 (100Hz下 50步 = 0.5秒)
BATCH_SIZE = 64
EPOCHS = 50
TEST_SIZE = 0.2           # 验证集比例
# ===========================================

def load_and_process_data(filepaths, window_size):
    """
    加载多个数据文件，生成标签，归一化，并制作4分支的滑动窗口序列
    """
    all_X_dvl, all_X_att, all_X_acc, all_X_gyro = [], [], [], []
    all_Y_seq = []
    
    # 为每个文件单独创建归一化器，最后统一处理
    all_X_data_list = []
    all_Y_data_list = []
    
    for filepath in filepaths:
        print(f"正在加载数据: {filepath} ...")
        df = pd.read_csv(filepath)
        
        # 1. 制作标签 (Label): 计算相邻时刻的位置差 (Delta X, Delta Y)
        df['delta_x'] = df['gt_px'].diff().fillna(0)
        df['delta_y'] = df['gt_py'].diff().fillna(0)
        
        # 2. 定义特征列 (适配 suiyi1.csv)
        # 分支1: DVL 速度
        cols_dvl = ['dvl_vx', 'dvl_vy', 'dvl_vz']
        # 分支2: 姿态角 (Attitude) - 使用 raw 数据
        cols_att = ['roll_raw', 'pitch_raw', 'yaw_raw']
        # 分支3: 加速度 (Acceleration) - 注意 csv 中的拼写 'axxY'
        cols_acc = ['accX', 'axxY', 'accZ']
        # 分支4: 角速度 (Angular Velocity/Gyro)
        cols_gyro = ['gryX', 'gryY', 'gryZ']
        
        # 合并所有特征列以便归一化
        all_features = cols_dvl + cols_att + cols_acc + cols_gyro
        label_cols = ['delta_x', 'delta_y']
        
        # 检查列是否存在
        available_features = [col for col in all_features if col in df.columns]
        available_labels = [col for col in label_cols if col in df.columns]
        
        if len(available_features) < 12:
            print(f"警告: 文件 {filepath} 缺少某些特征列，跳过")
            continue
            
        # 收集原始数据用于全局归一化
        all_X_data_list.append(df[available_features].values)
        all_Y_data_list.append(df[available_labels].values)
    
    # 3. 全局归一化 (使用所有数据)
    if not all_X_data_list:
        raise ValueError("没有有效的数据文件")
        
    X_all = np.vstack(all_X_data_list)
    Y_all = np.vstack(all_Y_data_list)
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_x.fit(X_all)
    scaler_y.fit(Y_all)
    
    print("全局数据归一化完成。")
    
    # 4. 处理每个文件并生成序列
    for i, filepath in enumerate(filepaths):
        df = pd.read_csv(filepath)
        df['delta_x'] = df['gt_px'].diff().fillna(0)
        df['delta_y'] = df['gt_py'].diff().fillna(0)
        
        cols_dvl = ['dvl_vx', 'dvl_vy', 'dvl_vz']
        cols_att = ['roll_raw', 'pitch_raw', 'yaw_raw']
        cols_acc = ['accX', 'axxY', 'accZ']
        cols_gyro = ['gryX', 'gryY', 'gryZ']
        all_features = cols_dvl + cols_att + cols_acc + cols_gyro
        label_cols = ['delta_x', 'delta_y']
        
        available_features = [col for col in all_features if col in df.columns]
        available_labels = [col for col in label_cols if col in df.columns]
        
        if len(available_features) < 12:
            continue
            
        X_data = scaler_x.transform(df[available_features].values)
        Y_data = scaler_y.transform(df[available_labels].values)
        
        # 制作滑动窗口序列
        X_seq, Y_seq = [], []
        for j in range(window_size, len(df)):
            X_seq.append(X_data[j-window_size:j])
            Y_seq.append(Y_data[j])
            
        if len(X_seq) == 0:
            continue
            
        X_seq = np.array(X_seq)
        Y_seq = np.array(Y_seq)
        
        # 拆分4个分支的数据
        X_dvl  = X_seq[:, :, 0:3]
        X_att  = X_seq[:, :, 3:6]
        X_acc  = X_seq[:, :, 6:9]
        X_gyro = X_seq[:, :, 9:12]
        
        all_X_dvl.append(X_dvl)
        all_X_att.append(X_att)
        all_X_acc.append(X_acc)
        all_X_gyro.append(X_gyro)
        all_Y_seq.append(Y_seq)
        
        print(f"文件 {filepath} 处理完成，样本数: {len(Y_seq)}")
    
    # 合并所有文件的数据
    X_dvl_all = np.vstack(all_X_dvl) if all_X_dvl else np.array([])
    X_att_all = np.vstack(all_X_att) if all_X_att else np.array([])
    X_acc_all = np.vstack(all_X_acc) if all_X_acc else np.array([])
    X_gyro_all = np.vstack(all_X_gyro) if all_X_gyro else np.array([])
    Y_all_seq = np.vstack(all_Y_seq) if all_Y_seq else np.array([])
    
    total_samples = len(Y_all_seq)
    print(f"所有文件处理完成，总样本数: {total_samples}")
    
    return [X_dvl_all, X_att_all, X_acc_all, X_gyro_all], Y_all_seq, scaler_y

def build_feature_extraction_branch(input_shape, name):
    """
    构建单分支特征提取网络 (1D-CNN + Residual)
    """
    inputs = Input(shape=input_shape, name=f'input_{name}')
    
    # 级联 1D-CNN (参考论文参数)
    x = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    
    x = layers.BatchNormalization()(x)
    
    # Residual Connection
    shortcut = layers.Conv1D(filters=256, kernel_size=1, padding='same')(inputs)
    x = layers.Add()([x, shortcut])
    
    return inputs, x

def build_4branch_model(window_size):
    """
    构建完整的 4分支 混合模型
    """
    input_shape = (window_size, 3)
    
    # 1. 四个独立分支
    in_dvl,  out_dvl  = build_feature_extraction_branch(input_shape, 'dvl')
    in_att,  out_att  = build_feature_extraction_branch(input_shape, 'att')
    in_acc,  out_acc  = build_feature_extraction_branch(input_shape, 'acc')
    in_gyro, out_gyro = build_feature_extraction_branch(input_shape, 'gyro')
    
    # 2. 特征拼接 (Concatenate)
    concat = layers.Concatenate(axis=-1)([out_dvl, out_att, out_acc, out_gyro])
    
    # 3. LSTM 融合
    x = layers.LSTM(256, return_sequences=True)(concat)
    x = layers.LSTM(256, return_sequences=False)(x)
    
    # 4. 回归输出
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='linear', name='pos_delta')(x)
    
    model = models.Model(
        inputs=[in_dvl, in_att, in_acc, in_gyro], 
        outputs=outputs, 
        name='Hybrid_4Branch_AUV_Net'
    )
    return model

# ================= 主程序 =================
if __name__ == '__main__':
    # 1. 加载数据
    inputs, targets, scaler_y = load_and_process_data(CSV_FILES, WINDOW_SIZE)
    
    # 2. 拆分训练/测试集
    idx = np.arange(len(targets))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, shuffle=False)
    
    X_train = [inp[train_idx] for inp in inputs]
    Y_train = targets[train_idx]
    
    X_test = [inp[test_idx] for inp in inputs]
    Y_test = targets[test_idx]
    
    # 3. 构建模型
    model = build_4branch_model(WINDOW_SIZE)
    model.summary()
    
    # 4. 训练
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    print("\n开始训练...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # 5. 结果可视化
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
    
    # 6. 轨迹预测对比
    print("\n生成轨迹对比...")
    preds = model.predict(X_test)
    
    # 反归一化
    preds_real = scaler_y.inverse_transform(preds)
    Y_test_real = scaler_y.inverse_transform(Y_test)
    
    # 累加位移得到轨迹 (画前1000个点，避免图太乱)
    plot_len = 1000
    if len(preds_real) < plot_len: plot_len = len(preds_real)
    
    # 假设从 (0,0) 开始，或者从测试集第一个真实位置开始
    traj_pred = np.cumsum(preds_real[:plot_len], axis=0)
    traj_gt   = np.cumsum(Y_test_real[:plot_len], axis=0)
    
    plt.figure(figsize=(8, 8))
    plt.plot(traj_gt[:, 0], traj_gt[:, 1], 'k-', label='Ground Truth', linewidth=2)
    plt.plot(traj_pred[:, 0], traj_pred[:, 1], 'r--', label='Prediction', linewidth=1.5)
    plt.title(f'Trajectory Reconstruction (First {plot_len} steps)')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # 保持比例
    plt.show()
