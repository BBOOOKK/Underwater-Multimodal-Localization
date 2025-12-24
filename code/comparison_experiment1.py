import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ================= 配置参数 =================
CSV_FILE = '/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/processed_data/zhengfangxing1.csv'
WINDOW_SIZE = 50       # 窗口大小
BATCH_SIZE = 64
EPOCHS = 50            # 训练轮数
TEST_SIZE = 0.2        # 测试集比例 (用于计算指标)
# ===========================================

def load_and_process_data(filepath, window_size):
    print(f"正在加载数据: {filepath} ...")
    df = pd.read_csv(filepath)
    
    # 1. 制作标签 (位移增量)
    df['delta_x'] = df['gt_px'].diff().fillna(0)
    df['delta_y'] = df['gt_py'].diff().fillna(0)
    
    # 2. 定义特征列
    cols_dvl = ['dvl_vx', 'dvl_vy', 'dvl_vz']
    cols_att = ['roll_raw', 'pitch_raw', 'yaw_raw']
    cols_acc = ['accX', 'axxY', 'accZ']
    cols_gyro = ['gryX', 'gryY', 'gryZ']
    
    all_features = cols_dvl + cols_att + cols_acc + cols_gyro
    label_cols = ['delta_x', 'delta_y']
    
    # 3. 归一化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_data = scaler_x.fit_transform(df[all_features].values)
    Y_data = scaler_y.fit_transform(df[label_cols].values)
    
    # 提取绝对坐标
    GT_pos = df[['gt_px', 'gt_py']].values
    
    # 4. 滑动窗口
    X_seq, Y_seq, GT_seq = [], [], []
    for i in range(window_size, len(df)):
        X_seq.append(X_data[i-window_size:i]) 
        Y_seq.append(Y_data[i])
        GT_seq.append(GT_pos[i]) 
        
    X_seq = np.array(X_seq)
    Y_seq = np.array(Y_seq)
    GT_seq = np.array(GT_seq)
    
    # 5. 拆分分支
    X_dvl  = X_seq[:, :, 0:3]
    X_att  = X_seq[:, :, 3:6]
    X_acc  = X_seq[:, :, 6:9]
    X_gyro = X_seq[:, :, 9:12]
    
    # 返回完整数据集 inputs
    return [X_dvl, X_att, X_acc, X_gyro], Y_seq, GT_seq, scaler_y

# ==================== 模型定义 (4种对比模型) ====================

def build_feature_extraction_branch(input_shape, name):
    inputs = Input(shape=input_shape, name=f'input_{name}')
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    shortcut = layers.Conv1D(256, 1, padding='same')(inputs)
    x = layers.Add()([x, shortcut])
    return inputs, x

# 1. Multi-branch CNN
def build_multi_cnn(window_size):
    input_shape = (window_size, 3)
    in_dvl, out_dvl = build_feature_extraction_branch(input_shape, 'dvl')
    in_att, out_att = build_feature_extraction_branch(input_shape, 'att')
    in_acc, out_acc = build_feature_extraction_branch(input_shape, 'acc')
    in_gyro, out_gyro = build_feature_extraction_branch(input_shape, 'gyro')
    x = layers.Concatenate(axis=-1)([out_dvl, out_att, out_acc, out_gyro])
    x = layers.Flatten()(x) 
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='linear')(x)
    return models.Model([in_dvl, in_att, in_acc, in_gyro], outputs, name="Multi_CNN")

# 2. 2-Layer LSTM
def build_pure_lstm(window_size):
    input_shape = (window_size, 3)
    in_dvl = Input(shape=input_shape, name='in_dvl')
    in_att = Input(shape=input_shape, name='in_att')
    in_acc = Input(shape=input_shape, name='in_acc')
    in_gyro = Input(shape=input_shape, name='in_gyro')
    x = layers.Concatenate(axis=-1)([in_dvl, in_att, in_acc, in_gyro])
    x = layers.LSTM(256, return_sequences=True)(x)
    x = layers.LSTM(256, return_sequences=False)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='linear')(x)
    return models.Model([in_dvl, in_att, in_acc, in_gyro], outputs, name="2Layer_LSTM")

# 3. 2-Layer TCN
def build_tcn(window_size):
    def tcn_block(x, filters, kernel_size, dilation_rate):
        prev = x
        x = layers.Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)
        x = layers.SpatialDropout1D(0.1)(x)
        if prev.shape[-1] != filters:
            prev = layers.Conv1D(filters, 1, padding='same')(prev)
        return layers.Add()([x, prev])

    input_shape = (window_size, 3)
    in_dvl = Input(shape=input_shape)
    in_att = Input(shape=input_shape)
    in_acc = Input(shape=input_shape)
    in_gyro = Input(shape=input_shape)
    x = layers.Concatenate(axis=-1)([in_dvl, in_att, in_acc, in_gyro])
    x = tcn_block(x, 32, 5, 1)
    x = tcn_block(x, 32, 5, 2)
    x = tcn_block(x, 32, 5, 4)
    x = tcn_block(x, 32, 5, 8)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='linear')(x)
    return models.Model([in_dvl, in_att, in_acc, in_gyro], outputs, name="2Layer_TCN")

# 4. IONet
def build_ionet(window_size):
    input_shape = (window_size, 3)
    in_dvl = Input(shape=input_shape)
    in_att = Input(shape=input_shape)
    in_acc = Input(shape=input_shape)
    in_gyro = Input(shape=input_shape)
    x = layers.Concatenate(axis=-1)([in_dvl, in_att, in_acc, in_gyro])
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=False))(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation='linear')(x)
    return models.Model([in_dvl, in_att, in_acc, in_gyro], outputs, name="IONet")

# ==================== 主运行逻辑 ====================
if __name__ == '__main__':
    # 1. 准备数据
    inputs, targets, gt_positions, scaler = load_and_process_data(CSV_FILE, WINDOW_SIZE)
    
    # 2. 划分数据集
    idx = np.arange(len(targets))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, shuffle=False)
    
    X_train = [inp[train_idx] for inp in inputs]
    Y_train = targets[train_idx]
    
    X_test = [inp[test_idx] for inp in inputs]
    Y_test = targets[test_idx]
    
    # 测试集真值 (用于计算指标)
    GT_test = gt_positions[test_idx]
    
    # 3. 准备对比
    models_to_train = {
        "Multi-branch CNN": build_multi_cnn,
        "2-Layer LSTM":     build_pure_lstm,
        "2-Layer TCN":      build_tcn,
        "IONet (Bi-LSTM)":  build_ionet
    }
    
    results_list = []
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    config = {
            "font.family": 'serif',
            "font.serif": ['Times New Roman', 'Times', 'serif'],
            "mathtext.fontset": 'stix',
            "font.size": 8,
            "axes.unicode_minus": False 
        }
    plt.rcParams.update(config)
    
    #完整轨迹
    start_pos_global = gt_positions[0]
    traj_gt_full = np.vstack([start_pos_global, gt_positions])
    
    # 真值
    ax.plot(traj_gt_full[:, 0], traj_gt_full[:, 1], 'k-', lw=1.5, label='Ground Truth', zorder=10)
    
    print(f"\n{'='*20} 开始对比实验 {'='*20}")
    
    styles = {
        "Multi-branch CNN": {'c': '#1f77b4', 'ls': '--'},
        "2-Layer LSTM":     {'c': '#ff7f0e', 'ls': '-.'},
        "2-Layer TCN":      {'c': '#d62728', 'ls': '--'},
        "IONet (Bi-LSTM)":  {'c': '#9467bd', 'ls': '--'}
    }

    for name, build_func in models_to_train.items():
        print(f"正在训练模型: {name} ...")
        
        # 构建与训练
        model = build_func(WINDOW_SIZE)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        
        # 预测全量数据
        preds_all = model.predict(inputs, verbose=1)
        preds_delta_all = scaler.inverse_transform(preds_all)
        traj_full = np.vstack([start_pos_global, np.cumsum(preds_delta_all, axis=0) + start_pos_global])
        
        # 计算测试集指标
        test_len = len(Y_test)
        preds_test = preds_all[-test_len:] 
        preds_delta_test = scaler.inverse_transform(preds_test)
        start_pos_test = GT_test[0]
        traj_test_rec = np.cumsum(preds_delta_test, axis=0) + start_pos_test
        
        final_err = np.linalg.norm(GT_test[-1] - traj_test_rec[-1])
        rmse = np.sqrt(np.mean(np.sum((GT_test - traj_test_rec)**2, axis=1)))
        
        results_list.append({
            "Model": name,
            "Final Error (m)": final_err,
            "Test RMSE (m)": rmse
        })
        
        # 画预测轨迹
        st = styles.get(name, {'c': 'r', 'ls': '--'}) # 如果没定义样式默认用红色
        ax.plot(traj_full[:, 0], traj_full[:, 1], label=f"{name} (RMSE:{rmse:.2f}m)",
                color=st['c'], linestyle=st['ls'], lw=1.0, alpha=0.9)

    # ================= 关键修正：调整比例与美化 =================
    
    ax.set_title('Trajectory Comparison', fontsize=8, fontweight='bold', pad=15)
    ax.set_xlabel('East (m)', fontsize=8)
    ax.set_ylabel('North (m)', fontsize=8)
    
    # 核心修正：强制横纵轴单位长度一致 (1米东 = 1米北)
    # adjustable='datalim' 会根据数据范围调整坐标轴边界，而不是强行扭曲盒子
    ax.set_aspect('equal', adjustable='datalim')
    
    # 添加网格线
    ax.grid(True, linestyle=':', alpha=0.6, color='gray')
    
    # 图例设置
    ax.legend(fontsize=5, loc='best', frameon=True, shadow=False)
    
    # 自动调整布局，防止标签被切掉
    plt.tight_layout()
    
    # 保存图片
    print("\n正在保存图片...")
    plt.savefig('comparison_full_fixed.png', dpi=300)
    plt.show()