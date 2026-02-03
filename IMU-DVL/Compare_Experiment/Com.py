import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import time
import json
from datetime import datetime

# ==========================================
# 0. 硬件配置
# ==========================================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ 使用 Apple MPS (Metal GPU 加速)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✅ 使用 NVIDIA CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ 使用 CPU")

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==========================================
# 1. 数据预处理函数 (处理多文件)
# ==========================================
def load_and_preprocess_file(path):
    """
    读取单个 CSV，清洗深度异常，并计算该文件内部的位移增量 dx, dy
    """
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return None
        
    df = pd.read_csv(path)
    
    if len(df) == 0:
        print(f"❌ 文件为空: {path}")
        return None
    
    # 1. 清洗深度 (去除 88888 异常值)
    median_depth = df.loc[df['depth'] < 1000, 'depth'].median()
    if pd.isna(median_depth):
        median_depth = 0.0
    df.loc[df['depth'] > 1000, 'depth'] = median_depth
    
    # 2. 计算位移增量 (GT Delta)
    df['dx'] = df['gt_px'].diff().fillna(0)
    df['dy'] = df['gt_py'].diff().fillna(0)
    
    # 3. 丢弃第一行
    df = df.iloc[1:].reset_index(drop=True)
    
    return df

# ==========================================
# 2. 数据集类 (ROVDataset)
# ==========================================
class ROVDataset(Dataset):
    def __init__(self, df, seq_len=200, scaler_x=None, scaler_y=None, fit_scaler=False, file_boundaries=None):
        """
        🔧 修复:标签取 i+seq_len-1,这样序列 [i:i+seq_len] 预测的是第 i+seq_len-1 个时刻的位移
        """
        self.df = df.copy()
        
        features = ['acce_x', 'acce_y', 'acce_z', 'gyro_x', 'gyro_y', 'gyro_z', 
                    'dvl_vx', 'dvl_vy', 'dvl_vz', 'depth']
        targets = ['dx', 'dy'] 

        x_data = self.df[features].astype(np.float32).values
        y_data = self.df[targets].astype(np.float32).values

        self.scaler_x = scaler_x if scaler_x is not None else StandardScaler()
        self.scaler_y = scaler_y if scaler_y is not None else StandardScaler()

        if fit_scaler:
            x_data = self.scaler_x.fit_transform(x_data)
            y_data = self.scaler_y.fit_transform(y_data)
        else:
            if scaler_x is None or scaler_y is None:
                raise ValueError("测试集必须传入训练集拟合好的 scaler")
            x_data = self.scaler_x.transform(x_data)
            y_data = self.scaler_y.transform(y_data)

        self.sequences = []
        self.labels = []
        
        if file_boundaries is None:
            # 单文件情况 (测试集)
            if len(x_data) < seq_len:
                raise ValueError(f"测试集长度 {len(x_data)} 必须 >= 序列长度 {seq_len}")
            # 🔧 修复:标签索引改为 i+seq_len-1
            for i in range(len(x_data) - seq_len + 1):
                self.sequences.append(x_data[i : i + seq_len])
                self.labels.append(y_data[i + seq_len - 1])
        else:
            # 多文件情况 (训练集)
            start_idx = 0
            for end_idx in file_boundaries:
                file_len = end_idx - start_idx
                if file_len >= seq_len:
                    # 🔧 修复:范围改为 end_idx - seq_len + 1
                    for i in range(start_idx, end_idx - seq_len + 1):
                        self.sequences.append(x_data[i : i + seq_len])
                        self.labels.append(y_data[i + seq_len - 1])
                start_idx = end_idx
        
        if len(self.sequences) == 0:
            raise ValueError("生成的序列数为 0，请检查数据长度和 seq_len 设置")
            
        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ==========================================
# 3. 模型定义
# ==========================================
INPUT_LEN = 200
INPUT_CHANNELS = 10
OUTPUT_SIZE = 2 

class MultiBranchCNN(nn.Module):
    def __init__(self):
        super(MultiBranchCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU()
        )
        self.flatten_dim = 256 * INPUT_LEN
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, OUTPUT_SIZE)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TwoLayerLSTM(nn.Module):
    def __init__(self):
        super(TwoLayerLSTM, self).__init__()
        self.lstm = nn.LSTM(INPUT_CHANNELS, 256, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, OUTPUT_SIZE)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class IONet(nn.Module):
    def __init__(self):
        super(IONet, self).__init__()
        self.lstm = nn.LSTM(INPUT_CHANNELS, 96, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(96*2, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, OUTPUT_SIZE)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class TwoLayerTCN(nn.Module):
    def __init__(self, input_channels=10, output_size=2, num_channels=[32, 64], kernel_size=3, dropout=0.2):
        super(TwoLayerTCN, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            in_channels = input_channels if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        self.tcn = nn.ModuleList(layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.tcn:
            x = layer(x)
            if x.size(2) > INPUT_LEN:
                x = x[:, :, :INPUT_LEN]
        x = torch.mean(x, dim=2)
        return self.fc(x)

# ==========================================
# 4. 训练与评估逻辑
# ==========================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_experiment(model_class, name, train_dataset, test_dataset, scaler_y, 
                   test_df_original, epochs=50, patience=10, save_dir="./checkpoints"):
    """
    🔧 完整修复:轨迹重建、N-SRMSE计算、推理时间测量
    """
    print(f"\n🚀 开始训练: {name}")
    os.makedirs(save_dir, exist_ok=True)
    
    use_pin_memory = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=use_pin_memory)

    model = model_class().to(DEVICE)
    print(f"📊 模型参数量: {count_parameters(model):,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_test_mse = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(save_dir, f"{name}_best.pth")
    
    history = {"train_loss": [], "test_mse": [], "epochs": []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Test Evaluation
        model.eval()
        test_mse = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                test_mse += criterion(pred, y).item()
        test_mse /= len(test_loader)
        
        history["train_loss"].append(avg_train_loss)
        history["test_mse"].append(test_mse)
        history["epochs"].append(epoch + 1)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | TrainLoss: {avg_train_loss:.5f} | TestMSE: {test_mse:.5e}", end="")
        
        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'test_mse': test_mse,
                'train_loss': avg_train_loss
            }, best_model_path)
            print(f" ✅ [Saved]")
        else:
            patience_counter += 1
            print(f" (pat: {patience_counter})")
            if patience_counter >= patience:
                print(f"\n⏹️ Early Stopping at epoch {epoch+1}")
                break
    
    # 🔧 保存训练历史
    history_path = os.path.join(save_dir, f"{name}_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    
    # Load Best Model
    print(f"\n📂 加载最佳模型 (Epoch {best_epoch})...")
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 🔧 修复:推理时间测量 (需要 GPU 同步)
    all_preds = []
    all_targets = []
    inference_times = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            
            # Warm up
            if len(inference_times) == 0:
                _ = model(x)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
            
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            pred = model(x)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            inference_times.append((end - start) * 1000 / x.size(0))
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    pred_delta = scaler_y.inverse_transform(np.concatenate(all_preds))
    target_delta = scaler_y.inverse_transform(np.concatenate(all_targets))
    
    # 🔧 修复:轨迹重建逻辑
    # 数据集第 0 个样本预测的是索引 INPUT_LEN-1 的位移 dx[t]
    # 物理含义: dx[t] = pos[t] - pos[t-1]
    # 要重建 pos[t]，必须从 pos[t-1] 开始积分
    # 因此起点索引必须是 INPUT_LEN - 2
    
    start_idx = INPUT_LEN - 2  # 关键修复: 从 L-2 开始，而不是 L-1
    
    test_start_pos = (test_df_original.iloc[start_idx]['gt_px'], 
                      test_df_original.iloc[start_idx]['gt_py'])
    
    # 积分重建轨迹 (从起点开始累加预测的位移)
    pred_path_x = np.concatenate([[test_start_pos[0]], test_start_pos[0] + np.cumsum(pred_delta[:, 0])])
    pred_path_y = np.concatenate([[test_start_pos[1]], test_start_pos[1] + np.cumsum(pred_delta[:, 1])])
    true_path_x = np.concatenate([[test_start_pos[0]], test_start_pos[0] + np.cumsum(target_delta[:, 0])])
    true_path_y = np.concatenate([[test_start_pos[1]], test_start_pos[1] + np.cumsum(target_delta[:, 1])])
    
    # 🔧 修复:验证重建轨迹 (对应原始数据的 [INPUT_LEN-2, ...])
    # 现在长度应该完美匹配，不会出现少一个点的情况
    gt_x = test_df_original.iloc[start_idx:start_idx+len(true_path_x)]['gt_px'].values
    gt_y = test_df_original.iloc[start_idx:start_idx+len(true_path_y)]['gt_py'].values
    
    assert len(gt_x) == len(true_path_x), f"长度不匹配: GT={len(gt_x)}, Rebuilt={len(true_path_x)}"
    assert np.allclose(true_path_x, gt_x, atol=1e-2), f"X轴重建错误! 最大误差: {np.max(np.abs(true_path_x - gt_x))}"
    assert np.allclose(true_path_y, gt_y, atol=1e-2), f"Y轴重建错误! 最大误差: {np.max(np.abs(true_path_y - gt_y))}"
    print(f"✅ 轨迹重建验证通过 (轨迹点数: {len(true_path_x)})")
    
    # 🔧 修复:计算指标
    error_x = pred_path_x - true_path_x
    error_y = pred_path_y - true_path_y
    euclidean_distances = np.sqrt(error_x**2 + error_y**2)
    
    rmse = np.sqrt(np.mean(euclidean_distances**2))
    
    # 🔧 修复:N-SRMSE 应该基于整条真实轨迹长度
    full_traj_len = np.sum(np.sqrt(np.diff(gt_x)**2 + np.diff(gt_y)**2))
    n_srmse = (rmse / full_traj_len) * 100 if full_traj_len > 0 else 0
    
    end_error = euclidean_distances[-1]
    avg_time = np.mean(inference_times[1:]) if len(inference_times) > 1 else inference_times[0]
    
    metrics = {
        "N-SRMSE": n_srmse,
        "RMSE": rmse,
        "Max Error": np.max(euclidean_distances),
        "End Point Error": end_error,
        "Time(ms)": avg_time,
        "Best MSE": best_test_mse,
        "Best Epoch": best_epoch,
        "Params": count_parameters(model)
    }
    
    # 🔧 保存预测轨迹
    traj_df = pd.DataFrame({
        'pred_x': pred_path_x,
        'pred_y': pred_path_y,
        'true_x': true_path_x,
        'true_y': true_path_y,
        'error': euclidean_distances
    })
    traj_path = os.path.join(save_dir, f"{name}_trajectory.csv")
    traj_df.to_csv(traj_path, index=False)
    
    return true_path_x, true_path_y, pred_path_x, pred_path_y, metrics, history

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    # 🔧 创建带时间戳的实验文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"/Users/book./Desktop/book/2021_2025/科研/experiment_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"📁 实验结果将保存到: {exp_dir}")
    
    file_paths = [
        "/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/data/alldata/SenseINS_aligned_1.csv",
        "/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/data/alldata/SenseINS_aligned_2.csv",
        "/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/data/alldata/SenseINS_aligned_3.csv",
        "/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/data/alldata/SenseINS_aligned_4.csv",
        "/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/data/alldata/SenseINS_aligned_5.csv"
    ]
    
    train_dfs = []
    test_df = None
    
    print("⏳ 正在预处理 5 个数据文件...")
    
    for i, path in enumerate(file_paths):
        print(f"  -> 处理: {os.path.basename(path)}")
        df = load_and_preprocess_file(path)
        if df is None or len(df) == 0:
            print(f"  ⚠️ 跳过空文件")
            continue
        
        if len(df) < INPUT_LEN:
            print(f"  ⚠️ 文件长度 {len(df)} < {INPUT_LEN}，跳过")
            continue
        
        if i < 4:
            train_dfs.append(df)
        else:
            test_df = df
            
    if not train_dfs or test_df is None:
        print("❌ 数据加载失败，请检查路径。")
        exit()

    file_boundaries = []
    cumulative_len = 0
    for df in train_dfs:
        cumulative_len += len(df)
        file_boundaries.append(cumulative_len)
    
    train_df_all = pd.concat(train_dfs, ignore_index=True)
    print(f"✅ 训练集样本数: {len(train_df_all)} (来自 {len(train_dfs)} 个文件)")
    print(f"✅ 测试集样本数: {len(test_df)}")
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    print("\n构建 Dataset...")
    try:
        train_dataset = ROVDataset(train_df_all, seq_len=INPUT_LEN, 
                                   scaler_x=scaler_x, scaler_y=scaler_y, 
                                   fit_scaler=True, file_boundaries=file_boundaries)
        test_dataset = ROVDataset(test_df, seq_len=INPUT_LEN, 
                                  scaler_x=scaler_x, scaler_y=scaler_y, 
                                  fit_scaler=False, file_boundaries=None)
    except ValueError as e:
        print(f"❌ Dataset 构建失败: {e}")
        exit()
    
    print(f"✅ 训练集序列数: {len(train_dataset)}")
    print(f"✅ 测试集序列数: {len(test_dataset)}")

    # 🔧 保存实验配置
    config = {
        "timestamp": timestamp,
        "device": str(DEVICE),
        "input_len": INPUT_LEN,
        "batch_size": 256,
        "epochs": 50,
        "patience": 10,
        "learning_rate": 0.001,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "train_files": [os.path.basename(p) for p in file_paths[:4]],
        "test_file": os.path.basename(file_paths[4])
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    models = [
        (MultiBranchCNN, "CNN"),
        (TwoLayerLSTM, "LSTM"),
        (IONet, "IONet"),
        (TwoLayerTCN, "TCN")
    ]
    
    plt.figure(figsize=(12, 5))
    
    # 子图1: 轨迹对比
    plt.subplot(1, 2, 1)
    results = {}
    histories = {}
    
    for m_class, name in models:
        true_x, true_y, pred_x, pred_y, metrics, history = run_experiment(
            m_class, name, train_dataset, test_dataset, scaler_y, 
            test_df_original=test_df, epochs=50, save_dir=checkpoint_dir
        )
        results[name] = metrics
        histories[name] = history
        
        if name == "CNN":
            plt.plot(true_x, true_y, 'k--', linewidth=2, label='Ground Truth')
        plt.plot(pred_x, pred_y, linewidth=1.5, label=f'{name}')

    plt.title("Trajectory Comparison on Test Set")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    # 子图2: 训练曲线
    plt.subplot(1, 2, 2)
    for name, hist in histories.items():
        plt.plot(hist['epochs'], hist['test_mse'], label=name, marker='o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Test MSE')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    fig_path = os.path.join(exp_dir, "results_comparison.png")
    plt.savefig(fig_path, dpi=300)
    print(f"\n✅ 轨迹图已保存: {fig_path}")
    
    # 🔧 保存结果表格
    results_df = pd.DataFrame(results).T
    results_df.index.name = 'Model'
    results_df = results_df.reset_index()
    
    # 重新排列列顺序
    col_order = ['Model', 'Params', 'Best Epoch', 'Best MSE', 'N-SRMSE', 
                 'RMSE', 'Max Error', 'End Point Error', 'Time(ms)']
    results_df = results_df[col_order]
    
    csv_path = os.path.join(exp_dir, "results_summary.csv")
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"✅ 结果表格已保存: {csv_path}")
    
    # 打印最终对比表
    print(f"\n{'='*120}")
    print(f"{'Model':<10} {'Params':<12} {'N-SRMSE(%)':<15} {'RMSE(m)':<15} {'MaxErr(m)':<15} {'EndErr(m)':<15} {'Time(ms)':<12}")
    print(f"{'-'*120}")
    for name, m in results.items():
        print(f"{name:<10} {m['Params']:<12,} {m['N-SRMSE']:<15.4f} {m['RMSE']:<15.4f} "
              f"{m['Max Error']:<15.4f} {m['End Point Error']:<15.4f} {m['Time(ms)']:<12.4f}")
    print(f"{'='*120}")
    
    # 🔧 找出最佳模型
    best_model_name = min(results.items(), key=lambda x: x[1]['N-SRMSE'])[0]
    print(f"\n🏆 最佳模型: {best_model_name} (N-SRMSE: {results[best_model_name]['N-SRMSE']:.4f}%)")
    print(f"\n📦 所有结果已保存到: {exp_dir}")
    print(f"   - 模型权重: {checkpoint_dir}/")
    print(f"   - 结果表格: {csv_path}")
    print(f"   - 训练曲线: {fig_path}")
    print(f"   - 实验配置: {os.path.join(exp_dir, 'config.json')}")