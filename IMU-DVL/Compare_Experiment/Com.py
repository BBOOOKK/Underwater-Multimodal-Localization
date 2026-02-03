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

# Device configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Fix random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def load_and_preprocess_file(path):
    """Load and preprocess a single CSV file."""
    if not os.path.exists(path):
        print(f"File does not exist: {path}")
        return None
        
    df = pd.read_csv(path)
    
    if len(df) == 0:
        print(f"File is empty: {path}")
        return None
    
    # 1. Clean depth data (remove 88888 outliers)
    median_depth = df.loc[df['depth'] < 1000, 'depth'].median()
    if pd.isna(median_depth):
        median_depth = 0.0
    df.loc[df['depth'] > 1000, 'depth'] = median_depth
    
    # 2. Calculate displacement increments (GT Delta)
    df['dx'] = df['gt_px'].diff().fillna(0)
    df['dy'] = df['gt_py'].diff().fillna(0)
    
    # 3. Drop the first row
    df = df.iloc[1:].reset_index(drop=True)
    
    return df

class ROVDataset(Dataset):
    """Dataset class for ROV trajectory prediction."""
    def __init__(self, df, seq_len=200, scaler_x=None, scaler_y=None, fit_scaler=False, file_boundaries=None):
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
                raise ValueError("Test set must use scalers fitted on training data")
            x_data = self.scaler_x.transform(x_data)
            y_data = self.scaler_y.transform(y_data)

        self.sequences = []
        self.labels = []
        
        if file_boundaries is None:
            # Single file case (test set)
            if len(x_data) < seq_len:
                raise ValueError(f"Test set length {len(x_data)} must be >= sequence length {seq_len}")
            # Fix: label index changed to i+seq_len-1
            for i in range(len(x_data) - seq_len + 1):
                self.sequences.append(x_data[i : i + seq_len])
                self.labels.append(y_data[i + seq_len - 1])
        else:
            # Multiple files case (training set)
            start_idx = 0
            for end_idx in file_boundaries:
                file_len = end_idx - start_idx
                if file_len >= seq_len:
                    # Fix: range changed to end_idx - seq_len + 1
                    for i in range(start_idx, end_idx - seq_len + 1):
                        self.sequences.append(x_data[i : i + seq_len])
                        self.labels.append(y_data[i + seq_len - 1])
                start_idx = end_idx
        
        if len(self.sequences) == 0:
            raise ValueError("Generated 0 sequences, check data length and seq_len settings")
            
        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Model definitions
INPUT_LEN = 200
INPUT_CHANNELS = 10
OUTPUT_SIZE = 2 

class MultiBranchCNN(nn.Module):
    """Multi-branch CNN model for trajectory prediction."""
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
    """Two-layer LSTM model for trajectory prediction."""
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
    """IONet model with bidirectional LSTM for trajectory prediction."""
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
    """Two-layer Temporal Convolutional Network for trajectory prediction."""
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

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_experiment(model_class, name, train_dataset, test_dataset, scaler_y, test_df_original, epochs=50, patience=10, save_dir="./checkpoints"):
    print(f"\nStarting training: {name}")
    os.makedirs(save_dir, exist_ok=True)
    
    use_pin_memory = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=use_pin_memory)

    model = model_class().to(DEVICE)
    print(f"Model parameters: {count_parameters(model):,}")
    
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
            print(f" [Saved]")
        else:
            patience_counter += 1
            print(f" (pat: {patience_counter})")
            if patience_counter >= patience:
                print(f"\nEarly Stopping at epoch {epoch+1}")
                break
    
    # Save training history
    history_path = os.path.join(save_dir, f"{name}_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    
    # Load Best Model
    print(f"\nLoading best model (Epoch {best_epoch})...")
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Inference time measurement
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
    
    start_idx = INPUT_LEN - 2  # Critical fix: start from L-2, not L-1
    
    test_start_pos = (test_df_original.iloc[start_idx]['gt_px'], 
                      test_df_original.iloc[start_idx]['gt_py'])
    
    # Integrate to reconstruct trajectory (accumulate predicted displacements from start point)
    pred_path_x = np.concatenate([[test_start_pos[0]], test_start_pos[0] + np.cumsum(pred_delta[:, 0])])
    pred_path_y = np.concatenate([[test_start_pos[1]], test_start_pos[1] + np.cumsum(pred_delta[:, 1])])
    true_path_x = np.concatenate([[test_start_pos[0]], test_start_pos[0] + np.cumsum(target_delta[:, 0])])
    true_path_y = np.concatenate([[test_start_pos[1]], test_start_pos[1] + np.cumsum(target_delta[:, 1])])
    
    gt_x = test_df_original.iloc[start_idx:start_idx+len(true_path_x)]['gt_px'].values
    gt_y = test_df_original.iloc[start_idx:start_idx+len(true_path_y)]['gt_py'].values
    
    assert len(gt_x) == len(true_path_x), f"Length mismatch: GT={len(gt_x)}, Rebuilt={len(true_path_x)}"
    assert np.allclose(true_path_x, gt_x, atol=1e-2), f"X-axis reconstruction error! Max error: {np.max(np.abs(true_path_x - gt_x))}"
    assert np.allclose(true_path_y, gt_y, atol=1e-2), f"Y-axis reconstruction error! Max error: {np.max(np.abs(true_path_y - gt_y))}"
    print(f"Trajectory reconstruction verified (trajectory points: {len(true_path_x)})")
    
    # Calculate metrics
    error_x = pred_path_x - true_path_x
    error_y = pred_path_y - true_path_y
    euclidean_distances = np.sqrt(error_x**2 + error_y**2)
    
    rmse = np.sqrt(np.mean(euclidean_distances**2))
    
    # N-SRMSE should be based on full true trajectory length
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
    
    # Save predicted trajectory
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

if __name__ == "__main__":
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"/Users/book./Desktop/book/2021_2025/科研/experiment_{timestamp}"
    os
