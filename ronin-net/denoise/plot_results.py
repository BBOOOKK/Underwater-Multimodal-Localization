# -*- coding: utf-8 -*-
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
import pickle
import sys

# ================= CONFIGURATION =================
# Please verify these paths are correct
RESULTS_PATH = '/home/ouc/ronin-net/output/denoise_results/2026_01_26_15_26_28/04/results.p'
CSV_PATH = '/home/ouc/ronin-net/data/all_data/04/SenseINS_aligned.csv'
DT = 0.01
SAVE_DIR = './plots_05'
# =================================================

def safe_load_results(path):
    """Try loading with pickle first, then torch."""
    print(f"[1/4] Loading results file: {os.path.basename(path)} ...")
    if not os.path.exists(path):
        print(f"Error: File not found -> {path}")
        return None

    try:
        # Try Method A: Pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e1:
        print(f"Pickle load failed ({e1}), trying Torch...")
        try:
            # Try Method B: Torch
            return torch.load(path, map_location='cpu')
        except Exception as e2:
            print(f"Error: Failed to load file. Reason: {e2}")
            return None

def load_data(results_path, csv_path):
    # 1. Load Model Prediction
    res = safe_load_results(results_path)
    if res is None: return None, None
    
    # Extract data
    if isinstance(res, dict) and 'hat_xs' in res:
        hat_xs = res['hat_xs']
    else:
        hat_xs = res
    
    if isinstance(hat_xs, torch.Tensor):
        hat_xs = hat_xs.detach().cpu().numpy()
    
    print(f"Model prediction shape: {hat_xs.shape}")

    # 2. Load Ground Truth CSV
    print(f"[2/4] Loading GT file: {os.path.basename(csv_path)} ...")
    if not os.path.exists(csv_path):
        print(f"Error: File not found -> {csv_path}")
        return None, None

    try:
        df = pd.read_csv(csv_path)
        
        # Extract Orientation
        if 'gt_q_w' in df.columns:
            gt_quat = df[['gt_q_x', 'gt_q_y', 'gt_q_z', 'gt_q_w']].values
            r_gt = R.from_quat(gt_quat)
            gt_euler = r_gt.as_euler('xyz', degrees=True)
        elif 'roll' in df.columns:
            gt_euler = df[['roll', 'pitch', 'yaw']].values
            # Unwrap
            gt_euler = np.unwrap(np.deg2rad(gt_euler), axis=0) * 180 / np.pi
        else:
            print("Error: No orientation data (roll/pitch/yaw or gt_q_*) found in CSV.")
            return None, None

        # Extract Raw Gyro
        gyro_cols = ['gyro_x', 'gyro_y', 'gyro_z']
        if not all(c in df.columns for c in gyro_cols):
            gyro_cols = ['gryX', 'gryY', 'gryZ'] 
        
        if all(c in df.columns for c in gyro_cols):
            raw_gyro = df[gyro_cols].values
        else:
            print("Warning: Raw gyro columns not found, using zeros.")
            raw_gyro = np.zeros_like(gt_euler)
        
        # Extract Timestamp
        if 'timestamp' in df.columns:
            ts = df['timestamp'].values
            ts = ts - ts[0]
        else:
            ts = np.arange(len(df)) * DT

        print(f"GT data loaded, shape: {gt_euler.shape}")
        return hat_xs, (ts, raw_gyro, gt_euler)

    except Exception as e:
        print(f"CSV Parse Error: {e}")
        return None, None

def print_metrics(raw_err, net_err, raw_gyro, net_gyro):
    """Print detailed quantitative metrics"""
    print("\n" + "="*80)
    print("QUALITY ASSESSMENT REPORT")
    print("="*80)
    
    axis_names = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    
    # 1. Cumulative Drift Error
    print(f"{'Metric':<20} | {'Axis':<10} | {'Raw (Noisy)':<12} | {'GyroNet':<15} | {'Improvement':<10}")
    print("-" * 80)
    
    for i in range(3):
        raw_final = raw_err[-1, i]
        net_final = net_err[-1, i]
        improv = (1 - net_final / (raw_final + 1e-6)) * 100
        print(f"{'Final Drift (deg)':<20} | {axis_names[i]:<10} | {raw_final:12.4f} | {net_final:15.4f} | {improv:9.1f}%")
        
    print("-" * 80)

    # 2. RMSE
    for i in range(3):
        raw_rmse = np.sqrt(np.mean(raw_err[:, i]**2))
        net_rmse = np.sqrt(np.mean(net_err[:, i]**2))
        improv = (1 - net_rmse / (raw_rmse + 1e-6)) * 100
        print(f"{'RMSE Error (deg)':<20} | {axis_names[i]:<10} | {raw_rmse:12.4f} | {net_rmse:15.4f} | {improv:9.1f}%")

    print("-" * 80)
    print("Note: Improvement > 0% means the model is working.\n")

def plot_comparison(hat_xs, raw_data, save_dir):
    ts, raw_gyro, gt_euler = raw_data
    
    # Align Data Length
    n = min(len(hat_xs), len(raw_gyro), len(gt_euler))
    print(f"[3/4] Data Alignment: Valid Length N = {n}")
    
    if n == 0:
        print("Error: Valid data length is 0!")
        return

    hat_xs = hat_xs[:n]
    raw_gyro = raw_gyro[:n]
    gt_euler = gt_euler[:n]
    ts = ts[:n]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(">>> Calculating metrics...")
    # Integration
    raw_integ = np.cumsum(raw_gyro, axis=0) * DT * 180 / np.pi
    net_integ = np.cumsum(hat_xs, axis=0) * DT * 180 / np.pi
    gt_integ = gt_euler - gt_euler[0]
    
    # Calculate Difference
    def calc_diff(a, b):
        diff = np.abs(a - b)
        return np.minimum(diff, 360 - diff)

    raw_err = calc_diff(raw_integ, gt_integ)
    net_err = calc_diff(net_integ, gt_integ)

    # ------------------ Print Report ------------------
    print_metrics(raw_err, net_err, raw_gyro, hat_xs)
    # --------------------------------------------------

    print(f"[4/4] Generating plots to {save_dir} ...")
    
    # Plot 1: Gyro Detail
    start = min(1000, n//2)
    end = min(start + 500, n)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        axes[i].plot(ts[start:end], raw_gyro[start:end, i], 'c-', alpha=0.3, label='Raw')
        axes[i].plot(ts[start:end], hat_xs[start:end, i], 'r-', label='GyroNet')
        axes[i].legend()
    plt.savefig(os.path.join(save_dir, '1_gyro_detail.png'))
    plt.close()

    # Plot 2: Orientation Integration
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for i in range(3):
        axes[i].plot(ts, gt_integ[:, i], 'k-', linewidth=2, label='GT')
        axes[i].plot(ts, raw_integ[:, i], 'c--', label='Raw')
        axes[i].plot(ts, net_integ[:, i], 'r-', label='GyroNet')
        axes[i].legend()
    plt.savefig(os.path.join(save_dir, '2_orientation.png'))
    plt.close()
    
    # Plot 3: Error
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for i in range(3):
        axes[i].plot(ts, raw_err[:, i], 'c--', alpha=0.6, label='Raw Err')
        axes[i].plot(ts, net_err[:, i], 'r-', label='GyroNet Err')
        axes[i].legend()
    plt.savefig(os.path.join(save_dir, '3_error.png'))
    plt.close()
    
    print("All Done!")

if __name__ == '__main__':
    hat_xs, raw_data = load_data(RESULTS_PATH, CSV_PATH)
    if hat_xs is not None and raw_data is not None:
        plot_comparison(hat_xs, raw_data, SAVE_DIR)
    else:
        print("\nProgram terminated due to loading errors.")