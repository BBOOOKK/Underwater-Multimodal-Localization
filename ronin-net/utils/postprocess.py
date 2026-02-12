"""
* This file is part of RNIN-VIO
* (Restored Version: Full GT Plotting + Interactive plt.show())
"""
import numpy as np
from os import path as osp
from scipy.interpolate import interp1d
import logging
import matplotlib
# [还原] 注释掉 Agg 模式，这样你在本地运行时可以看到弹窗
# matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def pose_integrate(cfg, dataset, preds):
    window_time = cfg['model_param']['window_time']
    imu_freq = cfg['data']['imu_freq']
    seq_len = cfg['train']['seq_len']

    dp_t = window_time
    pred_vels = preds / dp_t

    ind = np.array([i[1] for i in dataset.index_map], dtype=int)
    delta_int = int(
        window_time * imu_freq / 2.0
    )
    delta_int += int((seq_len - 1) * window_time * imu_freq)
    if not (window_time * imu_freq / 2.0).is_integer():
        logging.info("Trajectory integration point is not centered.")
    ind_intg = ind + delta_int

    ts = dataset.ts[0]
    
    # [关键修复] 严格对齐时间戳和速度预测的长度
    ts_segment = ts[ind_intg]
    real_dts = np.diff(ts_segment).reshape(-1, 1) # 长度 N-1
    
    pos_intg = np.zeros_like(pred_vels) 
    pos_intg[0] = dataset.gt_pos[0][ind_intg[0], 0:pred_vels.shape[1]]

    n_steps = len(real_dts)
    n_valid = min(n_steps, len(pred_vels) - 1)
    
    valid_vels = pred_vels[:n_valid]
    valid_dts = real_dts[:n_valid]
    
    increments = valid_vels * valid_dts
    pos_intg[1:n_valid+1] = np.cumsum(increments, axis=0) + pos_intg[0]
    
    pos_intg = pos_intg[:n_valid+1]
    ts_in_range = ts_segment[:n_valid+1]

    # 截取对应时间段的 GT
    pos_gt = dataset.gt_pos[0][ind_intg[0] : ind_intg[0]+n_valid+1, 0:pred_vels.shape[1]]

    # [保留] 获取全量 GT (用于画图展示全貌)
    pos_gt_full = dataset.gt_pos[0][:, 0:pred_vels.shape[1]]

    traj_attr_dict = {
        "ts": ts_in_range,
        "pos_pred": pos_intg,
        "pos_gt": pos_gt,
        "pos_gt_full": pos_gt_full, 
    }

    return traj_attr_dict
    
def compute_plot_dict(sample_freq, net_attr_dict, traj_attr_dict):

    ts = traj_attr_dict["ts"]
    pos_pred = traj_attr_dict["pos_pred"]
    pos_gt = traj_attr_dict["pos_gt"]
    pos_gt_full = traj_attr_dict.get("pos_gt_full", None)

    total_pred = net_attr_dict["preds"].shape[0]
    pred_ts = (1.0 / sample_freq) * np.arange(total_pred)
    pred_sigmas = np.exp(net_attr_dict["preds_cov"])
    
    plot_dict = {
        "ts": ts,
        "pos_pred": pos_pred,
        "pos_gt": pos_gt,
        "pos_gt_full": pos_gt_full,
        "pred_ts": pred_ts,
        "preds": net_attr_dict["preds"],
        "targets": net_attr_dict["targets"],
        "pred_sigmas": pred_sigmas,
    }

    return plot_dict

def plot_imus(feat, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    x = len(feat[:, 0])
    plt.subplot(2, 1, 1)
    for i in range(3):
        plt.plot(x, feat[:, i])
    plt.ylabel('gyr')
    plt.legend()
    plt.grid(True)
    plt.xlabel('t(s)')

    plt.subplot(2, 1, 2)
    for i in range(3):
        if feat.shape[1] >= 6:
            plt.plot(x, feat[:, 3+i])
    plt.ylabel('acc')
    plt.legend()
    plt.grid(True)
    plt.xlabel('t(s)')
    return fig

def make_plots(plot_dict, outdir):
    pos_pred = plot_dict["pos_pred"]
    pos_gt = plot_dict["pos_gt"]
    pos_gt_full = plot_dict.get("pos_gt_full", None)
    
    pred_ts = plot_dict["pred_ts"]
    preds = plot_dict["preds"]
    targets = plot_dict["targets"]
    pred_sigmas = plot_dict["pred_sigmas"]

    dpi = 90
    figsize = (16, 9)

    fig1 = plt.figure(num="ins_traj", dpi=dpi, figsize=figsize)
    targ_names = ["dx", "dy", "dz"]
    rows = preds.shape[1]

    # --- 绘制左侧轨迹图 ---
    plt.subplot2grid((rows, 2), (0, 0), rowspan=rows)
    
    # [保留] 先画全量真值 (灰色背景)
    if pos_gt_full is not None:
        plt.plot(pos_gt_full[:, 0], pos_gt_full[:, 1], color='lightgray', linewidth=3, label="Full Ground Truth", zorder=1)

    plt.plot(pos_gt[:, 0], pos_gt[:, 1], 'r--', linewidth=1.5, label="Aligned GT Segment", zorder=2)
    plt.plot(pos_pred[:, 0], pos_pred[:, 1], 'b-', linewidth=1.5, label="Network Prediction", zorder=3)

    if pos_gt_full is not None:
        plt.scatter(pos_gt_full[0, 0], pos_gt_full[0, 1], c='k', marker='o', s=50, label="Origin (0s)", zorder=4)
    plt.scatter(pos_pred[0, 0], pos_pred[0, 1], c='g', marker='o', s=100, label="Pred Start", zorder=5)
    plt.scatter(pos_pred[-1, 0], pos_pred[-1, 1], c='b', marker='x', s=100, label="Pred End", zorder=5)

    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.title("2D Trajectory: Full GT (Gray) vs Prediction (Blue)")

    # --- 绘制右侧各轴曲线 ---
    for i in range(rows):
        plt.subplot2grid((rows, 2), (i, 1))
        plt.plot(preds[:, i])
        plt.plot(targets[:, i])
        plt.legend(["network_pred", "Ground_truth"])
        title = targ_names[i] if i < len(targ_names) else f"dim_{i}"
        plt.title(f"{title}")

    plt.tight_layout()
    plt.grid(True)
    
    save_path = osp.join(outdir, "traj.png")
    fig1.savefig(save_path)
    logging.info(f"Trajectory plot saved to {save_path}")

    # --- 绘制不确定性图 ---
    fig2 = plt.figure(num="pred_sigma", dpi=dpi, figsize=figsize)
    preds_plus_sig = preds + 3 * pred_sigmas
    preds_minus_sig = preds - 3 * pred_sigmas
    ylbs = ["x(m)", "y(m)", "z(m)"]

    for i in range(rows):
        plt.subplot(rows, 1, i + 1)
        plt.plot(pred_ts, preds_plus_sig[:, i], "-g", linewidth=0.2)
        plt.plot(pred_ts, preds_minus_sig[:, i], "-g", linewidth=0.2)
        plt.plot(pred_ts, preds[:, i], "-b", linewidth=0.5, label='pred')
        plt.plot(pred_ts, targets[:, i], "-r", linewidth=0.5, label='gt')

        label = ylbs[i] if i < len(ylbs) else f"dim_{i}"
        plt.ylabel(label)
        plt.legend()
        plt.grid(True)

    plt.xlabel("t(s)")
    fig2.savefig(osp.join(outdir, "pred_sigma.svg"))

    # [还原] 这里加回了 plt.show()
    # 如果你在服务器跑，可能会报错；如果在本地跑，会弹窗。
    try:
        plt.show()
    except Exception as e:
        print(f"Cannot show plot: {e}")

    plt.close("all")

    return