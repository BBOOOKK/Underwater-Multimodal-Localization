import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ================= 配置参数 =================
CSV_FILES = [
    '/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/processed_data/suiyi1.csv',
    '/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/processed_data/suiyi2.csv',
    '/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/processed_data/zhengfangxing1.csv',
    '/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/processed_data/zhengfangxing2.csv',
    '/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/processed_data/zhengfangxing3.csv',
    '/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/processed_data/zhixian1.csv',
    '/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/processed_data/zhixian2.csv'
]

WINDOW_SIZE = 50
BATCH_SIZE = 64
EPOCHS = 200

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

OUTPUT_DIR = "outputs"

# 保存/复用模型
SAVE_MODELS = True
REUSE_MODELS = True
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

# 早停/检查点（对所有模型统一配置，保证公平）
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-4

USE_CHECKPOINT = True

# 保存训练曲线
SAVE_TRAIN_CURVES = True
CURVE_DIR = os.path.join(OUTPUT_DIR, "curves")
# ===========================================


def _check_split_ratios(train_r, val_r, test_r):
    s = train_r + val_r + test_r
    if abs(s - 1.0) > 1e-8:
        raise ValueError(f"TRAIN/VAL/TEST 比例之和必须为 1.0，但现在是 {s}")
    if min(train_r, val_r, test_r) <= 0:
        raise ValueError("TRAIN/VAL/TEST 比例必须都 > 0")


def _dataset_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def _safe_model_name(name: str) -> str:
    """让文件名稳定、可跨平台。"""
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def save_scaler_y_npz(path_npz: str, scaler_y: StandardScaler):
    os.makedirs(os.path.dirname(path_npz), exist_ok=True)
    np.savez(path_npz, mean_=scaler_y.mean_, scale_=scaler_y.scale_, var_=scaler_y.var_)


def load_scaler_y_npz(path_npz: str) -> StandardScaler:
    data = np.load(path_npz)
    sc = StandardScaler()
    sc.mean_ = data["mean_"]
    sc.scale_ = data["scale_"]
    sc.var_ = data["var_"]
    # sklearn 的 StandardScaler 还需要 n_features_in_
    sc.n_features_in_ = sc.mean_.shape[0]
    return sc


def load_and_process_single_file(filepath: str, window_size: int):
    """
    单文件处理：
    - 生成 delta 标签
    - 单文件内 fit scaler_x/scaler_y（避免跨数据集泄漏）
    - 生成滑窗序列
    返回：
      inputs(list of 4 np.array), targets(np.array), gt_positions(np.array), scaler_y
      其中 inputs 的每个分支 shape: (N, window, 3)
      targets shape: (N, 2) (scaled)
      gt_positions shape: (N, 2) (raw gt_px/gt_py 对应每个样本时刻 j)
    """
    print(f"正在加载数据: {filepath} ...")
    df = pd.read_csv(filepath)

    df['delta_x'] = df['gt_px'].diff().fillna(0)
    df['delta_y'] = df['gt_py'].diff().fillna(0)

    cols_dvl = ['dvl_vx', 'dvl_vy', 'dvl_vz']
    cols_att = ['roll_raw', 'pitch_raw', 'yaw_raw']
    cols_acc = ['accX', 'axxY', 'accZ']
    cols_gyro = ['gryX', 'gryY', 'gryZ']

    all_features = cols_dvl + cols_att + cols_acc + cols_gyro
    label_cols = ['delta_x', 'delta_y']

    for c in all_features + label_cols + ['gt_px', 'gt_py']:
        if c not in df.columns:
            raise ValueError(f"文件 {filepath} 缺少列: {c}")

    X_raw = df[all_features].values
    Y_raw = df[label_cols].values
    GT_raw = df[['gt_px', 'gt_py']].values

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_raw)
    Y_scaled = scaler_y.fit_transform(Y_raw)

    X_seq, Y_seq, GT_seq = [], [], []
    for j in range(window_size, len(df)):
        X_seq.append(X_scaled[j - window_size:j])
        Y_seq.append(Y_scaled[j])
        GT_seq.append(GT_raw[j])

    if len(X_seq) == 0:
        raise ValueError(f"文件 {filepath} 长度不足以构造窗口 window_size={window_size}")

    X_seq = np.asarray(X_seq, dtype=np.float32)
    Y_seq = np.asarray(Y_seq, dtype=np.float32)
    GT_seq = np.asarray(GT_seq, dtype=np.float32)

    X_dvl = X_seq[:, :, 0:3]
    X_att = X_seq[:, :, 3:6]
    X_acc = X_seq[:, :, 6:9]
    X_gyro = X_seq[:, :, 9:12]

    return [X_dvl, X_att, X_acc, X_gyro], Y_seq, GT_seq, scaler_y


def split_by_time(n_samples: int, train_ratio: float, val_ratio: float, test_ratio: float):
    _check_split_ratios(train_ratio, val_ratio, test_ratio)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"样本数太少无法三段式切分: n={n_samples}, train/val/test={n_train}/{n_val}/{n_test}")

    train_sl = slice(0, n_train)
    val_sl = slice(n_train, n_train + n_val)
    test_sl = slice(n_train + n_val, n_samples)
    return train_sl, val_sl, test_sl


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


def plot_full_trajectory(gt_positions, start_pos, preds_delta_all_dict, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    config = {
        "font.family": 'serif',
        "font.serif": ['Times New Roman', 'Times', 'serif'],
        "mathtext.fontset": 'stix',
        "font.size": 8,
        "axes.unicode_minus": False
    }
    plt.rcParams.update(config)

    traj_gt_full = np.vstack([start_pos, gt_positions])
    ax.plot(traj_gt_full[:, 0], traj_gt_full[:, 1], 'k-', lw=1.5, label='Ground Truth', zorder=10)

    styles = {
        "Multi-branch CNN": {'c': '#1f77b4', 'ls': '--'},
        "2-Layer LSTM":     {'c': '#ff7f0e', 'ls': '-.'},
        "2-Layer TCN":      {'c': '#d62728', 'ls': '--'},
        "IONet (Bi-LSTM)":  {'c': '#9467bd', 'ls': '--'}
    }

    for name, preds_delta_all in preds_delta_all_dict.items():
        traj_pred_full = np.vstack([start_pos, np.cumsum(preds_delta_all, axis=0) + start_pos])
        st = styles.get(name, {'c': 'r', 'ls': '--'})
        ax.plot(traj_pred_full[:, 0], traj_pred_full[:, 1], color=st['c'], linestyle=st['ls'], lw=1.0, alpha=0.9, label=name)

    ax.set_title(title, fontsize=8, fontweight='bold', pad=15)
    ax.set_xlabel('East (m)', fontsize=8)
    ax.set_ylabel('North (m)', fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax.legend(fontsize=5, loc='best', frameon=True, shadow=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_test_trajectory(gt_test, start_pos_test, preds_delta_test_dict, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    config = {
        "font.family": 'serif',
        "font.serif": ['Times New Roman', 'Times', 'serif'],
        "mathtext.fontset": 'stix',
        "font.size": 8,
        "axes.unicode_minus": False
    }
    plt.rcParams.update(config)

    ax.plot(gt_test[:, 0], gt_test[:, 1], 'k-', lw=1.5, label='Ground Truth (Test)', zorder=10)

    styles = {
        "Multi-branch CNN": {'c': '#1f77b4', 'ls': '--'},
        "2-Layer LSTM":     {'c': '#ff7f0e', 'ls': '-.'},
        "2-Layer TCN":      {'c': '#d62728', 'ls': '--'},
        "IONet (Bi-LSTM)":  {'c': '#9467bd', 'ls': '--'}
    }

    for name, preds_delta_test in preds_delta_test_dict.items():
        traj_test_rec = np.cumsum(preds_delta_test, axis=0) + start_pos_test
        st = styles.get(name, {'c': 'r', 'ls': '--'})
        ax.plot(traj_test_rec[:, 0], traj_test_rec[:, 1], color=st['c'], linestyle=st['ls'], lw=1.0, alpha=0.9,
                label=name)

    ax.set_title(title, fontsize=8, fontweight='bold', pad=15)
    ax.set_xlabel('East (m)', fontsize=8)
    ax.set_ylabel('North (m)', fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax.legend(fontsize=5, loc='best', frameon=True, shadow=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_setup():
    config = {
        "font.family": 'serif',
        "font.serif": ['Times New Roman', 'Times', 'serif'],
        "mathtext.fontset": 'stix',
        "font.size": 8,
        "axes.unicode_minus": False
    }
    plt.rcParams.update(config)


def save_history_csv(history, out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(history.history).to_csv(out_csv, index=False, encoding="utf-8-sig")


def plot_history_curves(history, out_png: str, title: str):
    """
    保存训练曲线图：loss/val_loss + mae/val_mae（若存在）。
    """
    _plot_setup()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    hist = history.history
    epochs = np.arange(1, len(next(iter(hist.values()))) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3), dpi=150)

    # Loss
    if "loss" in hist:
        axes[0].plot(epochs, hist["loss"], label="loss")
    if "val_loss" in hist:
        axes[0].plot(epochs, hist["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=7)

    # MAE（你的 compile 指定了 metrics=['mae']，一般会有）
    if "mae" in hist:
        axes[1].plot(epochs, hist["mae"], label="mae")
    if "val_mae" in hist:
        axes[1].plot(epochs, hist["val_mae"], label="val_mae")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=7)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def make_callbacks(dataset_model_dir: str, dataset_name: str, model_name: str):
    """
    统一生成 callbacks；checkpoint 文件名固定，方便复用。
    """
    cbs = []
    safe_name = _safe_model_name(model_name)

    if USE_CHECKPOINT:
        os.makedirs(dataset_model_dir, exist_ok=True)
        best_path = os.path.join(dataset_model_dir, f"{safe_name}.best.keras")
        cbs.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=best_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                mode="min",
                verbose=1
            )
        )

    if USE_EARLY_STOPPING:
        cbs.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                min_delta=EARLY_STOPPING_MIN_DELTA,
                restore_best_weights=True,
                mode="min",
                verbose=1
            )
        )

    return cbs


# ==================== 主运行逻辑 ====================
if __name__ == '__main__':
    _check_split_ratios(TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_MODELS or REUSE_MODELS:
        os.makedirs(MODEL_DIR, exist_ok=True)
    if SAVE_TRAIN_CURVES:
        os.makedirs(CURVE_DIR, exist_ok=True)

    models_to_train = {
        "Multi-branch CNN": build_multi_cnn,
        "2-Layer LSTM":     build_pure_lstm,
        "2-Layer TCN":      build_tcn,
        "IONet (Bi-LSTM)":  build_ionet
    }

    all_metrics = []

    print(f"\n{'='*20} 开始对比实验（按数据集逐个训练/出图）{'='*20}")
    print(f"三段式切分：train/val/test = {TRAIN_RATIO:.2f}/{VAL_RATIO:.2f}/{TEST_RATIO:.2f}")
    print(f"输出目录：{os.path.abspath(OUTPUT_DIR)}")

    for filepath in CSV_FILES:
        dataset_name = _dataset_name_from_path(filepath)
        print(f"\n--- 数据集: {dataset_name} ---")

        inputs, targets, gt_positions, scaler_y = load_and_process_single_file(filepath, WINDOW_SIZE)

        dataset_model_dir = os.path.join(MODEL_DIR, dataset_name)
        scaler_path = os.path.join(dataset_model_dir, "scaler_y.npz")
        if SAVE_MODELS:
            save_scaler_y_npz(scaler_path, scaler_y)
        if REUSE_MODELS and os.path.exists(scaler_path):
            scaler_y = load_scaler_y_npz(scaler_path)

        n = len(targets)
        train_sl, val_sl, test_sl = split_by_time(n, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

        X_train = [x[train_sl] for x in inputs]
        Y_train = targets[train_sl]
        X_val = [x[val_sl] for x in inputs]
        Y_val = targets[val_sl]
        X_test = [x[test_sl] for x in inputs]
        Y_test = targets[test_sl]
        GT_test = gt_positions[test_sl]

        start_pos = gt_positions[0]

        preds_delta_all_dict = {}
        preds_delta_test_dict = {}

        for model_name, build_func in models_to_train.items():
            safe_name = _safe_model_name(model_name)

            # 复用时优先加载 best checkpoint（val_loss 最优）
            best_path = os.path.join(dataset_model_dir, f"{safe_name}.best.keras")
            model_path_keras = os.path.join(dataset_model_dir, f"{safe_name}.keras")
            model_path_h5 = os.path.join(dataset_model_dir, f"{safe_name}.h5")

            if REUSE_MODELS and os.path.exists(best_path):
                print(f"复用已保存最优模型: {model_name} (dataset={dataset_name}) -> {best_path}")
                model = tf.keras.models.load_model(best_path)
                history = None
            elif REUSE_MODELS and os.path.exists(model_path_keras):
                print(f"复用已保存模型: {model_name} (dataset={dataset_name}) -> {model_path_keras}")
                model = tf.keras.models.load_model(model_path_keras)
                history = None
            else:
                print(f"正在训练模型: {model_name} (dataset={dataset_name}) ...")
                model = build_func(WINDOW_SIZE)
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])

                callbacks = make_callbacks(dataset_model_dir, dataset_name, model_name)

                history = model.fit(
                    X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    callbacks=callbacks
                )

                # 保存“最后一次训练结束时的模型”（注意：若 early stopping restore_best_weights=True，
                # 这里保存的是最佳权重对应的模型）
                if SAVE_MODELS:
                    os.makedirs(dataset_model_dir, exist_ok=True)
                    model.save(model_path_keras)
                    try:
                        model.save(model_path_h5)
                    except Exception as e:
                        print(f"保存 h5 失败（可忽略）：{e}")

                # 保存训练曲线（每模型每数据集一份）
                if SAVE_TRAIN_CURVES and history is not None:
                    hist_csv = os.path.join(CURVE_DIR, dataset_name, f"{safe_name}.history.csv")
                    hist_png = os.path.join(CURVE_DIR, dataset_name, f"{safe_name}.history.png")
                    save_history_csv(history, hist_csv)
                    plot_history_curves(history, hist_png, title=f"{dataset_name} | {model_name}")

            # 统一预测
            preds_all_scaled = model.predict(inputs, verbose=0)
            preds_delta_all = scaler_y.inverse_transform(preds_all_scaled)

            preds_test_scaled = model.predict(X_test, verbose=0)
            preds_delta_test = scaler_y.inverse_transform(preds_test_scaled)

            preds_delta_all_dict[model_name] = preds_delta_all
            preds_delta_test_dict[model_name] = preds_delta_test

            start_pos_test = GT_test[0]
            traj_test_rec = np.cumsum(preds_delta_test, axis=0) + start_pos_test

            final_err = float(np.linalg.norm(GT_test[-1] - traj_test_rec[-1]))
            rmse = float(np.sqrt(np.mean(np.sum((GT_test - traj_test_rec) ** 2, axis=1))))

            all_metrics.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Final Error (m)": final_err,
                "Test RMSE (m)": rmse,
                "N_total": n,
                "N_train": len(Y_train),
                "N_val": len(Y_val),
                "N_test": len(Y_test),
                "ModelPathBest": best_path if os.path.exists(best_path) else "",
                "ModelPathLast": model_path_keras if os.path.exists(model_path_keras) else ""
            })

            print(f"[{dataset_name} | {model_name}] FinalErr={final_err:.3f}m, TestRMSE={rmse:.3f}m")

        # 出图：全轨迹 + 测试段
        out_full = os.path.join(OUTPUT_DIR, f"{dataset_name}_full.png")
        out_test = os.path.join(OUTPUT_DIR, f"{dataset_name}_test.png")

        plot_full_trajectory(
            gt_positions=gt_positions,
            start_pos=start_pos,
            preds_delta_all_dict=preds_delta_all_dict,
            out_path=out_full,
            title=f"Trajectory Comparison (FULL) - {dataset_name}"
        )

        plot_test_trajectory(
            gt_test=GT_test,
            start_pos_test=GT_test[0],
            preds_delta_test_dict=preds_delta_test_dict,
            out_path=out_test,
            title=f"Trajectory Comparison (TEST) - {dataset_name}"
        )

        print(f"已保存：{out_full}")
        print(f"已保存：{out_test}")

    # 汇总指标输出
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\n指标已保存：{metrics_path}")
