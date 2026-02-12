import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time
import json

from datetime import datetime

# === 3D rotation helpers (Body -> World) ===

def _euler_rpy_to_Rwb_enu(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Build rotation matrix R_wb (ENU) from roll/pitch/yaw.

    Convention (right-handed ENU): R_wb = Rz(yaw) * Ry(pitch) * Rx(roll)
    Mapping: v_world = R_wb @ v_body
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.stack([
        np.stack([cy, -sy, np.zeros_like(cy)], axis=-1),
        np.stack([sy,  cy, np.zeros_like(cy)], axis=-1),
        np.stack([np.zeros_like(cy), np.zeros_like(cy), np.ones_like(cy)], axis=-1),
    ], axis=-2)

    Ry = np.stack([
        np.stack([ cp, np.zeros_like(cp), sp], axis=-1),
        np.stack([np.zeros_like(cp), np.ones_like(cp), np.zeros_like(cp)], axis=-1),
        np.stack([-sp, np.zeros_like(cp), cp], axis=-1),
    ], axis=-2)

    Rx = np.stack([
        np.stack([np.ones_like(cr), np.zeros_like(cr), np.zeros_like(cr)], axis=-1),
        np.stack([np.zeros_like(cr), cr, -sr], axis=-1),
        np.stack([np.zeros_like(cr), sr,  cr], axis=-1),
    ], axis=-2)

    return Rz @ Ry @ Rx


def _euler_rpy_to_Rwb_ned(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:

    phi = roll
    theta = pitch
    psi = yaw

    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cps,  sps  = np.cos(psi), np.sin(psi)

    r11 = cth * cps
    r12 = cth * sps
    r13 = -sth

    r21 = sphi * sth * cps - cphi * sps
    r22 = sphi * sth * sps + cphi * cps
    r23 = sphi * cth

    r31 = cphi * sth * cps + sphi * sps
    r32 = cphi * sth * sps - sphi * cps
    r33 = cphi * cth

    R = np.stack([
        np.stack([r11, r12, r13], axis=-1),
        np.stack([r21, r22, r23], axis=-1),
        np.stack([r31, r32, r33], axis=-1),
    ], axis=-2)

    return R


def _rotate_body_to_world(df: pd.DataFrame,
                          cols_body_xyz: tuple[str, str, str],
                          rpy_cols: tuple[str, str, str] = ("roll_body", "pitch_body", "yaw_body"),
                          out_prefix: str = "w_") -> pd.DataFrame:
    """Rotate a 3D vector from Body frame to World frame using roll/pitch/yaw."""
    r = pd.to_numeric(df[rpy_cols[0]], errors="coerce").to_numpy()
    p = pd.to_numeric(df[rpy_cols[1]], errors="coerce").to_numpy()
    y = pd.to_numeric(df[rpy_cols[2]], errors="coerce").to_numpy()

    # Heuristic: if angles look like degrees, convert to radians
    max_abs = np.nanmax(np.abs(np.stack([r, p, y], axis=0)))
    if np.isfinite(max_abs) and max_abs > 3.5:
        r = np.deg2rad(r)
        p = np.deg2rad(p)
        y = np.deg2rad(y)

    # NOTE: SenseINS yaw convention appears opposite to the standard NED heading sign.
    # Empirically, using yaw -> -yaw yields strong alignment between rotated DVL and GT velocities.
    if WORLD_FRAME.upper() == "NED":
        y = -y
        Rwb = _euler_rpy_to_Rwb_ned(r, p, y)  # (N,3,3)
    else:
        Rwb = _euler_rpy_to_Rwb_enu(r, p, y)  # (N,3,3)

    vx = pd.to_numeric(df[cols_body_xyz[0]], errors="coerce").to_numpy()
    vy = pd.to_numeric(df[cols_body_xyz[1]], errors="coerce").to_numpy()
    vz = pd.to_numeric(df[cols_body_xyz[2]], errors="coerce").to_numpy()
    vb = np.stack([vx, vy, vz], axis=-1)  # (N,3)

    vw = np.einsum("nij,nj->ni", Rwb, vb)
    df[out_prefix + "x"] = vw[:, 0]
    df[out_prefix + "y"] = vw[:, 1]
    df[out_prefix + "z"] = vw[:, 2]
    return df


def pack_standard_scaler(scaler: StandardScaler) -> dict:
    """Pack a fitted sklearn StandardScaler into a pure-Python dict (torch-save friendly).

    IMPORTANT: store lists (not numpy arrays) so the checkpoint is maximally portable.
    """
    mean_ = np.asarray(scaler.mean_, dtype=np.float64).tolist()
    scale_ = np.asarray(scaler.scale_, dtype=np.float64).tolist()
    var_ = np.asarray(scaler.var_, dtype=np.float64).tolist()

    return {
        "mean_": mean_,
        "scale_": scale_,
        "var_": var_,
        "n_features_in_": int(getattr(scaler, "n_features_in_", len(mean_))),
        "n_samples_seen_": int(getattr(scaler, "n_samples_seen_", 0)),
    }


def unpack_standard_scaler(state: dict) -> StandardScaler:
    """Rebuild a fitted StandardScaler from a packed dict."""
    s = StandardScaler()
    s.mean_ = np.asarray(state["mean_"], dtype=np.float64)
    s.scale_ = np.asarray(state["scale_"], dtype=np.float64)
    s.var_ = np.asarray(state["var_"], dtype=np.float64)
    s.n_features_in_ = int(state["n_features_in_"])
    # sklearn uses either int or ndarray; int is sufficient for transform
    s.n_samples_seen_ = int(state.get("n_samples_seen_", 0))
    return s

mpl.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
})

STYLE_MAP = {
    "Ground Truth": dict(color="k", linestyle="-",  linewidth=2.0, marker="s", markersize=3.5, markevery=25),
    
    "Multi-CNN":    dict(color="b",       linestyle="-", linewidth=1.6, marker="^", markersize=4.0, markevery=25),
    "LSTM":         dict(color="m",       linestyle="-", linewidth=1.6, marker="v", markersize=4.0, markevery=25),
    "TCN":          dict(color="g",       linestyle="-", linewidth=1.6, marker="<", markersize=4.0, markevery=25),
    "IONet":        dict(color="#f59f00", linestyle="-", linewidth=1.6, marker="h", markersize=4.0, markevery=25),
    "Single-branch-CNN-LSTM": dict(color="#7b2cbf", linestyle="-", linewidth=1.6, marker="*", markersize=5.0, markevery=25),
}
DEFAULT_STYLE = dict(color="0.35", linestyle="-", linewidth=1.2, marker=None)

LEGEND_ORDER = ["Ground Truth", "Multi-CNN", "LSTM", "TCN", "IONet", "Single-branch-CNN-LSTM"]

# Device configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Fix random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def _ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


# ==================== Data pipeline helpers for cleaning, dedup, clipping ====================

def _clip_series(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.clip(lower=lo, upper=hi)


def _clean_and_aggregate_by_timestamp(df: pd.DataFrame, sensor_cols: list[str], gt_cols: list[str]) -> pd.DataFrame:
    """Ensure timestamp is strictly increasing and remove duplicates.

    If duplicates exist, aggregate by timestamp using mean over sensor channels and GT positions.
    This prevents dt=0 rows from poisoning v = dpos/dt.
    """
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Sort by time in case alignment introduced shuffling
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Aggregate duplicates by mean (safe for already-aligned data)
    agg_cols = list(dict.fromkeys(sensor_cols + gt_cols))
    df = df.groupby("timestamp", as_index=False)[agg_cols].mean(numeric_only=True)

    # After grouping, timestamps are unique; ensure monotonic
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def plot_with_style(ax, x, y, label, markers: bool = True):
    """Plot with predefined style from STYLE_MAP.

    - For long sequences, markers are automatically thinned to avoid over-plotting.
    - For very dense curves (e.g., displacement/error), set markers=False.
    """
    style = dict(STYLE_MAP.get(label, DEFAULT_STYLE))

    if not markers:
        # Remove marker-related kwargs to keep curves clean
        style.pop("marker", None)
        style.pop("markersize", None)
        style.pop("markevery", None)
    else:
        # Auto-thin markers: aim for ~40 markers across the curve
        if style.get("marker", None) is not None:
            n = len(x) if hasattr(x, "__len__") else 0
            markevery = max(1, int(n / 40)) if n > 0 else 1
            style["markevery"] = markevery
            # Slightly smaller markers to reduce clutter
            style["markersize"] = min(style.get("markersize", 4.0), 4.0)

    ax.plot(x, y, label=label, **style)

def apply_axes_style(ax):
    """Apply consistent axes style (grid, ticks, spines)."""
    ax.grid(True)
    ax.tick_params(which="both", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

def add_top_center_legend(fig, handles, labels, ncol=3):
    """Add unified top-center legend without frame."""
    fig.legend(handles, labels,
               loc="upper center",
               ncol=ncol,
               frameon=False,
               bbox_to_anchor=(0.5, 1.08),
               handlelength=2.2,
               columnspacing=1.4)

def tighten_layout_for_paper(fig):
    """Apply paper-quality subplot spacing."""
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.12, top=0.82, wspace=0.32, hspace=0.30)

def _collect_handles_labels(ax_list, methods_present):
    """Collect handles/labels in fixed LEGEND_ORDER."""
    label_to_handle = {}
    for ax in ax_list:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            label_to_handle[ll] = hh
    
    ordered_labels = [k for k in LEGEND_ORDER if (k in label_to_handle and k in methods_present)]
    ordered_handles = [label_to_handle[k] for k in ordered_labels]
    return ordered_handles, ordered_labels

def load_and_preprocess_file(path: str) -> pd.DataFrame:

    if not os.path.exists(path):
        print(f"File does not exist: {path}")
        return None

    df = pd.read_csv(path)
    if len(df) == 0:
        print(f"File is empty: {path}")
        return None

    # --- 0) Column mapping (tolerate different spellings) ---
    rename_map = {
        "accX": "acce_x",
        "accY": "acce_y",
        "axxY": "acce_y",
        "accZ": "acce_z",
        "gryX": "gyro_x",
        "gyroX": "gyro_x",
        "gryY": "gyro_y",
        "gyroY": "gyro_y",
        "gryZ": "gyro_z",
        "gyroZ": "gyro_z",
        # Pose / attitude aliases
        "roll": "roll_body",
        "pitch": "pitch_body",
        "yaw": "yaw_body",
        "Roll": "roll_body",
        "Pitch": "pitch_body",
        "Yaw": "yaw_body",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # --- 0.5) Timestamp de-duplication / aggregation (prevents dt=0) ---
    if "timestamp" not in df.columns:
        raise KeyError("Missing 'timestamp' column (required to compute dt).")

    # Define the minimum set of numeric columns we may aggregate safely
    _sensor_cols = [
        "acce_x", "acce_y", "acce_z",
        "gyro_x", "gyro_y", "gyro_z",
        "dvl_vx", "dvl_vy", "dvl_vz",
        "roll_body", "pitch_body", "yaw_body", "yaw_sin", "yaw_cos",
    ]
    _gt_cols = ["gt_px", "gt_py"]
    present_sensor = [c for c in _sensor_cols if c in df.columns]
    present_gt = [c for c in _gt_cols if c in df.columns]
    df = _clean_and_aggregate_by_timestamp(df, present_sensor, present_gt)

    # --- 1.5) Pose columns: cast and convert to radians (if present) ---
    if all(c in df.columns for c in ("roll_body", "pitch_body", "yaw_body")):
        df["roll_body"]  = pd.to_numeric(df["roll_body"],  errors="coerce")
        df["pitch_body"] = pd.to_numeric(df["pitch_body"], errors="coerce")
        df["yaw_body"]   = pd.to_numeric(df["yaw_body"],   errors="coerce")

        # Heuristic: if angles look like degrees, convert to radians
        max_abs_pose = np.nanmax(
            np.abs(df[["roll_body", "pitch_body", "yaw_body"]].to_numpy(dtype=np.float64))
        )
        if np.isfinite(max_abs_pose) and max_abs_pose > 3.5:
            df["roll_body"]  = np.deg2rad(df["roll_body"])
            df["pitch_body"] = np.deg2rad(df["pitch_body"])
            df["yaw_body"]   = np.deg2rad(df["yaw_body"])

        # Angle wrap handling: encode yaw using sin/cos to avoid 2π discontinuity
        df["yaw_sin"] = np.sin(df["yaw_body"].to_numpy(dtype=np.float64))
        df["yaw_cos"] = np.cos(df["yaw_body"].to_numpy(dtype=np.float64))

    # --- 2) Cast required numeric columns ---
    required_features = [
        "acce_x", "acce_y", "acce_z",
        "gyro_x", "gyro_y", "gyro_z",
        "dvl_vx", "dvl_vy", "dvl_vz",
    ]
    if USE_POSE_FEATURES:
        for c in ("roll_body", "pitch_body"):
            if c not in df.columns:
                raise KeyError(f"Missing required pose column for features: {c}")
        for c in ("yaw_sin", "yaw_cos"):
            if c not in df.columns:
                raise KeyError(f"Missing required yaw encoding column for features: {c}")
        required_features += ["roll_body", "pitch_body", "yaw_sin", "yaw_cos"]

    for c in required_features:
        if c not in df.columns:
            raise KeyError(f"Missing required feature column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- 2.5) Robust outlier clipping (physics-informed bounds) ---
    # These bounds are conservative; adjust if your AUV operates faster.
    # DVL in m/s
    for c in ("dvl_vx", "dvl_vy", "dvl_vz"):
        if c in df.columns:
            df[c] = _clip_series(df[c], lo=-2.5, hi=2.5)

    # ACC in m/s^2 (includes gravity)
    for c in ("acce_x", "acce_y", "acce_z"):
        if c in df.columns:
            df[c] = _clip_series(df[c], lo=-30.0, hi=30.0)

    # GYRO in rad/s (if your gyro is deg/s, it will be scaled by StandardScaler anyway; clip broadly)
    for c in ("gyro_x", "gyro_y", "gyro_z"):
        if c in df.columns:
            df[c] = _clip_series(df[c], lo=-20.0, hi=20.0)

    # --- 3) Pose-as-feature mode: do NOT rotate/overwrite ACC/DVL ---
    # Keep raw ACC/DVL as measured (typically body-frame). Provide pose to the model as features.
    has_pose = all(c in df.columns for c in ("roll_body", "pitch_body", "yaw_body"))

    if USE_POSE_ROTATION and has_pose:
        df = _rotate_body_to_world(
            df,
            ("acce_x", "acce_y", "acce_z"),
            rpy_cols=("roll_body", "pitch_body", "yaw_body"),
            out_prefix="w_acc_",
        )
        df = _rotate_body_to_world(
            df,
            ("dvl_vx", "dvl_vy", "dvl_vz"),
            rpy_cols=("roll_body", "pitch_body", "yaw_body"),
            out_prefix="w_dvl_",
        )
        # overwrite model inputs with world-frame values (only if explicit rotation enabled)
        df["acce_x"], df["acce_y"], df["acce_z"] = df["w_acc_x"], df["w_acc_y"], df["w_acc_z"]
        df["dvl_vx"], df["dvl_vy"], df["dvl_vz"] = df["w_dvl_x"], df["w_dvl_y"], df["w_dvl_z"]

    # --- 5) Build dt + displacement increments from GT positions (frame-aware) ---
    if "timestamp" not in df.columns:
        raise KeyError("Missing 'timestamp' column (required to compute dt).")

    if ("gt_px" not in df.columns) or ("gt_py" not in df.columns):
        raise KeyError("Training requires 'gt_px' and 'gt_py' for GT (NED: gt_px=North, gt_py=East).")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["gt_px"] = pd.to_numeric(df["gt_px"], errors="coerce")  # North (N)
    df["gt_py"] = pd.to_numeric(df["gt_py"], errors="coerce")  # East  (E)

    # dt in seconds (infer timestamp unit: ms vs s)
    dt_raw = df["timestamp"].diff()  # raw units
    med = np.nanmedian(dt_raw.to_numpy(dtype=np.float64))
    # Heuristic: if median step < 1.0, timestamps are already in seconds.
    # If median step >= 1.0, treat as milliseconds and convert to seconds.
    if np.isfinite(med) and med < 1.0:
        df["dt"] = dt_raw
    else:
        df["dt"] = dt_raw / 1000.0

    if os.environ.get("DEBUG_DT", "0") == "1":
        print(f"[DT] median raw step={med:.6g} -> dt median={np.nanmedian(df['dt'].to_numpy(dtype=np.float64)):.6g}")

    # dt sanitation: keep variable dt, but guard against pathological values.
    # - Non-positive dt should not exist after timestamp de-duplication; still guard.
    # - Very large gaps are clipped (not dropped) so the model can learn variable-rate behavior.
    DT_MIN = 1e-6
    DT_MAX_CLIP = 0.20  # seconds; clip rare dropouts instead of removing samples
    df.loc[df["dt"] <= DT_MIN, "dt"] = np.nan
    df.loc[df["dt"] > DT_MAX_CLIP, "dt"] = DT_MAX_CLIP

    # Displacements (targets): NED order [dN, dE]
    df["dN"] = df["gt_px"].diff()
    df["dE"] = df["gt_py"].diff()

    # --- 6) Build DVL-based world-frame displacement increments (for residual learning) ---
    if all(c in df.columns for c in ("roll_body", "pitch_body", "yaw_body", "dvl_vx", "dvl_vy", "dvl_vz")):
        tmp = df[["roll_body", "pitch_body", "yaw_body", "dvl_vx", "dvl_vy", "dvl_vz"]].copy()
        tmp = _rotate_body_to_world(
            tmp,
            ("dvl_vx", "dvl_vy", "dvl_vz"),
            rpy_cols=("roll_body", "pitch_body", "yaw_body"),
            out_prefix="w_dvl_",
        )

        if WORLD_FRAME.upper() == "NED":
            vN = tmp["w_dvl_x"].to_numpy(np.float64)
            vE = tmp["w_dvl_y"].to_numpy(np.float64)
        else:  # ENU
            vE = tmp["w_dvl_x"].to_numpy(np.float64)
            vN = tmp["w_dvl_y"].to_numpy(np.float64)

        dt = df["dt"].to_numpy(np.float64)
        df["dvl_dN"] = vN * dt
        df["dvl_dE"] = vE * dt

        # Residual targets: GT - DVL baseline
        df["rN"] = df["dN"] - df["dvl_dN"]
        df["rE"] = df["dE"] - df["dvl_dE"]
    else:
        df["dvl_dN"] = np.nan
        df["dvl_dE"] = np.nan
        df["rN"] = np.nan
        df["rE"] = np.nan

    # NOTE: supervise displacement increments directly to avoid dt-division blow-ups
    required_all = required_features + ["dt", "dN", "dE", "dvl_dN", "dvl_dE", "rN", "rE"]
    df = df.dropna(subset=required_all).reset_index(drop=True)

    return df

class ROVDataset(Dataset):
    """Dataset class for ROV trajectory prediction."""
    def __init__(
        self,
        df,
        seq_len=200,
        scaler_x=None,
        scaler_y=None,
        fit_scaler: bool = False,
        fit_scaler_x: bool | None = None,
        fit_scaler_y: bool | None = None,
        file_boundaries=None,
        stride: int = 1,
        targets: list[str] = ["dN", "dE"],
    ):
        self.df = df.copy()
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride = int(stride)
        self.targets = list(targets)

        # Keep dataset features in sync with global FEATURE_NAMES
        features = list(FEATURE_NAMES)

        x_data = self.df[features].astype(np.float32).values
        y_data = self.df[targets].astype(np.float32).values

        # IMPORTANT: keep dt in physical units (seconds) even when other channels are standardized.
        # Standardizing dt makes the velocity->displacement integration inconsistent across train/test.
        dt_raw = self.df["dt"].astype(np.float32).values

        self.scaler_x = scaler_x if scaler_x is not None else StandardScaler()
        self.scaler_y = scaler_y if scaler_y is not None else StandardScaler()

        # Allow fitting X/Y scalers independently (needed for residual-learning runs)
        do_fit_x = fit_scaler if fit_scaler_x is None else bool(fit_scaler_x)
        do_fit_y = fit_scaler if fit_scaler_y is None else bool(fit_scaler_y)

        if do_fit_x:
            x_data = self.scaler_x.fit_transform(x_data)
        else:
            if scaler_x is None:
                raise ValueError("scaler_x must be provided when fit_scaler_x is False")
            x_data = self.scaler_x.transform(x_data)

        if do_fit_y:
            y_data = self.scaler_y.fit_transform(y_data)
        else:
            if scaler_y is None:
                raise ValueError("scaler_y must be provided when fit_scaler_y is False")
            y_data = self.scaler_y.transform(y_data)

        # Restore dt (seconds) after standardization
        x_data[:, DT_INDEX] = dt_raw

        # Defensive check: make sure feature dimensionality matches model INPUT_CHANNELS
        if x_data.shape[1] != INPUT_CHANNELS:
            raise ValueError(f"Feature dim mismatch: x has {x_data.shape[1]} channels but INPUT_CHANNELS={INPUT_CHANNELS}. Check FEATURE_NAMES / USE_POSE_FEATURES.")

        self.sequences = []
        self.labels = []

        if file_boundaries is None:
            # Single file case (test set)
            if len(x_data) < seq_len:
                raise ValueError(f"Test set length {len(x_data)} must be >= sequence length {seq_len}")
            # Fix: label index changed to i+seq_len-1
            for i in range(0, len(x_data) - seq_len + 1, self.stride):
                self.sequences.append(x_data[i : i + seq_len])
                self.labels.append(y_data[i + seq_len - 1])
        else:
            # Multiple files case (training set)
            start_idx = 0
            for end_idx in file_boundaries:
                file_len = end_idx - start_idx
                if file_len >= seq_len:
                    # Fix: range changed to end_idx - seq_len + 1
                    for i in range(start_idx, end_idx - seq_len + 1, self.stride):
                        self.sequences.append(x_data[i : i + seq_len])
                        self.labels.append(y_data[i + seq_len - 1])
                start_idx = end_idx

        if len(self.sequences) == 0:
            # Provide actionable diagnostics
            n_rows = len(self.df)
            msg = [
                "Generated 0 sequences.",
                f"Total rows after preprocessing: {n_rows}",
                f"seq_len={seq_len}, stride={self.stride}",
            ]
            if file_boundaries is not None:
                lens = []
                s = 0
                for e in file_boundaries:
                    lens.append(e - s)
                    s = e
                msg.append(f"Per-file lengths: {lens}")
            raise ValueError(" | ".join(msg))

        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ==================== Configuration (must come before FEATURE_NAMES) ====================

# Window stride: reduce overlap during training to avoid rapid overfitting on highly-correlated windows.
STRIDE_TRAIN = 10
STRIDE_EVAL = 1

# World frame convention for GT and for rotated sensor vectors.
# - "ENU": x=East, y=North, z=Up
# - "NED": x=North, y=East, z=Down
WORLD_FRAME = "NED"  # user: test=SenseINS_aligned_5.csv, use NED

# === Pose-as-feature mode (no explicit ACC/DVL rotation) ===
USE_POSE_ROTATION = False   # 不显式旋转/覆盖 ACC/DVL
USE_POSE_FEATURES = True    # 将 roll/pitch/yaw 作为输入特征

# ==================== Model IO definitions (keep consistent with dataset) ====================
INPUT_LEN = 200
OUTPUT_SIZE = 2

# Feature names are the single source of truth for input dimensionality.
BASE_FEATURE_NAMES = [
    "acce_x", "acce_y", "acce_z",
    "gyro_x", "gyro_y", "gyro_z",
    "dvl_vx", "dvl_vy", "dvl_vz",
    "dt",  # seconds
]

POSE_FEATURE_NAMES = [
    "roll_body", "pitch_body", "yaw_sin", "yaw_cos",
]

FEATURE_NAMES = list(BASE_FEATURE_NAMES) + (list(POSE_FEATURE_NAMES) if USE_POSE_FEATURES else [])
INPUT_CHANNELS = len(FEATURE_NAMES)

# Index of dt inside FEATURE_NAMES (used for debugging / integration-related logic)
DT_INDEX = FEATURE_NAMES.index("dt")

class MultiBranchCNN(nn.Module):
    """Improved Multi-CNN with Global Average Pooling (GAP) to reduce overfitting."""
    def __init__(self):
        super(MultiBranchCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(INPUT_CHANNELS, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, OUTPUT_SIZE)
        )

    def forward(self, x):
        # (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = x[:, :, -1]   # last timestep feature
        return self.fc(x)

class TwoLayerLSTM(nn.Module):
    """Refined two-layer LSTM with smaller hidden size for better generalization."""
    def __init__(self):
        super(TwoLayerLSTM, self).__init__()
        self.lstm = nn.LSTM(INPUT_CHANNELS, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, OUTPUT_SIZE)
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

class Chomp1d(nn.Module):
    """Remove extra timesteps introduced by padding in causal convolutions."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: (B, C, T)
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class DeepCausalTCN(nn.Module):
    """Deep causal Temporal Convolutional Network (TCN) with sufficient receptive field."""
    def __init__(self, input_channels=None, output_size=None, num_channels=None, kernel_size=3, dropout=0.2):
        super(DeepCausalTCN, self).__init__()
        if input_channels is None:
            input_channels = INPUT_CHANNELS
        if output_size is None:
            output_size = OUTPUT_SIZE
        if num_channels is None:
            # 7 levels: dilation 1..64 gives receptive field > 200 for k=3
            num_channels = [32] * 7

        layers = []
        for i in range(len(num_channels)):
            in_c = input_channels if i == 0 else num_channels[i - 1]
            out_c = num_channels[i]
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_c, out_c, kernel_size, padding=padding, dilation=dilation),
                    Chomp1d(padding),  # keep length unchanged (causal)
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

        self.tcn = nn.ModuleList(layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        for layer in self.tcn:
            x = layer(x)
        # Use the last timestep feature for prediction
        return self.fc(x[:, :, -1])

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_experiment(model_class, name, train_dataset, val_dataset, test_dataset, scaler_y, test_df_original, epochs=50, patience=10, save_dir="./checkpoints", use_residual=False):
    print(f"\nStarting training: {name}")
    os.makedirs(save_dir, exist_ok=True)
    
    use_pin_memory = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=use_pin_memory)

    model = model_class().to(DEVICE)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Robust regression loss: less sensitive to rare GT spikes that dominate MSE
    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(save_dir, f"{name}_best.pth")
    
    history = {"train_loss": [], "val_loss": [], "epochs": []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred.contiguous(), y.contiguous())
            
            # Zero-mean constraint for residual learning (suppress systematic bias)
            if use_residual:
                loss = loss + 0.01 * torch.mean(torch.abs(torch.mean(pred, dim=0)))
            
            loss.backward()
            # Gradient clipping improves stability for RNN/TCN on long sequences
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Evaluation (use this for early stopping)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                val_loss_i = criterion(pred.contiguous(), y.contiguous())
                if use_residual:
                    val_loss_i = val_loss_i + 0.01 * torch.mean(torch.abs(torch.mean(pred, dim=0)))
                val_loss += val_loss_i.item()
        val_loss /= len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["epochs"].append(epoch + 1)

        # Scheduler step based on validation metric
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{epochs} | LR: {current_lr:.2e} | TrainLoss: {avg_train_loss:.5f} | ValLoss: {val_loss:.5e}",
            end=""
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                # ===== weights / optimizer =====
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": int(epoch + 1),

                # ===== training metrics =====
                "val_loss": float(val_loss),
                "best_val_loss": float(best_val_loss if np.isfinite(best_val_loss) else val_loss),
                "train_loss": float(avg_train_loss),

                # ===== scalers (fit on TRAIN) =====
                # ONE .pth should be sufficient for later testing
                "scaler_x": pack_standard_scaler(train_dataset.scaler_x),
                "scaler_y": pack_standard_scaler(scaler_y),

                # ===== schema / configuration (方案 A) =====
                "model_name": str(name),
                "model_class": str(model_class.__name__),
                "world_frame": str(WORLD_FRAME),
                "use_pose_features": bool(USE_POSE_FEATURES),
                "use_pose_rotation": bool(USE_POSE_ROTATION),
                "feature_names": list(FEATURE_NAMES),
                "target_names": list(getattr(train_dataset, "targets", ["dN", "dE"])),
                "seq_len": int(INPUT_LEN),
                "input_channels": int(INPUT_CHANNELS),
                "output_size": int(OUTPUT_SIZE),
                "dt_index": int(DT_INDEX),

                # ===== training-windowing metadata =====
                "stride_train": int(getattr(train_dataset, "stride", 1)),

                # ===== residual-learning metadata =====
                "use_residual": bool(use_residual),
                "residual_baseline_cols": ["dvl_dN", "dvl_dE"],
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Final test evaluation (once) on the held-out test set
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            test_loss += criterion(pred, y).item()
    test_loss /= len(test_loader)
    print(f"Final TestLoss (held-out): {test_loss:.5e}")
    
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
    
    pred_inc = scaler_y.inverse_transform(np.concatenate(all_preds))
    target_inc = scaler_y.inverse_transform(np.concatenate(all_targets))

    # === Residual Learning: Reconstruct absolute increments from residuals ===
    if use_residual:
        # Get DVL baseline from test_df_original (stride=1, so indices align)
        dvl_dN_full = test_df_original["dvl_dN"].to_numpy(np.float64)
        dvl_dE_full = test_df_original["dvl_dE"].to_numpy(np.float64)
        
        # Align with predictions: first pred corresponds to row (INPUT_LEN - 1)
        start_idx_dvl = INPUT_LEN - 1
        dvl_dN_aligned = dvl_dN_full[start_idx_dvl : start_idx_dvl + len(pred_inc)]
        dvl_dE_aligned = dvl_dE_full[start_idx_dvl : start_idx_dvl + len(pred_inc)]
        
        # Reconstruct: absolute = baseline + residual
        if WORLD_FRAME.upper() == "NED":
            pred_inc[:, 0] += dvl_dN_aligned  # dN = dvl_dN + rN
            pred_inc[:, 1] += dvl_dE_aligned  # dE = dvl_dE + rE
            target_inc[:, 0] += dvl_dN_aligned  # GT also needs reconstruction for consistency
            target_inc[:, 1] += dvl_dE_aligned
        else:  # ENU
            pred_inc[:, 0] += dvl_dE_aligned
            pred_inc[:, 1] += dvl_dN_aligned
            target_inc[:, 0] += dvl_dE_aligned
            target_inc[:, 1] += dvl_dN_aligned

    # Displacement increment outputs:
    # - NED: [dN, dE]
    # - ENU: [dE, dN]
    if WORLD_FRAME.upper() == "NED":
        pred_delta_north = pred_inc[:, 0]
        pred_delta_east  = pred_inc[:, 1]
        gt_delta_north   = target_inc[:, 0]
        gt_delta_east    = target_inc[:, 1]
    else:
        pred_delta_east  = pred_inc[:, 0]
        pred_delta_north = pred_inc[:, 1]
        gt_delta_east    = target_inc[:, 0]
        gt_delta_north   = target_inc[:, 1]

    # Build GT trajectory slice directly from positions (plotting coords are always East, North)
    # Our labels come from the last timestep of each window: index i+seq_len-1.
    # With stride=1 on test, the first predicted increment corresponds to row (seq_len-1).
    start_idx = INPUT_LEN - 1

    # points = increments + 1, but GT slice may be shorter near file end;
    # align by trimming predictions to the available GT length.
    requested_points = len(pred_delta_east) + 1
    available_points = max(0, len(test_df_original) - start_idx)
    traj_len = min(requested_points, available_points)

    if traj_len < 2:
        raise ValueError(
            f"Not enough GT points to build trajectory: start_idx={start_idx}, "
            f"len(df)={len(test_df_original)}, traj_len={traj_len}"
        )

    # Trim increments to match traj_len-1
    pred_delta_east = pred_delta_east[: traj_len - 1]
    pred_delta_north = pred_delta_north[: traj_len - 1]
    gt_delta_east = gt_delta_east[: traj_len - 1]
    gt_delta_north = gt_delta_north[: traj_len - 1]

    if WORLD_FRAME.upper() == "NED":
        gt_x = test_df_original.iloc[start_idx : start_idx + traj_len]["gt_py"].to_numpy(dtype=np.float64)  # East
        gt_y = test_df_original.iloc[start_idx : start_idx + traj_len]["gt_px"].to_numpy(dtype=np.float64)  # North
    else:
        gt_x = test_df_original.iloc[start_idx : start_idx + traj_len]["gt_px"].to_numpy(dtype=np.float64)  # East
        gt_y = test_df_original.iloc[start_idx : start_idx + traj_len]["gt_py"].to_numpy(dtype=np.float64)  # North

    true_path_x = gt_x
    true_path_y = gt_y

    # Predicted trajectory: integrate displacement increments from the same start
    pred_path_x = np.concatenate([[true_path_x[0]], true_path_x[0] + np.cumsum(pred_delta_east)])
    pred_path_y = np.concatenate([[true_path_y[0]], true_path_y[0] + np.cumsum(pred_delta_north)])

    # Sanity check
    assert len(true_path_x) == len(pred_path_x), f"Length mismatch: GT={len(true_path_x)}, Pred={len(pred_path_x)}"
    assert len(gt_delta_east) == len(pred_delta_east), f"Increment length mismatch: GT={len(gt_delta_east)}, Pred={len(pred_delta_east)}"
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
        "Best ValLoss": best_val_loss,
        "Final TestLoss": float(test_loss),
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
    
    return (
        true_path_x, true_path_y,
        pred_path_x, pred_path_y,
        gt_delta_east, gt_delta_north,
        pred_delta_east, pred_delta_north,
        metrics, history
    )

def plot_case_4panel(case_name, gt_delta_east, gt_delta_north, methods, save_dir, units="m"):
    """Plot 4-panel comparison figure (journal quality)."""
    fig_dir = os.path.join(save_dir, "figures")
    _ensure_dir(fig_dir)

    steps = np.arange(len(gt_delta_east))

    any_m = next(iter(methods.values()))
    true_x = any_m["true_x"]
    true_y = any_m["true_y"]

    def pos_error(px, py, tx, ty):
        return np.sqrt((px - tx) ** 2 + (py - ty) ** 2)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), dpi=200)

    present = set(["Ground Truth"])
    present.update(methods.keys())

    # (a) North displacement increments
    ax = axes[0]
    plot_with_style(ax, steps, gt_delta_north, "Ground Truth", markers=False)
    for name, data in methods.items():
        plot_with_style(ax, steps, data["pred_delta_north"], name, markers=False)
    ax.set_xlabel("Steps")
    ax.set_ylabel(f"North Displacement Increments ({units})")
    ax.set_title("(a)")
    apply_axes_style(ax)

    # (b) East displacement increments
    ax = axes[1]
    plot_with_style(ax, steps, gt_delta_east, "Ground Truth", markers=False)
    for name, data in methods.items():
        plot_with_style(ax, steps, data["pred_delta_east"], name, markers=False)
    ax.set_xlabel("Steps")
    ax.set_ylabel(f"East Displacement Increments ({units})")
    ax.set_title("(b)")
    apply_axes_style(ax)

    # (c) Trajectory (East-North) + aspect equal (关键修改)
    ax = axes[2]
    plot_with_style(ax, true_x, true_y, "Ground Truth")
    for name, data in methods.items():
        plot_with_style(ax, data["pred_x"], data["pred_y"], name)
    ax.set_xlabel(f"East ({units})")
    ax.set_ylabel(f"North ({units})")
    ax.set_title("(c)")
    ax.set_aspect("equal", adjustable="box")  # 强制等比例
    apply_axes_style(ax)

    # (d) Position error
    ax = axes[3]
    for name, data in methods.items():
        err = pos_error(data["pred_x"], data["pred_y"], true_x, true_y)
        plot_with_style(ax, np.arange(len(err)), err, name, markers=False)
    ax.set_xlabel("Steps")
    ax.set_ylabel(f"Position Error ({units})")
    ax.set_title("(d)")
    apply_axes_style(ax)

    # 统一 legend
    handles, labels = _collect_handles_labels(list(axes), present)
    add_top_center_legend(fig, handles, labels, ncol=3)

    tighten_layout_for_paper(fig)

    out_png = os.path.join(fig_dir, f"case_{case_name}_4panel.png")
    out_pdf = os.path.join(fig_dir, f"case_{case_name}_4panel.pdf")
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_pdf}")


def plot_case_2panel(case_name, methods, save_dir, units="m"):
    """Plot 2-panel comparison figure (journal quality)."""
    fig_dir = os.path.join(save_dir, "figures")
    _ensure_dir(fig_dir)

    any_m = next(iter(methods.values()))
    true_x = any_m["true_x"]
    true_y = any_m["true_y"]

    def pos_error(px, py, tx, ty):
        return np.sqrt((px - tx) ** 2 + (py - ty) ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=200)

    present = set(["Ground Truth"])
    present.update(methods.keys())

    # (a) Trajectory + aspect equal
    ax = axes[0]
    plot_with_style(ax, true_x, true_y, "Ground Truth")
    for name, data in methods.items():
        plot_with_style(ax, data["pred_x"], data["pred_y"], name)
    ax.set_xlabel(f"East ({units})")
    ax.set_ylabel(f"North ({units})")
    ax.set_title("(a) AUV motion trajectory")
    ax.set_aspect("equal", adjustable="box")  # 关键修改
    apply_axes_style(ax)

    # (b) Position error
    ax = axes[1]
    for name, data in methods.items():
        err = pos_error(data["pred_x"], data["pred_y"], true_x, true_y)
        plot_with_style(ax, np.arange(len(err)), err, name, markers=False)
    ax.set_xlabel("Steps")
    ax.set_ylabel(f"Position Error ({units})")
    ax.set_title("(b) Position error")
    apply_axes_style(ax)

    handles, labels = _collect_handles_labels(list(axes), present)
    add_top_center_legend(fig, handles, labels, ncol=3)

    tighten_layout_for_paper(fig)

    out_png = os.path.join(fig_dir, f"case_{case_name}_2panel.png")
    out_pdf = os.path.join(fig_dir, f"case_{case_name}_2panel.pdf")
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_pdf}")

def load_model_and_scalers(ckpt_path: str, model_class: type):
    """Load a trained model AND its training scalers from a single .pth checkpoint.

    Returns: (model, scaler_x, scaler_y, ckpt_dict)
    """
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = model_class().to(DEVICE)

    # Safety: warn if schema mismatch (common source of silent failure)
    ckpt_feats = ckpt.get("feature_names", None)
    if ckpt_feats is not None and list(ckpt_feats) != list(FEATURE_NAMES):
        print("[WARN] Checkpoint feature_names != current FEATURE_NAMES. ")
        print(f"  ckpt:   {list(ckpt_feats)}")
        print(f"  code:   {list(FEATURE_NAMES)}")

    ckpt_tgts = ckpt.get("target_names", None)
    if ckpt_tgts is not None and len(ckpt_tgts) != OUTPUT_SIZE:
        print("[WARN] Checkpoint target_names length != OUTPUT_SIZE.")
        print(f"  ckpt target_names: {ckpt_tgts}, OUTPUT_SIZE={OUTPUT_SIZE}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler_x = unpack_standard_scaler(ckpt["scaler_x"])
    scaler_y = unpack_standard_scaler(ckpt["scaler_y"])
    return model, scaler_x, scaler_y, ckpt

if __name__ == "__main__":
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"/Users/book./Desktop/book/2021_2025/科研/experiment_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)

    # ==================== Dataset Paths (user-specified) ====================
    # Train on: 5,3,2,1  |  Validate on: 4
    # NOTE: The original pipeline expects a held-out test set for final reporting/plotting.
    # If you do not provide a separate test file, we will reuse the VAL file for the
    # final "test" evaluation + plotting (this is NOT a true held-out test).

    train_paths = [
        "/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/data/0730data/SenseINS_aligned_4.csv",
        "/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/data/0730data/SenseINS_aligned_2.csv",
        "/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/data/0730data/SenseINS_aligned_1.csv",
        "/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/data/0730data/SenseINS_aligned_3.csv"
    ]

    val_path =  "/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/data/0730data/SenseINS_aligned_5.csv"

    # Optional: set a true held-out test file here if you have one.
    test_path = "/Users/book./Desktop/book/2021_2025/科研/水下多模态定位技术/data/0730data/SenseINS_aligned_5.csv"

    # Pretty print split
    print("\n==================== Dataset Split ====================")
    print("Train CSVs:")
    for p in train_paths:
        print(f"  - {p}")
    print("Val CSVs:")
    print(f"  - {val_path}")
    if test_path is None:
        print("Test CSVs:")
        print(f"  - {val_path}  [WARN: using VAL as TEST]")
    else:
        print("Test CSVs:")
        print(f"  - {test_path}")
    print("=======================================================\n")

    # ====== Load training files (multi-file) and record boundaries ======
    train_dfs = []
    file_boundaries = []
    total_len = 0

    for p in train_paths:
        df_i = load_and_preprocess_file(p)
        if df_i is None:
            raise RuntimeError(f"Failed to load training file: {p}")
        train_dfs.append(df_i)
        total_len += len(df_i)
        file_boundaries.append(total_len)

    train_df = pd.concat(train_dfs, ignore_index=True)

    # ====== Load val file ======
    val_df = load_and_preprocess_file(val_path)
    if val_df is None:
        raise RuntimeError(f"Failed to load val file: {val_path}")

    # ====== Load test file ======
    _test_path_used = val_path if test_path is None else test_path
    if test_path is None:
        print("[WARN] No held-out test set provided; using VAL as TEST for final evaluation/plots.")

    test_df = load_and_preprocess_file(_test_path_used)
    if test_df is None:
        raise RuntimeError(f"Failed to load test file: {_test_path_used}")

    # ====== Build datasets (fit scalers on train, reuse on test/val) ======
    seq_len = INPUT_LEN

    train_dataset = ROVDataset(
        train_df,
        seq_len=seq_len,
        scaler_x=None,
        scaler_y=None,
        fit_scaler=True,
        file_boundaries=file_boundaries,
        stride=STRIDE_TRAIN,
    )

    val_dataset = ROVDataset(
        val_df,
        seq_len=seq_len,
        scaler_x=train_dataset.scaler_x,
        scaler_y=train_dataset.scaler_y,
        fit_scaler=False,
        file_boundaries=None,
        stride=STRIDE_EVAL,
    )

    test_dataset = ROVDataset(
        test_df,
        seq_len=seq_len,
        scaler_x=train_dataset.scaler_x,
        scaler_y=train_dataset.scaler_y,
        fit_scaler=False,
        file_boundaries=None,
        stride=STRIDE_EVAL,
    )

    scaler_y = train_dataset.scaler_y

    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    experiments = [
        (MultiBranchCNN, "Multi-CNN"),
        (TwoLayerLSTM, "LSTM"),
        (IONet, "IONet"),
        (DeepCausalTCN, "TCN"),
    ]

    methods = {}
    metrics_rows = []

    gt_de = None
    gt_dn = None

    for model_cls, label in experiments:
        # Enable residual learning for IONet and Multi-CNN
        use_residual = (label in ("Multi-CNN", "IONet"))
        
        # Use different targets for residual models
        if use_residual:
            train_dataset_res = ROVDataset(
                train_df,
                seq_len=seq_len,
                scaler_x=train_dataset.scaler_x,   # reuse X scaler (do NOT refit)
                scaler_y=None,                      # fit a fresh Y scaler for residuals
                fit_scaler=False,
                fit_scaler_x=False,
                fit_scaler_y=True,
                file_boundaries=file_boundaries,
                stride=STRIDE_TRAIN,
                targets=["rN", "rE"],
            )
            val_dataset_res = ROVDataset(
                val_df,
                seq_len=seq_len,
                scaler_x=train_dataset.scaler_x,
                scaler_y=train_dataset_res.scaler_y,
                fit_scaler=False,
                fit_scaler_x=False,
                fit_scaler_y=False,
                file_boundaries=None,
                stride=STRIDE_EVAL,
                targets=["rN", "rE"],
            )
            test_dataset_res = ROVDataset(
                test_df,
                seq_len=seq_len,
                scaler_x=train_dataset.scaler_x,
                scaler_y=train_dataset_res.scaler_y,
                fit_scaler=False,
                fit_scaler_x=False,
                fit_scaler_y=False,
                file_boundaries=None,
                stride=STRIDE_EVAL,
                targets=["rN", "rE"],
            )
            scaler_y_used = train_dataset_res.scaler_y
            train_ds, val_ds, test_ds = train_dataset_res, val_dataset_res, test_dataset_res
        else:
            # LSTM/TCN use absolute increments
            train_ds, val_ds, test_ds = train_dataset, val_dataset, test_dataset
            scaler_y_used = scaler_y

        out = run_experiment(
            model_class=model_cls,
            name=label,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            scaler_y=scaler_y_used,
            test_df_original=test_df,
            epochs=100,
            patience=15,
            save_dir=ckpt_dir,
            use_residual=use_residual,  # Pass residual flag
        )

        (
            true_x, true_y,
            pred_x, pred_y,
            gt_delta_east, gt_delta_north,
            pred_delta_east, pred_delta_north,
            metrics, history,
        ) = out

        # Use GT deltas from the first model run (identical across models)
        if gt_de is None:
            gt_de, gt_dn = gt_delta_east, gt_delta_north

        methods[label] = {
            "true_x": true_x,
            "true_y": true_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "pred_delta_east": pred_delta_east,
            "pred_delta_north": pred_delta_north,
        }

        row = {"Model": label}
        row.update(metrics)
        metrics_rows.append(row)

    # Save combined metrics summary
    pd.DataFrame(metrics_rows).to_csv(os.path.join(exp_dir, "results_summary.csv"), index=False)

    # Plot paper-style figures
    plot_case_4panel(
        case_name="Test",
        gt_delta_east=gt_de,
        gt_delta_north=gt_dn,
        methods=methods,
        save_dir=exp_dir,
        units="m",
    )

    plot_case_2panel(
        case_name="Test",
        methods=methods,
        save_dir=exp_dir,
        units="m",
    )

    print(f"\nDone. Outputs saved to: {exp_dir}")
