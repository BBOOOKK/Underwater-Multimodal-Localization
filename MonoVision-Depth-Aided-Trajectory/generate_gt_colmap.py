#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

from rosbags.highlevel import AnyReader

# ====== SOLAQUA topics ======
TOPIC_IMAGE = "/image/compressed_image/data"         # sensor_msgs/CompressedImage
TOPIC_CINFO = "/image/compressed_image/camera_info"  # sensor_msgs/CameraInfo
TOPIC_DEPTH = "/sensor/depth_temperature"            # messages/msg/DepthTemperature

# ====== Tunables ======
FPS_LIMIT = 15
LAP_VAR_THRESH = 20.0
BRIGHT_MIN, BRIGHT_MAX = 25, 230

COLMAP_BIN = "colmap"  # in PATH

def colmap_supports_option(group: str, option: str) -> bool:
    """Check if `colmap <group> -h` contains the option string."""
    try:
        out = subprocess.check_output([COLMAP_BIN, group, "-h"], text=True, stderr=subprocess.STDOUT)
        return option in out
    except Exception:
        return False

def valid_camera_params(cam: dict | None) -> tuple[bool, str]:
    """Return (ok, params_str) for OPENCV model."""
    if cam is None:
        return False, ""
    fx, fy, cx, cy = cam.get("fx", 0.0), cam.get("fy", 0.0), cam.get("cx", 0.0), cam.get("cy", 0.0)
    if fx <= 1e-6 or fy <= 1e-6:
        return False, ""
    D = cam.get("D", [])
    k1 = D[0] if len(D) > 0 else 0.0
    k2 = D[1] if len(D) > 1 else 0.0
    p1 = D[2] if len(D) > 2 else 0.0
    p2 = D[3] if len(D) > 3 else 0.0
    return True, f"{fx},{fy},{cx},{cy},{k1},{k2},{p1},{p2}"

def run(cmd, cwd=None):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)

def _bag_msgs(bag_path: Path, topics: set[str]):
    """
    Yield (t_sec, topic, msg) where msg is DESERIALIZED.
    """
    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, t_ns, raw in reader.messages(connections=conns):
            # IMPORTANT: raw is bytes; need to deserialize
            msg = reader.deserialize(raw, conn.msgtype)
            yield t_ns * 1e-9, conn.topic, msg

def read_camera_info(video_bag: Path):
    for t, topic, msg in _bag_msgs(video_bag, {TOPIC_CINFO}):
        # ROS2 style: K, D ; ROS1 style: k, d
        K_flat = getattr(msg, "K", None)
        if K_flat is None:
            K_flat = getattr(msg, "k", None)
        if K_flat is None:
            raise AttributeError("CameraInfo has neither 'K' nor 'k' field.")

        D_flat = getattr(msg, "D", None)
        if D_flat is None:
            D_flat = getattr(msg, "d", None)
        if D_flat is None:
            D_flat = []

        K = np.array(K_flat, dtype=float).reshape(3, 3)
        D = np.array(D_flat, dtype=float)

        # width/height sometimes also vary in naming (but usually width/height)
        width = int(getattr(msg, "width"))
        height = int(getattr(msg, "height"))
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        return {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "D": D.tolist(),
            "width": width, "height": height
        }
    return None

def extract_images(video_bag: Path, out_dir: Path, fps_limit=8):
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    min_dt = 1.0 / float(fps_limit)
    last_t = None

    ts_rows = []
    rej_rows = []

    for t, topic, msg in tqdm(_bag_msgs(video_bag, {TOPIC_IMAGE}), desc="Extract images"):
        if last_t is not None and (t - last_t) < min_dt:
            continue

        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            rej_rows.append((t, "decode_fail"))
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_b = float(gray.mean())
        if mean_b < BRIGHT_MIN or mean_b > BRIGHT_MAX:
            rej_rows.append((t, f"brightness_{mean_b:.1f}"))
            continue

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < LAP_VAR_THRESH:
            rej_rows.append((t, f"blur_{lap_var:.1f}"))
            continue

        name = f"{t:.6f}.jpg"
        cv2.imwrite(str(img_dir / name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        ts_rows.append((t, name))
        last_t = t

    ts_df = pd.DataFrame(ts_rows, columns=["t", "image"])
    ts_df.to_csv(out_dir / "timestamps.csv", index=False)

    rej_df = pd.DataFrame(rej_rows, columns=["t", "reason"])
    rej_df.to_csv(out_dir / "reject_log.csv", index=False)

    if len(ts_df) < 80:
        raise RuntimeError(f"Too few images after filtering: {len(ts_df)}. "
                           f"Try lowering LAP_VAR_THRESH or increasing FPS_LIMIT.")
    return ts_df

def extract_depth(data_bag: Path, out_dir: Path):
    rows = []
    for t, topic, msg in tqdm(_bag_msgs(data_bag, {TOPIC_DEPTH}), desc="Extract depth"):
        depth = None
        for key in ("depth", "depth_m", "depthMeters", "depth_meter"):
            if hasattr(msg, key):
                depth = float(getattr(msg, key))
                break
        if depth is None:
            continue
        rows.append((t, depth))

    df = pd.DataFrame(rows, columns=["t", "depth_m"]).sort_values("t")
    df.to_csv(out_dir / "depth.csv", index=False)
    if len(df) < 50:
        raise RuntimeError(
            f"Too few depth samples ({len(df)}). "
            f"DepthTemperature field name may not match; adjust extract_depth()."
        )
    return df

def setup_colmap_dirs(work_dir: Path):
    (work_dir / "colmap").mkdir(parents=True, exist_ok=True)
    (work_dir / "colmap" / "sparse").mkdir(parents=True, exist_ok=True)

def _colmap_help_contains(group: str, token: str) -> bool:
    import subprocess
    try:
        out = subprocess.check_output([COLMAP_BIN, group, "-h"], text=True, stderr=subprocess.STDOUT)
        return token in out
    except Exception:
        return False


def run_colmap(work_dir: Path, _camera_info_unused=None):
    img_path = work_dir / "images"
    db_path = work_dir / "colmap" / "database.db"
    sparse_path = work_dir / "colmap" / "sparse"

    # 逐个检查模型文件夹，处理每个模型
    model_folders = sorted([folder for folder in sparse_path.iterdir() if folder.is_dir()])
    for model_folder in model_folders:
        # 如果模型文件夹存在 images.txt 和 cameras.txt 文件，就跳过
        if (model_folder / "images.txt").exists() and (model_folder / "cameras.txt").exists():
            print(f"[COLMAP] Existing sparse TXT model found in {model_folder}. Skip COLMAP.")
            continue

        # 如果仅存在 .bin 文件，则转换为 .txt 格式
        if (model_folder / "images.bin").exists():
            print(f"[COLMAP] Existing sparse BIN model found in {model_folder}. Converting to TXT...")
            run([COLMAP_BIN, "model_converter",
                 "--input_path", str(model_folder),
                 "--output_path", str(model_folder),
                 "--output_type", "TXT"])

    # ===== feature_extractor：自动估计内参（不传 camera_params）=====
    cmd = [
        COLMAP_BIN, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(img_path),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",  # 推荐：内参未知时 SIMPLE_RADIAL 最稳
    ]

    # GPU 选项：按你当前 COLMAP 实际支持的参数名自动选择（避免 unknown option 崩溃）
    if _colmap_help_contains("feature_extractor", "--SiftExtraction.use_gpu"):
        cmd += ["--SiftExtraction.use_gpu", "1"]
    elif _colmap_help_contains("feature_extractor", "--SiftExtraction.gpu_index"):
        cmd += ["--SiftExtraction.gpu_index", "0"]
    elif _colmap_help_contains("feature_extractor", "--SiftExtraction.gpu_indices"):
        cmd += ["--SiftExtraction.gpu_indices", "0"]
    else:
        print("[WARN] This COLMAP build does not expose SIFT GPU CLI options. "
              "It may still use GPU internally if compiled with CUDA, but cannot be forced via CLI.")

    if _colmap_help_contains("feature_extractor", "--SiftExtraction.num_threads"):
        cmd += ["--SiftExtraction.num_threads", "0"]

    run(cmd)

    # ===== sequential_matcher =====
    match_cmd = [
        COLMAP_BIN, "sequential_matcher",
        "--database_path", str(db_path),
        "--SequentialMatching.overlap", "10",
        "--SequentialMatching.quadratic_overlap", "1",
    ]

    if _colmap_help_contains("sequential_matcher", "--SiftMatching.use_gpu"):
        match_cmd += ["--SiftMatching.use_gpu", "1"]
    elif _colmap_help_contains("sequential_matcher", "--SiftMatching.gpu_index"):
        match_cmd += ["--SiftMatching.gpu_index", "0"]
    elif _colmap_help_contains("sequential_matcher", "--SiftMatching.gpu_indices"):
        match_cmd += ["--SiftMatching.gpu_indices", "0"]

    run(match_cmd)

    # ===== mapper（CPU为主）=====
    run([COLMAP_BIN, "mapper",
         "--database_path", str(db_path),
         "--image_path", str(img_path),
         "--output_path", str(sparse_path),
         "--Mapper.ba_global_function_tolerance", "1e-6",
    ])

    # 检查模型是否存在
    model0 = sparse_path / "0"
    if not model0.exists():
        raise RuntimeError("COLMAP mapper failed: sparse/0 not found.")

    # ===== 转 TXT 便于解析 =====
    run([
        COLMAP_BIN, "model_converter",
        "--input_path", str(model0),
        "--output_path", str(model0),
        "--output_type", "TXT"
    ])


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ], dtype=float)

def _count_registered_images(images_txt: Path) -> int:
    """COLMAP images.txt: 每张注册图像占 2 行（位姿行 + points2D行），跳过注释/空行"""
    if not images_txt.exists():
        return 0
    n_pose_lines = 0
    with images_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # 位姿行至少有 10 个字段：IMAGE_ID qw qx qy qz tx ty tz CAMERA_ID IMAGE_NAME
            if len(parts) >= 10:
                n_pose_lines += 1
                _ = next(f, None)  # 跳过 points2D 行
    return n_pose_lines


def export_colmap_traj(work_dir: Path):
    sparse_root = work_dir / "colmap" / "sparse"
    if not sparse_root.exists():
        raise RuntimeError("COLMAP sparse folder not found.")

    ts = pd.read_csv(work_dir / "timestamps.csv")
    name_to_t = {row["image"]: float(row["t"]) for _, row in ts.iterrows()}

    # 1) 自动选择注册图像最多的模型
    candidates = []
    for d in sorted(sparse_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 10**9):
        if not d.is_dir():
            continue
        images_txt = d / "images.txt"
        nreg = _count_registered_images(images_txt)
        if nreg > 0:
            candidates.append((nreg, d))
    if not candidates:
        raise RuntimeError("No valid COLMAP models found under sparse/* (no images.txt).")

    candidates.sort(reverse=True, key=lambda x: x[0])
    best_nreg, best_model = candidates[0]
    print(f"[COLMAP] Use sparse/{best_model.name} (registered images = {best_nreg})")

    images_txt = best_model / "images.txt"
    if not images_txt.exists():
        raise RuntimeError(f"images.txt not found in {best_model}")

    # 2) 解析轨迹（相机中心 C = -R^T t）
    rows = []
    with images_txt.open("r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue

            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            image_name = parts[9]
            _ = f.readline()  # points2D line

            if image_name not in name_to_t:
                continue

            qvec = np.array([qw, qx, qy, qz], dtype=float)
            tvec = np.array([tx, ty, tz], dtype=float)
            R = qvec2rotmat(qvec)
            C = -R.T @ tvec

            rows.append((name_to_t[image_name], C[0], C[1], C[2]))

    df = pd.DataFrame(rows, columns=["t", "x", "y", "z"]).sort_values("t")
    df.to_csv(work_dir / "traj_vis.csv", index=False)

    if len(df) < 30:
        raise RuntimeError(f"Too few registered images in best COLMAP model: {len(df)}")

    return df

def build_gt_from_vis_and_depth(traj_vis: pd.DataFrame, depth_df: pd.DataFrame, out_dir: Path):
    t_vis = traj_vis["t"].to_numpy()
    x = traj_vis["x"].to_numpy()
    y = traj_vis["y"].to_numpy()
    z = traj_vis["z"].to_numpy()

    t_d = depth_df["t"].to_numpy()
    d = depth_df["depth_m"].to_numpy()

    f_depth = interp1d(t_d, d, kind="linear", fill_value="extrapolate")
    d_vis = f_depth(t_vis)
    target = -d_vis

    def corr(a, b):
        a0 = a - a.mean()
        b0 = b - b.mean()
        denom = (np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-12)
        return float((a0 @ b0) / denom)

    cands = [("x", x), ("y", y), ("z", z)]
    corrs = [(name, abs(corr(arr, target))) for name, arr in cands]
    best_axis, _ = max(corrs, key=lambda t: t[1])
    best_arr = {"x": x, "y": y, "z": z}[best_axis]
    print(f"[GT] axis best correlated with depth: {best_axis}")

    A = np.vstack([best_arr, np.ones_like(best_arr)]).T
    sol, *_ = np.linalg.lstsq(A, target, rcond=None)
    s, b = float(sol[0]), float(sol[1])
    print(f"[GT] estimated scale s={s:.6f}, offset b={b:.6f}")

    x_s = s * x
    y_s = s * y
    z_gt = target

    out_dir.mkdir(parents=True, exist_ok=True)
    gt = pd.DataFrame({"t": t_vis, "x_gt": x_s, "y_gt": y_s, "z_gt": z_gt}).sort_values("t")
    gt.to_csv(out_dir / "gt_traj.csv", index=False)

    report = {
        "scale_s": s,
        "offset_b": b,
        "depth_axis_selected": best_axis,
        "note": "Pseudo-GT from COLMAP monocular trajectory; scale from depth; z locked to -depth; USBL NOT used."
    }
    (out_dir / "gt_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return gt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fps", type=int, default=FPS_LIMIT)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    seq = args.seq

    video_bag = dataset_root / f"{seq}_video.bag"
    data_bag = dataset_root / f"{seq}_data.bag"
    if not video_bag.exists():
        raise FileNotFoundError(video_bag)
    if not data_bag.exists():
        raise FileNotFoundError(data_bag)

    work_dir = out_root / seq
    if args.force and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cam = None
    if cam is None:
        print("[WARN] CameraInfo not found. COLMAP will estimate intrinsics.")
    else:
        (work_dir / "camera_info.json").write_text(json.dumps(cam, indent=2), encoding="utf-8")
        print(f"[OK] CameraInfo fx={cam['fx']:.1f} fy={cam['fy']:.1f} cx={cam['cx']:.1f} cy={cam['cy']:.1f}")

    extract_images(video_bag, work_dir, fps_limit=args.fps)
    depth_df = extract_depth(data_bag, work_dir)

    setup_colmap_dirs(work_dir)
    run_colmap(work_dir, cam)

    traj_vis = export_colmap_traj(work_dir)
    build_gt_from_vis_and_depth(traj_vis, depth_df, work_dir / "gt")

    print("\n[DONE]")
    print("Work dir:", work_dir)
    print("Visual traj:", work_dir / "traj_vis.csv")
    print("GT traj:", work_dir / "gt/gt_traj.csv")

if __name__ == "__main__":
    main()
