# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import os
import glob
import pickle
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation
from torch.utils.data import Dataset
from src.utils import pdump, pload, bmtm
from src.utils import SO3


# =============================================================================
# 1. SenseINSSequence Class
# =============================================================================
class SenseINSSequence(object):
    def __init__(self, data_path, imu_freq=100.0, verbose=True):
        self.valid = False
        self.features = None
        self.gt_qs = None
        self.gt_ps = None
        self.ts = None
        self.imu_freq = imu_freq

        if data_path and os.path.exists(data_path):
            self.valid = self.load(data_path, verbose)

    def load(self, csv_file, verbose=True):
        try:
            # 1. Read CSV
            df = pd.read_csv(csv_file)
            df = df.loc[:, ~df.columns.duplicated()]

            # Timestamp processing
            ts_col = 'timestamp' if 'timestamp' in df.columns else 'time'
            if ts_col not in df.columns: return False
            df[ts_col] = pd.to_numeric(df[ts_col], errors='coerce')
            df = df[np.isfinite(df[ts_col])].sort_values(by=ts_col).drop_duplicates(subset=[ts_col],
                                                                                    keep='first').reset_index(drop=True)
            if len(df) < 50: return False

            raw_ts = df[ts_col].values

            # 2. Extract IMU
            gyro_map = {'x': ['gyro_x', 'gryX'], 'y': ['gyro_y', 'gryY'], 'z': ['gyro_z', 'gryZ']}
            acce_map = {'x': ['acce_x', 'accX'], 'y': ['acce_y', 'axxY'], 'z': ['acce_z', 'accZ']}

            def get_col(m):
                return next((c for c in m if c in df.columns), None)

            gx, gy, gz = get_col(gyro_map['x']), get_col(gyro_map['y']), get_col(gyro_map['z'])
            ax, ay, az = get_col(acce_map['x']), get_col(acce_map['y']), get_col(acce_map['z'])

            if not (gx and gy and gz and ax and ay and az): return False

            gyro = df[[gx, gy, gz]].values.astype(float)
            acce = df[[ax, ay, az]].values.astype(float)

            # 3. Extract GT
            q_scipy = None
            if 'gt_q_w' in df.columns and 'gt_q_x' in df.columns:
                q_scipy = df[['gt_q_x', 'gt_q_y', 'gt_q_z', 'gt_q_w']].values
            elif 'roll' in df.columns or 'roll_body' in df.columns:
                r_col = 'roll' if 'roll' in df.columns else 'roll_body'
                p_col = 'pitch' if 'pitch' in df.columns else 'pitch_body'
                y_col = 'yaw' if 'yaw' in df.columns else 'yaw_body'
                euler = df[[r_col, p_col, y_col]].values

                # Check GT Unit
                is_degrees = np.max(np.abs(euler)) > 6.3
                q_scipy = Rotation.from_euler('xyz', np.nan_to_num(euler), degrees=is_degrees).as_quat()
            else:
                return False

            # Unit Conversion
            if np.max(np.abs(gyro)) > 10.0:
                if verbose: print(f"[INFO] Converting {os.path.basename(csv_file)} to RADIANS (detected deg).")
                gyro = np.deg2rad(gyro)
            else:
                if verbose: print(f"[INFO] Keeping {os.path.basename(csv_file)} as is (detected rad or slow motion).")

            # 4. Extract Position
            pos = np.zeros((len(df), 3))
            if 'gt_p_x' in df.columns:
                pos = df[['gt_p_x', 'gt_p_y', 'gt_p_z']].fillna(0).values
            elif 'gt_px' in df.columns:
                pos[:, 0] = df['gt_px'].values
                if 'gt_py' in df.columns: pos[:, 1] = df['gt_py'].values
                if 'depth' in df.columns:
                    pos[:, 2] = df['depth'].values
                elif 'gt_pz' in df.columns:
                    pos[:, 2] = df['gt_pz'].values
            elif 'est_pz' in df.columns:
                pos[:, 2] = df['est_pz'].values

            # 5. Interpolation
            t_start, t_end = raw_ts[0], raw_ts[-1]
            num_samples = int((t_end - t_start) * self.imu_freq)
            if num_samples < 10: return False
            new_ts = np.linspace(t_start, t_end, num_samples)

            f_g = interp1d(raw_ts, gyro, axis=0, kind='linear', fill_value="extrapolate")
            f_a = interp1d(raw_ts, acce, axis=0, kind='linear', fill_value="extrapolate")
            interp_g = f_g(new_ts)
            interp_a = f_a(new_ts)

            q_scipy /= (np.linalg.norm(q_scipy, axis=1, keepdims=True) + 1e-8)
            slerp = Slerp(raw_ts, Rotation.from_quat(q_scipy))
            interp_q = slerp(new_ts).as_quat()

            f_p = interp1d(raw_ts, pos, axis=0, kind='linear', fill_value="extrapolate")
            interp_p = f_p(new_ts)

            # 6. Safety Checks
            q_wxyz = interp_q[:, [3, 0, 1, 2]]
            if np.max(np.linalg.norm(q_wxyz, axis=1)) > 2.0: return False
            if np.max(np.abs(interp_g)) > 5000.0: return False

            self.features = np.concatenate([interp_g, interp_a], axis=1)
            self.gt_qs = q_wxyz
            self.gt_ps = interp_p
            self.ts = new_ts - t_start

            return True

        except Exception as e:
            if verbose: print(f"[ERR] Error loading {csv_file}: {e}")
            return False


# =============================================================================
# 2. BaseDataset (🔥 核心修复：恢复统计计算 & 健全填充)
# =============================================================================
class BaseDataset(Dataset):
    def __init__(self, predata_dir, train_seqs, val_seqs, test_seqs, mode, N,
                 min_train_freq=128, max_train_freq=512, dt=0.01):
        super().__init__()
        self.predata_dir = predata_dir
        self.path_normalize_factors = os.path.join(predata_dir, 'nf.p')
        self.mode = mode
        self.dt = dt
        self.N = N
        self.min_train_freq = min_train_freq
        self.max_train_freq = max_train_freq

        train_seqs, self.sequences = self.get_sequences(train_seqs, val_seqs, test_seqs)

        # 🔥 [恢复] 真实的均值和方差计算
        # 这将允许 Expert 网络学习 (u - mean) / std，从而去除零偏
        self.mean_u, self.std_u = self.init_normalize_factors(train_seqs)

        # 动态噪声参数
        self.imu_std = torch.Tensor([0.01, 0.3]).float()
        self.imu_b0 = torch.Tensor([0.05, 0.2]).float()

        self._train = (mode == 'train')
        self._val = (mode == 'val')

    def get_sequences(self, train_seqs, val_seqs, test_seqs):
        sequences_dict = {'train': train_seqs, 'val': val_seqs, 'test': test_seqs}
        return sequences_dict['train'], sequences_dict[self.mode]

    def __len__(self):
        if self._train:
            return len(self.sequences) * 100
        else:
            return len(self.sequences)

    def __getitem__(self, i):
        idx = i % len(self.sequences)
        mondict = self.load_seq(idx)

        u = mondict['us']
        x = mondict['xs']
        N_max = u.shape[0]

        import torch.nn.functional as F
        if self._train:
            if N_max < self.N:
                pad_len = self.N - N_max
                # 🔥 [优化] 使用 replicate 填充，保持信号幅度，防止拉低均值
                # F.pad replicate 模式需要 3D/4D 输入，且对最后维度操作
                # (L, D) -> (1, D, L)
                u = u.permute(1, 0).unsqueeze(0)
                u = F.pad(u, (0, pad_len), mode='replicate')
                u = u.squeeze(0).permute(1, 0) # 还原 (L+pad, D)

                # GT 也同步填充
                x = x.permute(1, 0).unsqueeze(0)
                x = F.pad(x, (0, pad_len), mode='replicate')
                x = x.squeeze(0).permute(1, 0)
            else:
                high = N_max - self.N
                n0 = torch.randint(0, high + 1, (1,)).item() if high > 0 else 0
                nend = n0 + self.N
                u = u[n0: nend]
                x = x[n0: nend]

        else:
            n0 = 0
            nend = N_max - (N_max % self.max_train_freq)
            u = u[n0: nend]
            x = x[n0: nend]

        if self._train:
            u = self.add_noise(u.unsqueeze(0)).squeeze(0)

        return {'u': u, 'x': x}

    def add_noise(self, u):
        if u.dim() == 2:
            u = u.unsqueeze(0);
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len, dim = u.shape
        device = u.device

        max_std = self.imu_std.to(device)
        max_b0 = self.imu_b0.to(device)

        rand_std_scale = torch.rand(batch_size, 1, 1, device=device)
        rand_bias_scale = torch.rand(batch_size, 1, 1, device=device)

        # A. 比例误差
        scale_noise = 1.0 + (torch.rand(batch_size, 1, dim, device=device) * 2 - 1) * 0.02
        u_scaled = u * scale_noise

        # B. 白噪声
        noise = torch.randn_like(u)
        noise[:, :, :3] *= (max_std[0] * rand_std_scale)
        noise[:, :, 3:6] *= (max_std[1] * rand_std_scale)

        # C. 随机零偏
        b0_dir = (torch.rand(batch_size, 1, dim, device=device) * 2 - 1)
        b0 = b0_dir
        b0[:, :, :3] *= (max_b0[0] * rand_bias_scale)
        b0[:, :, 3:6] *= (max_b0[1] * rand_bias_scale)

        out = u_scaled + noise + b0

        if squeeze_output:
            out = out.squeeze(0)
        return out

    def init_train(self):
        self._train = True;
        self._val = False

    def init_val(self):
        self._train = False;
        self._val = True

    def load_seq(self, i):
        return pload(self.predata_dir, self.sequences[i] + '.p')

    def load_gt(self, i):
        return pload(self.predata_dir, self.sequences[i] + '_gt.p')

    # 🔥 [重要修复] 恢复真实的统计计算逻辑
    def init_normalize_factors(self, train_seqs):
        if len(train_seqs) == 0:
            print("[INFO] No training sequences. Using default stats (Raw Mode).")
            return torch.zeros(6), torch.ones(6)

        print(f"[INFO] Computing statistics for {len(train_seqs)} training sequences...")
        all_us = []
        for seq in train_seqs:
            try:
                data = pload(self.predata_dir, seq + '.p')
                all_us.append(data['us'])
            except:
                continue

        if not all_us:
            return torch.zeros(6), torch.ones(6)

        all_us = torch.cat(all_us, dim=0)
        mean_u = all_us.mean(dim=0)
        std_u = all_us.std(dim=0)

        # 防止 std 过小导致除零
        std_u[std_u < 1e-6] = 1.0

        print(f"[INFO] Stats computed.\n   Mean: {mean_u[:3].numpy()}\n   Std : {std_u[:3].numpy()}")
        return mean_u, std_u


# =============================================================================
# 3. EUROCDataset
# =============================================================================
class EUROCDataset(BaseDataset):
    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs, test_seqs, mode, N, min_train_freq, max_train_freq,
                 dt=0.01):
        self.predata_dir = predata_dir
        self.min_train_freq = min_train_freq
        self.dt = dt

        self.convert_raw_data(data_dir)
        train_seqs = self.filter_missing(train_seqs)
        val_seqs = self.filter_missing(val_seqs)
        test_seqs = self.filter_missing(test_seqs)

        super().__init__(predata_dir, train_seqs, val_seqs, test_seqs, mode, N, min_train_freq, max_train_freq, dt)

    def filter_missing(self, seq_list):
        valid_list = []
        for seq in seq_list:
            p_path = os.path.join(self.predata_dir, seq + '.p')
            if os.path.exists(p_path):
                valid_list.append(seq)
            else:
                print(f"[WARN] Sequence '{seq}' not found in cache. Removed from dataset.")
        return valid_list

    def convert_raw_data(self, data_dir):
        if not os.path.exists(self.predata_dir): os.makedirs(self.predata_dir)

        csv_files = glob.glob(os.path.join(data_dir, "*", "SenseINS_aligned.csv"))
        if not csv_files: csv_files = glob.glob(os.path.join(data_dir, "SenseINS_aligned.csv"))
        if not csv_files:
            csv_files = glob.glob(os.path.join(data_dir, "*", "*.csv")) + glob.glob(os.path.join(data_dir, "*.csv"))
            csv_files = [f for f in csv_files if
                         "SenseINS" in f or "suiyi" in f or "zhengfangxing" in f or "zhixian" in f]

        if not csv_files:
            print("No valid csv files found!")
            return

        print(f"Found {len(csv_files)} files. Checking cache...")

        for csv_path in csv_files:
            seq_name = os.path.basename(os.path.dirname(csv_path))
            if not seq_name or seq_name == "processed_data":
                seq_name = os.path.splitext(os.path.basename(csv_path))[0]

            p_path = os.path.join(self.predata_dir, seq_name + '.p')
            if os.path.exists(p_path): continue

            print(f"Processing: {seq_name} from {csv_path}...")
            seq_processor = SenseINSSequence(csv_path, imu_freq=1 / self.dt)

            if not seq_processor.valid:
                print(f"Skipping invalid file: {csv_path}")
                continue

            us = torch.from_numpy(seq_processor.features).float()
            qs = torch.from_numpy(seq_processor.gt_qs).double()
            ps = torch.from_numpy(seq_processor.gt_ps).float()

            mtf = self.min_train_freq
            if qs.shape[0] <= mtf: continue

            R_all = SO3.from_quaternion(qs.to('cpu'), ordering='wxyz')
            R_i = R_all[:-mtf]
            R_j = R_all[mtf:]
            dRot = bmtm(R_i, R_j)
            xs = SO3.log(dRot).float()

            xs_mask = torch.ones(xs.shape[0], 1)
            xs = torch.cat([xs, xs_mask], dim=1)

            data_dict = {'us': us[:-mtf], 'xs': xs, 't': seq_processor.ts[:-mtf]}
            pdump(data_dict, self.predata_dir, seq_name + '.p')

            gt_dict = {'qs': qs.float(), 'ps': ps, 'ts': seq_processor.ts}
            pdump(gt_dict, self.predata_dir, seq_name + '_gt.p')
            print(f"Converted: {seq_name}")