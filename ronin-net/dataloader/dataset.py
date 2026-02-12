"""
* This file is part of RNIN-VIO
* (Scientific Version: Implemented Global Z-Score Normalization for DVL)
"""

import pandas as pd
import numpy as np
import os
import logging
from os import path as osp
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import random
from numpy.random import normal as gen_normal
from torch.utils.data import Dataset

# =============================================================================
# 1. SenseINSSequence
# =============================================================================
class SenseINSSequence(object):
    def __init__(self, data_path, imu_freq, window_size, verbose=True, plot=False):
        super().__init__()
        (
            self.ts,
            self.features,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (None, None, None, None, None, None)
        self.imu_freq = imu_freq
        self.interval = window_size
        self.data_valid = False
        self.sum_dur = 0
        self.valid = False
        self.plot = plot

        if data_path is not None:
            self.valid = self.load(data_path, verbose=verbose)

    def load(self, data_path, verbose=True):
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        csv_file = osp.join(data_path, 'SenseINS_aligned.csv')
        h5_file = osp.join(data_path, 'SenseINS_aligned.h5')

        if osp.exists(csv_file):
            imu_all = pd.read_csv(csv_file)
        elif osp.exists(h5_file):
            imu_all = pd.read_hdf(h5_file, 'imu_all')
        else:
            logging.info(f"dataset.py: file is not exist. {csv_file}")
            return False

        # [辅助函数]
        def get_col_data(possible_names):
            for name in possible_names:
                if name in imu_all.columns:
                    return imu_all[name].values
            return None

        # --- 1. 读取时间戳 ---
        tmp_ts = get_col_data(['timestamp', 'time', 'times'])
        if tmp_ts is None:
            logging.error("No time column found")
            return False
        if tmp_ts.shape[0] < 100:
            return False

        # --- 2. 读取姿态 (GT Quaternion) ---
        q_x = get_col_data(['gt_q_x', 'qx'])
        q_y = get_col_data(['gt_q_y', 'qy'])
        q_z = get_col_data(['gt_q_z', 'qz'])
        q_w = get_col_data(['gt_q_w', 'qw'])

        if q_x is not None and q_w is not None:
            tmp_vio_q = np.stack([q_x, q_y, q_z, q_w], axis=1)
            self.get_gt = True
        else:
            logging.warning(f"GT Quaternion missing, utilizing identity.")
            tmp_vio_q = np.zeros((len(tmp_ts), 4))
            tmp_vio_q[:, 3] = 1.0
            self.get_gt = False

        # --- 3. 读取位置 (GT Position) ---
        p_x = get_col_data(['gt_p_x', 'gt_px', 'pos_x'])
        p_y = get_col_data(['gt_p_y', 'gt_py', 'pos_y'])
        p_z = get_col_data(['depth', 'gt_p_z', 'gt_pz', 'est_pz', 'pos_z'])

        if p_x is not None:
            if p_z is None: p_z = np.zeros_like(p_x)
            tmp_vio_p = np.stack([p_x, p_y, p_z], axis=1)
        else:
            tmp_vio_p = np.zeros((len(tmp_ts), 3))

        # --- 4. 读取传感器数据 ---
        # Gyro
        g_x = get_col_data(['gyro_x', 'gryX'])
        g_y = get_col_data(['gyro_y', 'gryY'])
        g_z = get_col_data(['gyro_z', 'gryZ'])
        tmp_gyro = np.stack([g_x, g_y, g_z], axis=1) if g_x is not None else np.zeros((len(tmp_ts), 3))

        # Acc
        a_x = get_col_data(['acce_x', 'accX'])
        a_y = get_col_data(['acce_y', 'axxY'])
        a_z = get_col_data(['acce_z', 'accZ'])
        tmp_acce = np.stack([a_x, a_y, a_z], axis=1) if a_x is not None else np.zeros((len(tmp_ts), 3))

        # DVL
        d_x = get_col_data(['dvl_vx', 'dvl_x'])
        d_y = get_col_data(['dvl_vy', 'dvl_y'])
        d_z = get_col_data(['dvl_vz', 'dvl_z'])

        has_dvl_sensor = (d_x is not None)
        if has_dvl_sensor:
            tmp_dvl = np.stack([d_x, d_y, d_z], axis=1)
        else:
            tmp_dvl = np.zeros((len(tmp_ts), 3))

        # --- 5. 插值对齐 (Resampling) ---
        start_ts = tmp_ts[0] + 0.05
        end_ts = tmp_ts[-1] - 0.05
        if start_ts >= end_ts:
             start_ts = tmp_ts[0]
             end_ts = tmp_ts[-1]

        ts = np.arange(start_ts, end_ts, 1.0 / self.imu_freq)
        self.data_valid = True
        self.sum_dur = end_ts - start_ts

        # 姿态插值
        try:
            key_rots = Rotation.from_quat(tmp_vio_q)
            vio_q_slerp = Slerp(tmp_ts, key_rots)
            vio_r = vio_q_slerp(ts)
        except ValueError:
            f_nearest = interp1d(tmp_ts, tmp_vio_q, axis=0, kind='nearest')
            vio_r = Rotation.from_quat(f_nearest(ts))

        # 线性插值
        def robust_interp(y):
            return interp1d(tmp_ts, y, axis=0, bounds_error=False, fill_value=(y[0], y[-1]))(ts)

        vio_p = robust_interp(tmp_vio_p)
        gyro = robust_interp(tmp_gyro)
        acce = robust_interp(tmp_acce)
        dvl = robust_interp(tmp_dvl)

        ts = ts[:, np.newaxis]
        ori_R = vio_r

        gt_disp = np.zeros_like(vio_p)
        if len(vio_p) > self.interval:
            gt_disp[:-self.interval] = vio_p[self.interval:] - vio_p[:-self.interval]

        # --- 6. 构建特征 (Feature Construction) ---
        # IMU (Body) -> Global
        glob_gyro = np.einsum("tip,tp->ti", ori_R.as_matrix(), gyro)
        glob_acce = np.einsum("tip,tp->ti", ori_R.as_matrix(), acce)

        # [修改] 移除所有硬编码缩放 (Scale)，只进行必要的坐标系旋转
        # 归一化逻辑上移至 BasicSequenceData 类中统一处理
        if has_dvl_sensor:
            glob_dvl = np.einsum("tip,tp->ti", ori_R.as_matrix(), dvl)
        else:
            glob_dvl = dvl

        # 去重力
        glob_acce -= np.array([0.0, 0.0, 9.81])

        self.features = np.concatenate([glob_gyro, glob_acce, glob_dvl], axis=1)

        self.ts = ts
        self.orientations = ori_R.as_quat()
        self.gt_pos = vio_p
        self.gt_ori = ori_R.as_quat()
        self.targets = gt_disp

        if verbose:
            logging.info(f"Loaded {data_path}: duration {self.sum_dur:.2f}s, feature shape {self.features.shape}")

        return True

    def get_feature(self): return self.features
    def get_target(self): return self.targets
    def get_data_valid(self): return self.data_valid
    def get_aux(self): return np.concatenate([self.ts, self.orientations, self.gt_pos, self.gt_ori], axis=1)


# =============================================================================
# 2. BasicSequenceData: (新增 Z-Score 归一化逻辑)
# =============================================================================
class BasicSequenceData(object):
    def __init__(self, cfg, data_list, verbose=True, **kwargs):
        super(BasicSequenceData, self).__init__()
        self.window_size = int(cfg['model_param']['window_time'] * cfg['data']['imu_freq'])
        self.past_data_size = int(cfg['model_param']['past_time'] * cfg['data']['imu_freq'])
        self.future_data_size = int(cfg['model_param']['future_time'] * cfg['data']['imu_freq'])
        self.step_size = int(cfg['data']['imu_freq'] / cfg['data']['sample_freq'])
        self.seq_len = cfg['train']["seq_len"]

        self.index_map = []
        self.ts, self.orientations, self.gt_pos, self.gt_ori = [], [], [], []
        self.features, self.targets = [], []
        self.valid_t, self.valid_samples = [], []
        self.data_paths = []
        self.valid_continue_good_time = 0.1

        self.mode = kwargs.get("mode", "train")
        sum_t = 0
        win_dt = self.window_size / cfg['data']['imu_freq']
        self.valid_sum_t = 0
        self.valid_all_samples = 0
        max_v_norm = 4.0
        valid_i = 0

        # 1. 加载所有数据
        for i in range(len(data_list)):
            seq = SenseINSSequence(
                data_list[i], cfg['data']['imu_freq'], self.window_size, verbose=verbose
            )
            if seq.valid is False:
                continue
            feat, targ, aux = seq.get_feature(), seq.get_target(), seq.get_aux()
            sum_t += seq.sum_dur
            valid_samples = 0
            index_map = []
            step_size = self.step_size

            safe_end = targ.shape[0] - self.future_data_size - (self.seq_len * self.window_size)

            if safe_end <= self.past_data_size:
                continue

            if self.mode in ["train", "val"] and getattr(seq, 'get_gt', True):
                for j in range(self.past_data_size, safe_end, step_size):
                    outlier = False
                    for k in range(self.seq_len):
                        index = j + k * self.window_size
                        if index >= targ.shape[0]:
                            outlier = True
                            break
                        velocity = np.linalg.norm(targ[index] / win_dt)
                        if velocity > max_v_norm:
                            outlier = True
                            break
                    if outlier is False:
                        index_map.append([valid_i, j])
                        self.valid_all_samples += 1
                        valid_samples += 1
            else:
                for j in range(self.past_data_size, safe_end, step_size):
                    index_map.append([valid_i, j])
                    self.valid_all_samples += 1
                    valid_samples += 1

            if len(index_map) > 0:
                self.data_paths.append(data_list[i])
                self.index_map.append(index_map)
                self.features.append(feat)
                self.targets.append(targ)
                self.ts.append(aux[:, 0])
                self.orientations.append(aux[:, 1:5])
                self.gt_pos.append(aux[:, 5:8])
                self.gt_ori.append(aux[:, 8:12])
                self.valid_samples.append(valid_samples)
                valid_i += 1

        if verbose:
            logging.info(f"datasets sum time {sum_t}")

        # =========================================================================
        # 🔥 [新增] 科学的 Z-Score 归一化 (Scientific Data Normalization)
        # =========================================================================
        if len(self.features) > 0:
            # 1. 拼接所有序列以计算全局统计量
            all_feats = np.concatenate(self.features, axis=0)

            # 假设 DVL 在最后 3 列 (index 6, 7, 8)
            # feature结构: [Gyro(3), Acc(3), DVL(3)]
            dvl_data = all_feats[:, 6:9]

            # 2. 计算均值和标准差
            self.dvl_mean = np.mean(dvl_data, axis=0)
            self.dvl_std = np.std(dvl_data, axis=0) + 1e-6 # 加微小值防除零

            if verbose:
                logging.info(f"[{self.mode.upper()}] Global DVL Stats:")
                logging.info(f"   Mean: {self.dvl_mean}")
                logging.info(f"   Std : {self.dvl_std}")

            # 3. 应用归一化: (x - mean) / std
            # 这会将 DVL 分布拉伸到 N(0, 1)，与 Acc/Gyro 的数量级对齐
            for i in range(len(self.features)):
                self.features[i][:, 6:9] = (self.features[i][:, 6:9] - self.dvl_mean) / self.dvl_std

    def get_data(self):
        return self.features, self.targets, self.ts, self.orientations, self.gt_pos, self.gt_ori

    def get_index_map(self):
        return self.index_map

    def get_merged_index_map(self):
        index_map = []
        for i in range(len(self.index_map)):
            index_map += self.index_map[i]
        return index_map


# =============================================================================
# 3. ResNetLSTMSeqToSeqDataset: (完全还原，不做修改)
# =============================================================================
class ResNetLSTMSeqToSeqDataset(Dataset):
    def __init__(self, cfg, basic_data: BasicSequenceData, index_map, **kwargs):
        super(ResNetLSTMSeqToSeqDataset, self).__init__()
        self.window_size = basic_data.window_size
        self.past_data_size = basic_data.past_data_size
        self.future_data_size = basic_data.future_data_size
        self.step_size = basic_data.step_size
        self.seq_len = basic_data.seq_len

        self.add_bias_noise = cfg['augment']['add_bias_noise']
        self.accel_bias_range = cfg['augment']['accel_bias_range']
        self.gyro_bias_range = cfg['augment']['gyro_bias_range']
        if self.add_bias_noise is False:
            self.accel_bias_range = 0.0
            self.gyro_bias_range = 0.0
        self.add_gravity_noise = cfg['augment']['add_gravity_noise']
        self.gravity_noise_theta_range = cfg['augment']['gravity_noise_theta_range']

        self.feat_acc_sigma = cfg['augment']['feat_acc_sigma']
        self.feat_gyr_sigma = cfg['augment']['feat_gyr_sigma']

        self.mode = kwargs.get("mode", "train")
        self.shuffle = False
        self.transform = False
        self.gauss = False
        if self.mode == "train":
            self.shuffle = True
            self.transform = True
            self.gauss = True
        elif self.mode == "val":
            self.shuffle = True
        elif self.mode == "test":
            self.shuffle = False

        self.features, self.targets, self.ts, self.orientations, self.gt_pos, self.gt_ori = basic_data.get_data()
        self.index_map = index_map
        if self.shuffle:
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        
        feat = self.features[seq_id][frame_id - self.past_data_size:
                                     frame_id + self.seq_len * self.window_size + self.future_data_size]

        targ = self.targets[seq_id][frame_id:
                                    frame_id + self.seq_len * self.window_size:
                                    self.window_size] 

        if self.mode in ["train"]:
            targ_aug = np.copy(targ)
            feat_aug = np.copy(feat)
            if self.transform:
                angle = np.random.random() * (2 * np.pi)
                rm = np.array(
                    [[np.cos(angle), -(np.sin(angle))], [np.sin(angle), np.cos(angle)]]
                )
                feat_aug[:, 0:2] = np.matmul(rm, feat_aug[:, 0:2].T).T
                feat_aug[:, 3:5] = np.matmul(rm, feat_aug[:, 3:5].T).T
                if feat_aug.shape[1] >= 8:
                    feat_aug[:, 6:8] = np.matmul(rm, feat_aug[:, 6:8].T).T

                targ_aug[:, 0:2] = np.matmul(rm, targ_aug[:, 0:2].T).T

            if self.add_bias_noise:
                random_bias = np.random.random((1, 6))
                random_bias[:, 0:3] = (random_bias[:, 0:3] - 0.5) * self.gyro_bias_range / 0.5
                random_bias[:, 3:6] = (random_bias[:, 3:6] - 0.5) * self.accel_bias_range / 0.5
                feat_aug[:, 0:6] += random_bias

            if self.add_gravity_noise:
                angle_rand = random.random() * np.pi * 2
                vec_rand = np.array([np.cos(angle_rand), np.sin(angle_rand), 0])
                theta_rand = (
                        random.random() * np.pi * self.gravity_noise_theta_range / 180.0
                )
                rvec = theta_rand * vec_rand
                r = Rotation.from_rotvec(rvec)
                R_mat = r.as_matrix()
                feat_aug[:, 0:3] = np.matmul(R_mat, feat_aug[:, 0:3].T).T
                feat_aug[:, 3:6] = np.matmul(R_mat, feat_aug[:, 3:6].T).T
                if feat_aug.shape[1] >= 9:
                    feat_aug[:, 6:9] = np.matmul(R_mat, feat_aug[:, 6:9].T).T

            if self.gauss:
                if self.feat_gyr_sigma > 0:
                    feat_aug[:, 0:3] += gen_normal(loc=0.0, scale=self.feat_gyr_sigma, size=(len(feat_aug[:, 0]), 3))
                if self.feat_acc_sigma > 0:
                    feat_aug[:, 3:6] += gen_normal(loc=0.0, scale=self.feat_acc_sigma, size=(len(feat_aug[:, 0]), 3))
            feat = feat_aug
            targ = targ_aug

        seq_feat = []
        for i in range(self.seq_len):
            seq_feat.append(feat[i * self.window_size:
                                 self.past_data_size + (
                                             i + 1) * self.window_size + self.future_data_size, :].T)
        seq_feat = np.array(seq_feat)
        return seq_feat.astype(np.float32), targ.astype(np.float32)

    def __len__(self):
        return len(self.index_map)


def SeqToSeqDataset(cfg, basic_data: BasicSequenceData, index_map, **kwargs):
    return ResNetLSTMSeqToSeqDataset(cfg, basic_data, index_map, **kwargs)


# =============================================================================
# 4. partition_data: 数据划分函数
# =============================================================================
def partition_data(index_map, valid_samples, valid_all_samples, training_rate=0.9, valuation_rate=0.1, data_rate=1.0,
                   shuffle=False, **kwargs):
    
    if shuffle:
        pass

    all_size = 0
    sum_valid_samples = valid_all_samples * data_rate

    accum_samples = 0.0
    for i in range(len(index_map)):
        # index_map[i] 代表一个完整序列的所有帧索引: [[seq_0, frame_0], [seq_0, frame_1], ...]
        # 我们累加每个序列的样本数
        accum_samples += valid_samples[index_map[i][0][0]]
        all_size = i
        if accum_samples > sum_valid_samples:
            break

    valuation_samples = sum_valid_samples * valuation_rate

    train_index_map, valuation_index_map = [], []
    accum_valuation_samples = 0

    for i in range(all_size):
        if accum_valuation_samples < valuation_samples:
            valuation_index_map += index_map[i]
            accum_valuation_samples += valid_samples[index_map[i][0][0]]
        else:
            train_index_map += index_map[i]

    return train_index_map, valuation_index_map