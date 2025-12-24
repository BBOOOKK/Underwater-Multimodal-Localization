from src.utils import pdump, pload, bmtv, bmtm
from src.lie_algebra import SO3
from termcolor import cprint
from torch.utils.data.dataset import Dataset
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch
import sys


class BaseDataset(Dataset):

    def __init__(self, predata_dir, train_seqs, val_seqs, test_seqs, mode, N,
                 min_train_freq=128, max_train_freq=512, dt=0.01):
        super().__init__()
        # where record pre loaded data
        self.predata_dir = predata_dir
        self.path_normalize_factors = os.path.join(predata_dir, 'nf.p')

        self.mode = mode
        # choose between training, validation or test sequences
        train_seqs, self.sequences = self.get_sequences(train_seqs, val_seqs,
                                                        test_seqs)

        # get and compute value for normalizing inputs
        # 这个函数会依赖 pickle 文件里的 'us' 键
        self.mean_u, self.std_u = self.init_normalize_factors(train_seqs)

        self.mode = mode  # train, val or test
        self._train = False
        self._val = False
        # noise density
        self.imu_std = torch.Tensor([8e-5, 1e-3]).float()
        # bias repeatability (without in-run bias stability)
        self.imu_b0 = torch.Tensor([1e-3, 1e-3]).float()
        # IMU sampling time
        self.dt = dt  # (s)
        # sequence size during training
        self.N = N  # power of 2
        self.min_train_freq = min_train_freq
        self.max_train_freq = max_train_freq
        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1),
                                                       torch.ones(1))

    def get_sequences(self, train_seqs, val_seqs, test_seqs):
        """Choose sequence list depending on dataset mode"""
        sequences_dict = {
            'train': train_seqs,
            'val': val_seqs,
            'test': test_seqs,
        }
        return sequences_dict['train'], sequences_dict[self.mode]

    def __getitem__(self, i):
        mondict = self.load_seq(i)
        N_max = mondict['xs'].shape[0]
        if self._train:  # random start
            n0 = torch.randint(0, self.max_train_freq, (1,))
            nend = n0 + self.N
        elif self._val:  # end sequence
            n0 = self.max_train_freq + self.N
            nend = N_max - ((N_max - n0) % self.max_train_freq)
        else:  # full sequence
            n0 = 0
            nend = N_max - (N_max % self.max_train_freq)

        # 这里的索引非常关键，xs 的长度比 us 短 min_train_freq
        # us[0] 对应 xs[0] (它是 R_0 到 R_n 的增量)
        u = mondict['us'][n0: nend]
        x = mondict['xs'][n0: nend]
        return u, x

    def __len__(self):
        return len(self.sequences)

    def add_noise(self, u):
        """Add Gaussian noise and bias to input"""
        noise = torch.randn_like(u)
        noise[:, :, :3] = noise[:, :, :3] * self.imu_std[0]
        noise[:, :, 3:6] = noise[:, :, 3:6] * self.imu_std[1]

        # bias repeatability (without in run bias stability)
        # 注意：这里要确保 device 一致
        if u.is_cuda:
            self.uni = torch.distributions.uniform.Uniform(
                -torch.ones(1).cuda(), torch.ones(1).cuda())
            self.imu_b0 = self.imu_b0.cuda()

        b0 = self.uni.sample(u[:, 0].shape)
        if not u.is_cuda:  # 如果 CPU，确保 b0 也是 CPU
            b0 = b0.cpu()

        b0[:, :, :3] = b0[:, :, :3] * self.imu_b0[0]
        b0[:, :, 3:6] = b0[:, :, 3:6] * self.imu_b0[1]
        u = u + noise + b0.transpose(1, 2)
        return u

    def init_train(self):
        self._train = True
        self._val = False

    def init_val(self):
        self._train = False
        self._val = True

    def length(self):
        return self._length

    def load_seq(self, i):
        return pload(self.predata_dir, self.sequences[i] + '.p')

    def load_gt(self, i):
        return pload(self.predata_dir, self.sequences[i] + '_gt.p')

    def init_normalize_factors(self, train_seqs):
        if os.path.exists(self.path_normalize_factors):
            mondict = pload(self.path_normalize_factors)
            return mondict['mean_u'], mondict['std_u']

        # 检查第一个文件是否存在
        path = os.path.join(self.predata_dir, train_seqs[0] + '.p')
        if not os.path.exists(path):
            print("init_normalize_factors: First training file not found:", path)
            return torch.zeros(6), torch.ones(6)

        print('Start computing normalizing factors ...')
        cprint("Do it only on training sequences, it is vital!", 'yellow')
        # first compute mean
        num_data = 0

        mean_u = 0
        for i, sequence in enumerate(train_seqs):
            pickle_dict = pload(self.predata_dir, sequence + '.p')
            # 这里必须要有 'us'，否则前面步骤出错
            if 'us' not in pickle_dict:
                print(f"Error: 'us' not found in {sequence}.p. Run read_data first.")
                continue
            us = pickle_dict['us']
            if i == 0:
                mean_u = us.sum(dim=0)
            else:
                mean_u += us.sum(dim=0)
            num_data += us.shape[0]

        if num_data == 0:
            return torch.zeros(6), torch.ones(6)

        mean_u = mean_u / num_data

        # second compute standard deviation
        std_u = 0
        for i, sequence in enumerate(train_seqs):
            pickle_dict = pload(self.predata_dir, sequence + '.p')
            if 'us' not in pickle_dict: continue
            us = pickle_dict['us']
            if i == 0:
                std_u = ((us - mean_u) ** 2).sum(dim=0)
            else:
                std_u += ((us - mean_u) ** 2).sum(dim=0)
        std_u = (std_u / num_data).sqrt()

        normalize_factors = {
            'mean_u': mean_u,
            'std_u': std_u,
        }
        print('... ended computing normalizing factors')
        print('mean_u    :', mean_u)
        print('std_u     :', std_u)
        print('num_data  :', num_data)
        pdump(normalize_factors, self.path_normalize_factors)
        return mean_u, std_u

    def read_data(self, data_dir):
        raise NotImplementedError

    @staticmethod
    def interpolate(x, t, t_int):
        """
        Interpolate ground truth at the sensor timestamps
        """

        # vector interpolation
        x_int = np.zeros((t_int.shape[0], x.shape[1]))
        for i in range(x.shape[1]):
            if i in [4, 5, 6, 7]:
                continue
            x_int[:, i] = np.interp(t_int, t, x[:, i])
        # quaternion interpolation
        t_int = torch.Tensor(t_int - t[0])
        t = torch.Tensor(t - t[0])
        qs = SO3.qnorm(torch.Tensor(x[:, 4:8]))
        x_int[:, 4:8] = SO3.qinterp(qs, t, t_int).numpy()
        return x_int


class EUROCDataset(BaseDataset):
    """
    Modified Dataloader for Custom Data (Auto-convert format).
    """

    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs,
                 test_seqs, mode, N, min_train_freq, max_train_freq, dt=0.01):
        self.predata_dir = predata_dir
        self.min_train_freq = min_train_freq
        self.dt = dt

        # [修正点] 在父类初始化前，先进行数据格式转换
        self.read_data(data_dir)

        # [重要] 建议删除旧的归一化参数，因为数据变了
        nf_path = os.path.join(predata_dir, 'nf.p')
        if os.path.exists(nf_path) and mode == 'train':
            print("⚠️ 警告: 检测到旧的归一化文件 nf.p，建议手动删除它以重新计算！")
            # os.remove(nf_path) # 如果你想自动删除，取消注释这行

        super().__init__(predata_dir, train_seqs, val_seqs, test_seqs, mode, N, min_train_freq, max_train_freq, dt)

    def read_data(self, data_dir):
        r"""
        Check and convert data to model format (us, xs).
        """
        print("Checking data format in:", self.predata_dir)

        if not os.path.exists(self.predata_dir):
            print(f"Error: predata_dir {self.predata_dir} does not exist.")
            return

        files = os.listdir(self.predata_dir)
        for f in files:
            # 只处理 .p 文件，且排除 _gt.p, nf.p 等辅助文件
            if not f.endswith('.p') or '_gt' in f or 'nf.p' in f:
                continue

            file_path = os.path.join(self.predata_dir, f)
            try:
                data = pload(self.predata_dir, f)
            except Exception as e:
                print(f"Could not load {f}: {e}")
                continue

            # ==================================================================
            # 情况 A: 已经是处理好的数据 (包含 us 和 xs)
            # ==================================================================
            # 即使已经包含 us, xs，我们也要检查是否应用了轴向修正。
            # 但如果你之前是用 check_data_quality_batch.py 生成的，可能没保存修正。
            # 为了保险起见，这里假设如果已经是 us/xs 格式，就不动它了。
            # 如果你之前的 .p 文件是错误的轴向，请手动删除所有 .p 文件重新生成！
            if 'us' in data and 'xs' in data:
                continue

            # ==================================================================
            # 情况 B: 你的原子格式 (w, a, q) -> 需要转换并【应用轴向修正】
            # ==================================================================
            elif 'w' in data and 'a' in data and 'q' in data:
                print(f"Converting Atomic format {f} to model format...")

                # 原始数据
                raw_w = data['w']  # (N, 3) [gx, gy, gz]
                raw_a = data['a']  # (N, 3) [ax, ay, az]

                # ----------------------------------------------------------
                # 🚨 核心修正：应用轴向映射 (根据诊断报告)
                # GT_Roll (X) <== +gryY
                # GT_Pitch (Y) <== -gryX
                # GT_Yaw (Z) <== +gryZ
                # ----------------------------------------------------------
                # 注意 numpy 的列索引: 0=x, 1=y, 2=z

                new_w = np.zeros_like(raw_w)
                new_w[:, 0] = raw_w[:, 1]  # New Gx = Old Gy
                new_w[:, 1] = -raw_w[:, 0]  # New Gy = -Old Gx
                new_w[:, 2] = raw_w[:, 2]  # New Gz = Old Gz

                # 加速度计通常也需要同样的旋转修正 (假设安装是刚体)
                new_a = np.zeros_like(raw_a)
                new_a[:, 0] = raw_a[:, 1]  # New Ax = Old Ay
                new_a[:, 1] = -raw_a[:, 0]  # New Ay = -Old Ax
                new_a[:, 2] = raw_a[:, 2]  # New Az = Old Az

                # 转为 Tensor
                gyro = torch.from_numpy(new_w).double()
                acc = torch.from_numpy(new_a).double()
                us = torch.cat([gyro, acc], dim=1)  # (N, 6)

                # ----------------------------------------------------------
                # 处理 GT (计算相对旋转 xs)
                # ----------------------------------------------------------
                # data['q'] 假设是 [qx, qy, qz, qw] (scipy格式)
                # SO3 库通常需要 [qw, qx, qy, qz]
                q_xyzw = torch.from_numpy(data['q']).double()
                q_wxyz = q_xyzw[:, [3, 0, 1, 2]]  # 调整顺序
                q_wxyz = SO3.qnorm(q_wxyz)

                # 生成旋转矩阵
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                R_all = SO3.from_quaternion(q_wxyz.to(device), ordering='wxyz')

                # 计算 xs (Label)
                mtf = self.min_train_freq
                if R_all.shape[0] <= mtf:
                    print(f"Skipping {f}, too short.")
                    continue

                R_i = R_all[:-mtf]
                R_j = R_all[mtf:]
                dRot = bmtm(R_i, R_j)
                dRot = SO3.dnormalize(dRot)
                xs = SO3.log(dRot).cpu().float()  # (N-mtf, 3)

                # ----------------------------------------------------------
                # 保存回 .p 文件
                # ----------------------------------------------------------
                # 注意：我们要截断 us 和 t 以匹配 xs 的长度 (通常 dataset getitem 会处理，
                # 但为了对其，最好保持 us 完整，getitem 里去切片)
                # 这里我们遵循原有的结构，保存完整的 us 和计算好的 xs

                data['us'] = us.float()  # 转回 float32 节省空间
                data['xs'] = xs
                # 移除原始的大数组以节省空间 (可选)
                # del data['w'], data['a'], data['q']

                pdump(data, self.predata_dir, f)
                print(f"✅ Converted and Fixed Axis for {f}")