import torch
import time
import matplotlib
import sys

# 设置后端防止无GUI报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from termcolor import cprint
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入 scipy 用于快速积分
from scipy.spatial.transform import Rotation as R

# 🔥 必须加上 src. 前缀
from src.utils import pload, pdump, yload, ydump, mkdir, bmv
from src.utils import bmtm, bmtv, bmmt
from src.utils import SO3, CPUSO3, unwrap_rpy
from datetime import datetime

# Matplotlib 设置
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'x-large'
plt.rcParams['legend.fontsize'] = 'x-large'
plt.rcParams['xtick.labelsize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'


class LearningBasedProcessing:
    def __init__(self, res_dir, tb_dir, net_class, net_params, address, dt):
        self.res_dir = res_dir
        self.tb_dir = tb_dir
        self.net_class = net_class
        self.net_params = net_params
        self._ready = False
        self.train_params = {}
        self.figsize = (20, 12)
        self.dt = dt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 解析地址
        raw_address, self.tb_address = self.find_address(address)

        if raw_address and os.path.isfile(raw_address):
            self.address = os.path.dirname(raw_address)
            self.path_weights = raw_address
        else:
            self.address = raw_address
            self.path_weights = os.path.join(self.address, 'weights.pt')

        # 加载配置
        if address is None:
            pdump(self.net_params, self.address, 'net_params.p')
            ydump(self.net_params, self.address, 'net_params.yaml')
        else:
            try:
                self.net_params = pload(self.address, 'net_params.p')
                self.train_params = pload(self.address, 'train_params.p')
                self._ready = True
            except:
                pass  # 忽略加载错误

        self.net = self.net_class(**self.net_params).to(self.device)

        if self._ready:
            self.load_weights()

    def find_address(self, address):
        if address == 'last':
            if not os.path.exists(self.res_dir): os.makedirs(self.res_dir)
            addresses = sorted(os.listdir(self.res_dir))
            if len(addresses) == 0: return None, None
            tb_address = os.path.join(self.tb_dir, str(len(addresses)))
            address = os.path.join(self.res_dir, addresses[-1])
        elif address is None:
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            address = os.path.join(self.res_dir, now)
            mkdir(address)
            tb_address = os.path.join(self.tb_dir, now)
        else:
            tb_address = None
        return address, tb_address

    def load_weights(self):
        if os.path.exists(self.path_weights):
            try:
                print(f"Loading weights from: {self.path_weights}")
                weights = torch.load(self.path_weights, map_location=self.device, weights_only=True)
            except:
                weights = torch.load(self.path_weights, map_location=self.device)
            self.net.load_state_dict(weights)

    def train(self, dataset_class, dataset_params, train_params):
        self.train_params = train_params

        # 🔥 [关键修复 1] 清洗参数，防止 lambda 函数导致 Pickle 报错
        params_to_save = train_params.copy()
        # 将无法序列化的对象转为字符串
        for k, v in params_to_save.items():
            if callable(v):
                params_to_save[k] = str(v)

        pdump(params_to_save, self.address, 'train_params.p')

        dataset_train = dataset_class(**dataset_params, mode='train')
        dataset_train.init_train()
        dataset_val = dataset_class(**dataset_params, mode='val')
        dataset_val.init_val()

        dl_params = train_params['dataloader'].copy()
        dl_params['drop_last'] = True

        dataloader = DataLoader(dataset_train, **dl_params)
        optimizer = train_params['optimizer_class'](self.net.parameters(), **train_params['optimizer'])
        scheduler = train_params['scheduler_class'](optimizer, **train_params['scheduler'])

        # 实例化 Loss
        criterion = train_params['loss_class'](**train_params['loss']).to(self.device)

        # 🔥 [关键修复 2] 安全调用 set_normalized_factors
        # 只有当网络支持该方法时才调用 (MoEGyroNet 和 GyroNet 现在都支持了)
        if hasattr(self.net, 'set_normalized_factors'):
            self.net.set_normalized_factors(dataset_train.mean_u.to(self.device), dataset_train.std_u.to(self.device))

        writer = SummaryWriter(self.tb_address)
        best_loss = float('Inf')

        print(f"\n🚀 Start Training on {self.device}")

        use_adv = train_params.get('adv_training', False)
        if use_adv:
            print(f"🛡️ Adversarial Training Enabled (eps={train_params.get('adv_eps', 0.01)})")

        print("=" * 90)
        print(f"{'Epoch':^5} | {'Train Loss':^12} | {'Val Loss':^12} | {'Best Val':^12} | {'LR':^10} | {'Status':^10}")
        print("-" * 90)

        start_time = time.time()

        for epoch in range(1, train_params['n_epochs'] + 1):
            loss_train = self.loop_train(dataloader, optimizer, criterion, epoch)
            loss_train_val = loss_train.item() if isinstance(loss_train, torch.Tensor) else loss_train

            loss_val_val = float('nan')
            status = ""

            if epoch % train_params['freq_val'] == 0:
                loss_val = self.loop_val(dataset_val, criterion)
                loss_val_val = loss_val.item() if isinstance(loss_val, torch.Tensor) else loss_val

                if loss_val_val <= best_loss:
                    best_loss = loss_val_val
                    self.save_net()
                    status = "★ SAVED"
                else:
                    status = ""

                writer.add_scalar('loss/val', loss_val_val, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            val_str = f"{loss_val_val:.4f}" if not np.isnan(loss_val_val) else "-"
            best_str = f"{best_loss:.4f}" if best_loss != float('Inf') else "-"

            print(
                f"{epoch:^5d} | {loss_train_val:^12.4f} | {val_str:^12} | {best_str:^12} | {current_lr:^10.2e} | {status}")

            writer.add_scalar('loss/train', loss_train_val, epoch)
            scheduler.step()

        writer.close()
        total_time = time.time() - start_time
        print("=" * 90)
        print(f"🏁 Training Finished in {total_time / 60:.1f} minutes.")

    # =========================================================================
    # 🔥 [增强版] 训练循环：兼容 Tuple 输出 (MoE)
    # =========================================================================
    def loop_train(self, dataloader, optimizer, criterion, epoch):
        loss_epoch = 0
        self.net.train()

        use_adv = self.train_params.get('adv_training', False)
        adv_eps = self.train_params.get('adv_eps', 0.01)

        pbar = tqdm(dataloader, desc=f"Ep {epoch}", leave=False, unit="batch")

        for i, data in enumerate(pbar):
            us = data['u']
            xs = data['x']

            if hasattr(dataloader.dataset, 'add_noise'):
                us = dataloader.dataset.add_noise(us)

            us = us.to(self.device)
            xs = xs.to(self.device)

            if us.abs().max().item() > 50.0:
                self.diagnose_explosion(us, f"Normalized Input > 50", i)

            # --- 1. Clean Pass ---
            if use_adv:
                us.requires_grad_(True)
                us.retain_grad()

            optimizer.zero_grad()
            hat_xs = self.net(us)

            # 🔥 MoE 兼容：Criterion (MoELoss) 会处理 Tuple，这里直接传
            loss = criterion(xs, hat_xs)

            if not isinstance(loss, torch.Tensor):
                continue

            # --- 2. Adversarial Pass ---
            if use_adv:
                loss.backward(retain_graph=True)

                if us.grad is not None:
                    data_grad = us.grad.data
                    us_adv = us + adv_eps * data_grad.sign()

                    optimizer.zero_grad()
                    hat_xs_adv = self.net(us_adv)
                    loss_adv = criterion(xs, hat_xs_adv)

                    if isinstance(loss_adv, torch.Tensor):
                        loss_adv.backward()
                        curr_loss = (loss.item() + loss_adv.item()) / 2
                    else:
                        curr_loss = loss.item()
                else:
                    curr_loss = loss.item()
            else:
                loss.backward()
                curr_loss = loss.detach().cpu().item()

            if isinstance(loss, torch.Tensor) and (torch.isnan(loss) or loss.item() > 1e6):
                self.diagnose_explosion(us, f"Loss Exploded", i)

            optimizer.step()

            loss_epoch += curr_loss
            pbar.set_postfix({'loss': f'{curr_loss:.4f}'})

        return loss_epoch / len(dataloader)

    def diagnose_explosion(self, us, reason, batch_idx):
        print("\n" + "!" * 80)
        cprint(f"🚨 [CRITICAL STOP] Training Aborted!", 'red', attrs=['bold', 'blink'])
        cprint(f"👉 Reason: {reason}", 'yellow', attrs=['bold'])
        print(f"👉 Batch Index: {batch_idx}")
        print("-" * 80)
        raise RuntimeError("Training aborted.")

    def loop_val(self, dataset, criterion):
        loss_epoch = 0
        self.net.eval()
        val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for data in val_loader:
                us = data['u'].to(self.device)
                xs = data['x'].to(self.device)
                hat_xs = self.net(us)
                loss = criterion(xs, hat_xs)
                if isinstance(loss, torch.Tensor):
                    loss_epoch += loss.cpu().item()
        return loss_epoch / len(val_loader)

    def save_net(self, name=None):
        if name is None:
            save_path = self.path_weights
        else:
            save_path = os.path.join(self.address, name)
        torch.save(self.net.state_dict(), save_path)

    def test(self, dataset_class, dataset_params, modes):
        for mode in modes:
            dataset = dataset_class(**dataset_params, mode=mode)
            self.loop_test(dataset)
            self.display_test(dataset, mode)

    # =========================================================================
    # 🔥 [增强版] 测试循环：自动处理 MoE 的元组输出
    # =========================================================================
    def loop_test(self, dataset):
        mc_samples = self.train_params.get('mc_samples', 1)

        if mc_samples > 1:
            print(f"🎲 MC-Dropout Enabled: Sampling {mc_samples} times per sequence...")
            self.net.train()
        else:
            self.net.eval()

        for i in tqdm(range(len(dataset)), desc="Testing"):
            seq = dataset.sequences[i]
            data = dataset[i]
            us = data['u']

            if mc_samples > 1:
                hat_xs_sum = 0
                with torch.no_grad():
                    us_in = us.unsqueeze(0).to(self.device)
                    for _ in range(mc_samples):
                        out = self.net(us_in)
                        # 🔥 [关键修复 3] 如果是 Tuple (MoE)，只取第一个元素(预测值)
                        if isinstance(out, tuple): out = out[0]
                        hat_xs_sum += out
                hat_xs = hat_xs_sum / mc_samples
            else:
                with torch.no_grad():
                    us_in = us.unsqueeze(0).to(self.device)
                    hat_xs = self.net(us_in)
                    # 🔥 [关键修复 4] 同上，提取预测结果
                    if isinstance(hat_xs, tuple):
                        hat_xs = hat_xs[0]

            mkdir(self.address, seq)
            # 此时 hat_xs 已经是 (1, Seq, 3) 的 Tensor 了
            hat_vel = hat_xs[0]

            pdump({'hat_xs': hat_vel.cpu()}, self.address, seq, 'results.p')

    def display_test(self, dataset, mode):
        raise NotImplementedError


class GyroLearningBasedProcessing(LearningBasedProcessing):
    def __init__(self, res_dir, tb_dir, net_class, net_params, address, dt):
        super().__init__(res_dir, tb_dir, net_class, net_params, address, dt)
        self.min_train_freq = 16.0

    def display_test(self, dataset, mode):
        if hasattr(dataset, 'min_train_freq'):
            self.min_train_freq = float(dataset.min_train_freq)

        print(f"\n[Generating Plots for {len(dataset.sequences)} sequences...]")
        for i, seq in enumerate(tqdm(dataset.sequences, desc="Plotting")):
            self.seq = seq

            data_dict = dataset.load_seq(i)
            gt_dict = dataset.load_gt(i)
            self.raw_us = data_dict['us']

            self.gt = {
                'ps': gt_dict['ps'],
                'qs': gt_dict['qs'],
            }

            self.net_us = pload(self.address, seq, 'results.p')['hat_xs']

            N = self.net_us.shape[0]
            self.ts = torch.linspace(0, N * self.dt, N).numpy() / 60.0

            raw_us_np = self.raw_us
            net_us_np = self.net_us
            if isinstance(raw_us_np, torch.Tensor): raw_us_np = raw_us_np.cpu().numpy()
            if isinstance(net_us_np, torch.Tensor): net_us_np = net_us_np.cpu().numpy()

            self.gyro_corrections = (raw_us_np[:N, :3] - net_us_np[:N, :3]) * 180 / np.pi

            self.plot_gyro()
            self.plot_gyro_correction()

            plt.close('all')

    def integrate_with_quaternions_superfast(self, N, raw_us, net_us):
        imu_qs = SO3.qnorm(SO3.qexp(raw_us[:, :3].to(self.device).double() * self.dt))
        net_qs = SO3.qnorm(SO3.qexp(net_us[:, :3].to(self.device).double() * self.dt))

        Rot0 = SO3.qnorm(self.gt['qs'][:2].to(self.device).double())

        imu_qs[0] = Rot0[0]
        net_qs[0] = Rot0[0]

        N_log = np.log2(imu_qs.shape[0])
        for i in range(int(N_log)):
            k = 2 ** i
            imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:-k], imu_qs[k:]))
            net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:-k], net_qs[k:]))

        if int(N_log) < N_log:
            k = 2 ** int(N_log)
            k2 = imu_qs[k:].shape[0]
            imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:k2], imu_qs[k:]))
            net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:k2], net_qs[k:]))

        imu_Rots = SO3.from_quaternion(imu_qs).float()
        net_Rots = SO3.from_quaternion(net_qs).float()
        return net_qs.cpu(), imu_Rots, net_Rots

    def plot_gyro(self):
        N = self.net_us.shape[0]
        raw_us = self.raw_us[:N, :3]
        net_us = self.net_us[:N, :3]

        net_qs, imu_Rots, net_Rots = self.integrate_with_quaternions_superfast(N, raw_us, net_us)

        imu_rpys = 180 / np.pi * SO3.to_rpy(imu_Rots).cpu()
        net_rpys = 180 / np.pi * SO3.to_rpy(net_Rots).cpu()

        gt_qs_wxyz = self.gt['qs'][:N]
        gt_Rots = SO3.from_quaternion(gt_qs_wxyz)
        gt_rpys = 180 / np.pi * SO3.to_rpy(gt_Rots)

        self.plot_orientation(gt_rpys, imu_rpys, net_rpys)
        self.plot_orientation_error(imu_Rots, net_Rots, N)

    def plot_orientation(self, gt_s, imu_s, net_s):
        step = 10
        ts = self.ts[::step]

        if isinstance(gt_s, torch.Tensor): gt_s = gt_s.numpy()
        if isinstance(imu_s, torch.Tensor): imu_s = imu_s.numpy()
        if isinstance(net_s, torch.Tensor): net_s = net_s.numpy()

        gt_s = np.unwrap(gt_s * np.pi / 180, axis=0) * 180 / np.pi
        imu_s = np.unwrap(imu_s * np.pi / 180, axis=0) * 180 / np.pi
        net_s = np.unwrap(net_s * np.pi / 180, axis=0) * 180 / np.pi

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        labels = ['roll (deg)', 'pitch (deg)', 'yaw (deg)']

        for i in range(3):
            axs[i].plot(ts, gt_s[::step, i], 'k-', label='Ground Truth', linewidth=2)
            axs[i].plot(ts, imu_s[::step, i], 'r--', label='Raw IMU', alpha=0.7)
            axs[i].plot(ts, net_s[::step, i], 'b-', label='Net Denoised', linewidth=1.5)
            axs[i].set_ylabel(labels[i])

        axs[2].set_xlabel('$t$ (min)')
        axs[0].set_title(f"Orientation Estimation: {self.seq}")
        self.savefig(axs, fig, 'orientation')

    def plot_orientation_error(self, imu_Rots, net_Rots, N):
        step = 10
        ts = self.ts[::step]

        gt = self.gt['qs'][:N]
        gt_Rots = SO3.from_quaternion(gt).to(self.device)

        raw_err = 180 / np.pi * SO3.log(bmtm(imu_Rots.to(self.device), gt_Rots)).cpu()
        net_err = 180 / np.pi * SO3.log(bmtm(net_Rots.to(self.device), gt_Rots)).cpu()

        raw_err = raw_err.numpy()
        net_err = net_err.numpy()

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        labels = ['roll err', 'pitch err', 'yaw err']
        for i in range(3):
            axs[i].plot(ts, raw_err[::step, i], 'r', label='Raw IMU Error', alpha=0.6)
            axs[i].plot(ts, net_err[::step, i], 'b', label='Net Denoised Error', alpha=0.8)
            axs[i].set_ylabel(labels[i])

        axs[2].set_xlabel('$t$ (min)')
        axs[0].set_title(f"Rotation Error ($SO3$): {self.seq}")
        self.savefig(axs, fig, 'orientation_error')

    def plot_gyro_correction(self):
        step = 10
        ts = self.ts[::step]
        data = self.gyro_corrections[::step]

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(ts, data[:, 0], label='X-Correction')
        ax.plot(ts, data[:, 1], label='Y-Correction')
        ax.plot(ts, data[:, 2], label='Z-Correction')
        ax.set(xlabel='$t$ (min)', ylabel='deg/s', title=f"Gyro Correction: {self.seq}")
        self.savefig(ax, fig, 'gyro_correction')

    def savefig(self, axs, fig, name):
        save_dir = os.path.join(self.address, self.seq)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        axes_list = axs if isinstance(axs, np.ndarray) else [axs]
        for ax in axes_list:
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if labels: ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, name + '.png'))