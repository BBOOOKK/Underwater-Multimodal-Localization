import torch
import time
import matplotlib.pyplot as plt

# 设置绘图风格
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'x-large'
plt.rcParams['legend.fontsize'] = 'x-large'
plt.rcParams['xtick.labelsize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'

from termcolor import cprint
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.utils import pload, pdump, yload, ydump, mkdir, bmv
from src.utils import bmtm, bmtv, bmmt
from datetime import datetime
from src.lie_algebra import SO3, CPUSO3


class LearningBasedProcessing:
    def __init__(self, res_dir, tb_dir, net_class, net_params, address, dt):
        self.res_dir = res_dir
        self.tb_dir = tb_dir
        self.net_class = net_class
        self.net_params = net_params
        self._ready = False
        self.train_params = {}
        self.figsize = (20, 12)
        self.dt = dt  # (s)

        # [关键修改] 定义设备 (GPU or CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.address, self.tb_address = self.find_address(address)
        if address is None:  # create new address
            pdump(self.net_params, self.address, 'net_params.p')
            ydump(self.net_params, self.address, 'net_params.yaml')
        else:  # pick the network parameters
            self.net_params = pload(self.address, 'net_params.p')
            self.train_params = pload(self.address, 'train_params.p')
            self._ready = True
        self.path_weights = os.path.join(self.address, 'weights.pt')
        self.net = self.net_class(**self.net_params)

        # 将模型移动到对应设备
        self.net.to(self.device)

        if self._ready:  # fill network parameters
            self.load_weights()

    def find_address(self, address):
        """return path where net and training info are saved"""
        if address == 'last':
            # 检查目录是否存在，防止报错
            if not os.path.exists(self.res_dir):
                os.makedirs(self.res_dir)
            addresses = sorted(os.listdir(self.res_dir))
            if len(addresses) == 0:
                # 如果没有历史记录，强制新建
                return None, None
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
        # [关键修改] 使用 self.device 和 weights_only=True 以适应新版 torch 安全要求
        try:
            weights = torch.load(self.path_weights, map_location=self.device, weights_only=True)
        except:
            # 兼容旧版本 torch
            weights = torch.load(self.path_weights, map_location=self.device)

        self.net.load_state_dict(weights)
        self.net.to(self.device)

    def train(self, dataset_class, dataset_params, train_params):
        """train the neural network."""
        self.train_params = train_params
        pdump(self.train_params, self.address, 'train_params.p')
        ydump(self.train_params, self.address, 'train_params.yaml')

        hparams = self.get_hparams(dataset_class, dataset_params, train_params)
        ydump(hparams, self.address, 'hparams.yaml')

        # define datasets
        dataset_train = dataset_class(**dataset_params, mode='train')
        dataset_train.init_train()
        dataset_val = dataset_class(**dataset_params, mode='val')
        dataset_val.init_val()

        # get class
        Optimizer = train_params['optimizer_class']
        Scheduler = train_params['scheduler_class']
        Loss = train_params['loss_class']

        # get parameters
        dataloader_params = train_params['dataloader']
        optimizer_params = train_params['optimizer']
        scheduler_params = train_params['scheduler']
        loss_params = train_params['loss']

        # define optimizer, scheduler and loss
        dataloader = DataLoader(dataset_train, **dataloader_params)
        optimizer = Optimizer(self.net.parameters(), **optimizer_params)
        scheduler = Scheduler(optimizer, **scheduler_params)
        # [关键修改] 将 loss 移动到设备
        criterion = Loss(**loss_params).to(self.device)

        # remaining training parameters
        freq_val = train_params['freq_val']
        n_epochs = train_params['n_epochs']

        # init net w.r.t dataset
        self.net = self.net.to(self.device)
        mean_u, std_u = dataset_train.mean_u, dataset_train.std_u
        self.net.set_normalized_factors(mean_u, std_u)

        # start tensorboard writer
        writer = SummaryWriter(self.tb_address)
        start_time = time.time()
        best_loss = torch.Tensor([float('Inf')])

        # define some function for seeing evolution of training
        def write(epoch, loss_epoch):
            writer.add_scalar('loss/train', loss_epoch.item(), epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            print('Train Epoch: {:2d} \tLoss: {:.4f}'.format(
                epoch, loss_epoch.item()))
            scheduler.step(epoch)

        def write_time(epoch, start_time):
            delta_t = time.time() - start_time
            print("Amount of time spent for epochs " +
                  "{}-{}: {:.1f}s\n".format(epoch - freq_val, epoch, delta_t))
            writer.add_scalar('time_spend', delta_t, epoch)

        def write_val(loss, best_loss):
            if 0.5 * loss <= best_loss:
                msg = 'validation loss decreases! :) '
                msg += '(curr/prev loss {:.4f}/{:.4f})'.format(loss.item(),
                                                               best_loss.item())
                cprint(msg, 'green')
                best_loss = loss
                self.save_net()
            else:
                msg = 'validation loss increases! :( '
                msg += '(curr/prev loss {:.4f}/{:.4f})'.format(loss.item(),
                                                               best_loss.item())
                cprint(msg, 'yellow')
            writer.add_scalar('loss/val', loss.item(), epoch)
            return best_loss

        # training loop !
        for epoch in range(1, n_epochs + 1):
            loss_epoch = self.loop_train(dataloader, optimizer, criterion)
            write(epoch, loss_epoch)
            scheduler.step(epoch)
            if epoch % freq_val == 0:
                loss = self.loop_val(dataset_val, criterion)
                write_time(epoch, start_time)
                best_loss = write_val(loss, best_loss)
                start_time = time.time()
        # training is over !

        # test on new data
        dataset_test = dataset_class(**dataset_params, mode='test')
        self.load_weights()
        test_loss = self.loop_val(dataset_test, criterion)
        dict_loss = {
            'final_loss/val': best_loss.item(),
            'final_loss/test': test_loss.item()
        }
        writer.add_hparams(hparams, dict_loss)
        ydump(dict_loss, self.address, 'final_loss.yaml')
        writer.close()

    def loop_train(self, dataloader, optimizer, criterion):
        """Forward-backward loop over training data"""
        loss_epoch = 0
        optimizer.zero_grad()
        for us, xs in dataloader:
            # [关键修改] 数据移动到设备
            us = dataloader.dataset.add_noise(us.to(self.device))
            xs = xs.to(self.device)
            hat_xs = self.net(us)
            loss = criterion(xs, hat_xs) / len(dataloader)
            loss.backward()
            loss_epoch += loss.detach().cpu()
        optimizer.step()
        return loss_epoch

    def loop_val(self, dataset, criterion):
        """Forward loop over validation data"""
        loss_epoch = 0
        self.net.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                us, xs = dataset[i]
                # [关键修改] 数据移动到设备
                us = us.to(self.device).unsqueeze(0)
                xs = xs.to(self.device).unsqueeze(0)
                hat_xs = self.net(us)
                loss = criterion(xs, hat_xs) / len(dataset)
                loss_epoch += loss.cpu()
        self.net.train()
        return loss_epoch

    def save_net(self):
        """save the weights on the net in CPU"""
        self.net.eval().cpu()
        torch.save(self.net.state_dict(), self.path_weights)
        self.net.train().to(self.device)

    def get_hparams(self, dataset_class, dataset_params, train_params):
        """return all training hyperparameters in a dict"""
        Optimizer = train_params['optimizer_class']
        Scheduler = train_params['scheduler_class']
        Loss = train_params['loss_class']

        # get training class parameters
        dataloader_params = train_params['dataloader']
        optimizer_params = train_params['optimizer']
        scheduler_params = train_params['scheduler']
        loss_params = train_params['loss']

        # remaining training parameters
        freq_val = train_params['freq_val']
        n_epochs = train_params['n_epochs']

        dict_class = {
            'Optimizer': str(Optimizer),
            'Scheduler': str(Scheduler),
            'Loss': str(Loss)
        }

        return {**dict_class, **dataloader_params, **optimizer_params,
                **loss_params, **scheduler_params,
                'n_epochs': n_epochs, 'freq_val': freq_val}

    def test(self, dataset_class, dataset_params, modes):
        """test a network once training is over"""

        # get loss function
        Loss = self.train_params['loss_class']
        loss_params = self.train_params['loss']
        criterion = Loss(**loss_params).to(self.device)

        # test on each type of sequence
        for mode in modes:
            dataset = dataset_class(**dataset_params, mode=mode)
            self.loop_test(dataset, criterion)
            self.display_test(dataset, mode)

    def loop_test(self, dataset, criterion):
        """Forward loop over test data"""
        self.net.eval()
        for i in range(len(dataset)):
            seq = dataset.sequences[i]
            us, xs = dataset[i]
            with torch.no_grad():
                # [关键修改] 移动到设备
                hat_xs = self.net(us.to(self.device).unsqueeze(0))
            loss = criterion(xs.to(self.device).unsqueeze(0), hat_xs)
            mkdir(self.address, seq)
            mondict = {
                'hat_xs': hat_xs[0].cpu(),
                'loss': loss.cpu().item(),
            }
            pdump(mondict, self.address, seq, 'results.p')

    def display_test(self, dataset, mode):
        raise NotImplementedError


class GyroLearningBasedProcessing(LearningBasedProcessing):
    def __init__(self, res_dir, tb_dir, net_class, net_params, address, dt):
        super().__init__(res_dir, tb_dir, net_class, net_params, address, dt)
        self.roe_dist = [7, 14, 21, 28, 35]  # m
        self.freq = 100  # subsampling frequency for RTE computation
        self.roes = {  # relative trajectory errors
            'Rots': [],
            'yaws': [],
        }

    def display_test(self, dataset, mode):
        self.roes = {
            'Rots': [],
            'yaws': [],
        }
        # self.to_open_vins(dataset)
        from scipy.spatial.transform import Rotation as R

        for i, seq in enumerate(dataset.sequences):
            print('\n', 'Results for sequence ' + seq)
            self.seq = seq

            # 读取数据
            data_dict = dataset.load_seq(i)
            # 加载真值文件
            gt_dict = dataset.load_gt(i)

            self.gt = {}

            # ======================================================
            # [核心修复] 兼容性处理：优先读取四元数，转换为旋转矩阵
            # ======================================================
            if 'qs' in gt_dict:
                # 新格式：从 gt 文件中读取 qs (N, 4)
                qs_gt = gt_dict['qs']
                if qs_gt.is_cuda: qs_gt = qs_gt.cpu()

                # 转为旋转矩阵
                # 注意：generate_data_complete.py 中我们保存的是 wxyz 顺序
                self.gt['Rots'] = SO3.from_quaternion(qs_gt, ordering='wxyz').cpu()
                self.gt['qs'] = qs_gt
            elif 'rot' in data_dict:
                # 旧格式：直接读取 rot
                self.gt['Rots'] = data_dict['rot'].cpu()
                # 从矩阵转四元数
                r = R.from_matrix(self.gt['Rots'].numpy())
                self.gt['qs'] = torch.from_numpy(r.as_quat()).float()
            else:
                print(f"Error: No rotation ground truth found for {seq}")
                continue
            # ======================================================

            self.gt['rpys'] = SO3.to_rpy(self.gt['Rots']).cpu()

            # 读取网络预测结果
            self.net_us = pload(self.address, seq, 'results.p')['hat_xs']
            self.raw_us, _ = dataset[i]
            N = self.net_us.shape[0]

            # 计算陀螺仪修正量
            self.gyro_corrections = (self.raw_us[:, :3] - self.net_us[:N, :3])
            self.ts = torch.linspace(0, N * self.dt, N)

            self.convert()
            self.plot_gyro()
            self.plot_gyro_correction()
            plt.show()  # 显示图片

    def to_open_vins(self, dataset):
        pass

    def convert(self):
        # s -> min
        l = 1 / 60
        self.ts *= l

        # rad -> deg
        l = 180 / np.pi
        self.gyro_corrections *= l
        self.gt['rpys'] *= l

    def integrate_with_quaternions_superfast(self, N, raw_us, net_us):
        # [关键修改] 移动到设备 (.to(self.device))
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
        N = self.raw_us.shape[0]
        raw_us = self.raw_us[:, :3]
        net_us = self.net_us[:, :3]

        net_qs, imu_Rots, net_Rots = self.integrate_with_quaternions_superfast(N,
                                                                               raw_us, net_us)
        imu_rpys = 180 / np.pi * SO3.to_rpy(imu_Rots).cpu()
        net_rpys = 180 / np.pi * SO3.to_rpy(net_Rots).cpu()
        self.plot_orientation(imu_rpys, net_rpys, N)
        self.plot_orientation_error(imu_Rots, net_Rots, N)

    def plot_orientation(self, imu_rpys, net_rpys, N):
        title = "Orientation estimation"
        # 获取真值
        gt = self.gt['rpys'][:N]

        # 处理真值 (GT)
        gt_rad = np.deg2rad(gt.numpy() if isinstance(gt, torch.Tensor) else gt)
        gt_unwrapped = np.rad2deg(np.unwrap(gt_rad, axis=0))

        imu_rad = np.deg2rad(imu_rpys.numpy() if isinstance(imu_rpys, torch.Tensor) else imu_rpys)
        imu_unwrapped = np.rad2deg(np.unwrap(imu_rad, axis=0))

        net_rad = np.deg2rad(net_rpys.numpy() if isinstance(net_rpys, torch.Tensor) else net_rpys)
        net_unwrapped = np.rad2deg(np.unwrap(net_rad, axis=0))
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        for i in range(3):
            # 注意：这里改用 unwrapped 后的数据进行绘图
            axs[i].plot(self.ts, gt_unwrapped[:, i], color='black', label=r'ground truth')
            axs[i].plot(self.ts, imu_unwrapped[:, i], color='red', label=r'raw IMU')
            axs[i].plot(self.ts, net_unwrapped[:, i], color='blue', label=r'net IMU')
            axs[i].set_xlim(self.ts[0], self.ts[-1])

        self.savefig(axs, fig, 'orientation')

    def plot_orientation_error(self, imu_Rots, net_Rots, N):
        # [关键修改] 移动到设备
        gt = self.gt['Rots'][:N].to(self.device)

        raw_err = 180 / np.pi * SO3.log(bmtm(imu_Rots.to(self.device), gt)).cpu()
        net_err = 180 / np.pi * SO3.log(bmtm(net_Rots.to(self.device), gt)).cpu()
        title = "$SO(3)$ orientation error"
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        for i in range(3):
            axs[i].plot(self.ts, raw_err[:, i], color='red', label=r'raw IMU')
            axs[i].plot(self.ts, net_err[:, i], color='blue', label=r'net IMU')
            # axs[i].set_ylim(-10, 10)
            axs[i].set_xlim(self.ts[0], self.ts[-1])
        self.savefig(axs, fig, 'orientation_error')

    def plot_gyro_correction(self):
        title = "Gyro correction" + self.end_title
        ylabel = 'gyro correction (deg/s)'
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set(xlabel='$t$ (min)', ylabel=ylabel, title=title)
        plt.plot(self.ts, self.gyro_corrections, label=r'net IMU')
        ax.set_xlim(self.ts[0], self.ts[-1])
        self.savefig(ax, fig, 'gyro_correction')

    @property
    def end_title(self):
        return " for sequence " + self.seq.replace("_", " ")

    def savefig(self, axs, fig, name):
        # 确保保存目录存在
        save_dir = os.path.join(self.address, self.seq)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if isinstance(axs, np.ndarray):
            for i in range(len(axs)):
                axs[i].grid()
                axs[i].legend()
        else:
            axs.grid()
            axs.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, name + '.png'))