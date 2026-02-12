"""
* This file is part of RNIN-VIO
* (Single-GPU Version)
"""

import numpy as np
from tqdm import tqdm
import json
import os
from os import path as osp
import torch
from model import function
from torch.utils.data import DataLoader
import logging
from utils.metric import compute_ate_rte
from utils import postprocess
from dataloader import dataset as dataset_utils

def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()

class tester(object):
    def __init__(self, args, cfg, model):
        super(tester, self).__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.out_dir = cfg['test']['out_dir']

        self.pred_velocity = cfg['model']['pred_velocity']
        self.window_time = cfg['model_param']['window_time']
        self.start_cov_epochs = cfg['train']['start_cov_epochs']
        self.plot_cnt = 0

    def inference_step(self, data_loader, epoch):
        targets_all, preds_all, preds_cov_all, losses_all = [], [], [], []

        with torch.no_grad():
            for bid, batch in tqdm(enumerate(data_loader)):
                batch = [t.to(self.device) for t in batch]

                pred, pred_cov, targ, loss = \
                    function.fun_test_forward(self.cfg, self.model, batch, self.start_cov_epochs, epoch)

                targets_all.append(torch_to_numpy(targ))
                preds_all.append(torch_to_numpy(pred))
                preds_cov_all.append(torch_to_numpy(pred_cov))
                losses_all.append(np.mean(torch_to_numpy(loss)))

        if len(targets_all) == 0:
            logging.warning("【警告】inference_step: 未产生任何数据。")
            return None

        targets_all = np.concatenate(targets_all, axis=0)
        preds_all = np.concatenate(preds_all, axis=0)
        preds_cov_all = np.concatenate(preds_cov_all, axis=0)

        attr_dict = {
            "targets": targets_all,
            "preds": preds_all,
            "preds_cov": preds_cov_all,
            "losses": losses_all,
        }

        return attr_dict

    def test(self, test_data_path):
        ate_all, t_rte_all, d_rte_all = [], [], []
        all_metrics = {}

        if not osp.exists(self.out_dir):
            os.makedirs(self.out_dir)

        for data in test_data_path:
            logging.info(f"Processing {data}...")

            try:
                test_basic_data = dataset_utils.BasicSequenceData(self.cfg, [data], mode="test")
                index_map = test_basic_data.get_merged_index_map()
                test_dataset = dataset_utils.SeqToSeqDataset(self.cfg, test_basic_data, index_map, mode="test")
            except Exception as e:
                logging.warning(f"加载数据集 {data} 失败: {e}，跳过。")
                continue

            if len(test_dataset) == 0:
                logging.warning(f"【跳过】数据集 {data} 有效样本数为 0。")
                continue

            test_loader = DataLoader(test_dataset, batch_size=self.cfg['test']['batch_size'], shuffle=False)

            data_name = os.path.basename(os.path.normpath(data))
            outdir = osp.join(self.out_dir, data_name)
            if osp.exists(outdir) is False:
                os.makedirs(outdir)

            net_attr_dict = self.inference_step(test_loader, epoch=1000)

            if net_attr_dict is None:
                logging.warning(f"【跳过】数据集 {data_name} 推理结果为空。")
                continue

            try:
                traj_attr_dict = postprocess.pose_integrate(self.cfg, test_dataset, net_attr_dict["preds"])
            except Exception as e:
                logging.error(f"姿态积分失败 {data_name}: {e}")
                continue

            outfile = osp.join(outdir, "trajectory.txt")

            pos_pred = traj_attr_dict["pos_pred"]
            pos_gt = traj_attr_dict["pos_gt"]

            min_dim = min(pos_pred.shape[1], pos_gt.shape[1])
            pos_pred = pos_pred[:, :min_dim]
            pos_gt = pos_gt[:, :min_dim]

            trajectory_data = np.concatenate(
                [traj_attr_dict["ts"].reshape(-1, 1), pos_pred, pos_gt], axis=1
            )
            np.savetxt(outfile, trajectory_data, delimiter=",")

            plot_dict = postprocess.compute_plot_dict(
                self.cfg['data']['sample_freq'], net_attr_dict, traj_attr_dict
            )
            outfile_net = osp.join(outdir, "net_outputs.txt")

            out_list = [plot_dict["pred_ts"].reshape(-1, 1), plot_dict["preds"], plot_dict["targets"]]
            if plot_dict["pred_sigmas"] is not None and plot_dict["pred_sigmas"].size > 0:
                out_list.append(plot_dict["pred_sigmas"])

            net_outputs_data = np.concatenate(out_list, axis=1)
            np.savetxt(outfile_net, net_outputs_data, delimiter=",")

            try:
                postprocess.make_plots(plot_dict, outdir)
            except Exception as e:
                logging.warning(f"绘图跳过: {e}")

            fps = self.cfg['data'].get('sample_freq', 20.0)
            seq_len_frames = traj_attr_dict["pos_pred"].shape[0]
            rte_window = int(fps * 60)

            if seq_len_frames < rte_window:
                logging.warning(f"数据 {data_name} 太短，仅计算 ATE。")
                ate = compute_ate_rte(traj_attr_dict["pos_pred"], traj_attr_dict["pos_gt"], 0)[0]
                t_rte, d_rte = 0.0, 0.0
            else:
                ate, t_rte, d_rte = compute_ate_rte(traj_attr_dict["pos_pred"], traj_attr_dict["pos_gt"], rte_window)

            all_metrics[data_name] = {"ate": ate, "t_rte": t_rte, "d_rte": d_rte}
            logging.info(f"data {data_name}, ate: {ate:.4f}, t_rte {t_rte:.4f}, d_rte {d_rte:.4f}")

            ate_all.append(ate)
            t_rte_all.append(t_rte)
            d_rte_all.append(d_rte)

        if len(ate_all) > 0:
            all_metrics["summary"] = {
                "avg_ate": float(np.mean(ate_all)),
                "avg_t_rte": float(np.mean(t_rte_all)),
                "avg_d_rte": float(np.mean(d_rte_all)),
            }
            print('----------\navg ATE:{:.4f}, avg T_RTE:{:.4f}, avg D_RTE:{:.4f}'.format(
                np.mean(ate_all), np.mean(t_rte_all), np.mean(d_rte_all)))
        else:
            print("----------\n提示：本次运行没有产生任何有效测试结果。")

        with open(osp.join(self.out_dir, "metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=4)