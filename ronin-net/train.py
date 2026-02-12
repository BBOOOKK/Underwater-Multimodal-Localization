"""
* This file is part of RNIN-VIO
* (Single-GPU Version)
"""

import os
import time
import matplotlib.pyplot as plt
from os import path as osp
from tqdm import tqdm
import numpy as np
import torch
from model import function
from torch.utils.tensorboard import SummaryWriter
import logging

def torch_to_numpy(torch_data):
    return torch_data.cpu().detach().numpy()

def write_summary(summary_writer, attr_dict, epoch, optimizer, mode):
    if attr_dict is None:
        return

    mse_loss = np.mean((attr_dict["targets"] - attr_dict["preds"]) ** 2, axis=0)
    ml_loss = np.average(attr_dict["losses"])
    summary_writer.add_scalar(f"{mode}_loss/avg", np.mean(mse_loss), epoch)
    summary_writer.add_scalar(f"{mode}_dist/loss_full", ml_loss, epoch)
    if epoch > 0:
        summary_writer.add_scalar(
            "optimizer/lr", optimizer.param_groups[0]["lr"], epoch - 1
        )
    logging.info(f"{mode}: average ml loss: {ml_loss}")

def save_model(path, epoch, network, optimizer, use_multi_gpu=False):
    model_path = osp.join(path, "checkpoints", "checkpoint_%d.pt" % epoch)
    if not osp.isdir(osp.join(path, "checkpoints")):
        os.makedirs(osp.join(path, "checkpoints"))

    state_dict = {
        "model_state_dict": network.state_dict(),
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state_dict, model_path)
    logging.info(f"Model saved to {model_path}")

class trainer(object):
    def __init__(self, args, cfg, model, optimizer, start_epoch=0):
        super(trainer, self).__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.optimizer = optimizer

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=cfg['train']['scheduler']['factor'],
            patience=cfg['train']['scheduler']['patience'], eps=1e-12
        )
        logging.info(f"Optimizer: {self.optimizer}, Scheduler: {self.scheduler}")

        self.out_dir = cfg['train']['out_dir']
        self.summary_writer = SummaryWriter(os.path.join(self.out_dir, "logs"))

        self.start_epoch = start_epoch
        self.use_multi_gpu = False # Force False

        self.predict_start = cfg['train']['predict_start']
        self.predict_end = cfg['train']['predict_end']

        self.window_time = cfg['model_param']['window_time']
        self.epochs = cfg['train']['epochs']
        self.start_cov_epochs = cfg['train']['start_cov_epochs']
        self.pred_velocity = cfg['model']['pred_velocity']

    def inference_step(self, data_loader, epoch):
        targets_all, preds_all, preds_cov_all, losses_all = [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for bid, batch in tqdm(enumerate(data_loader)):
                batch = [t.to(self.device) for t in batch]

                pred, pred_cov, targ, loss = \
                    function.fun_train_forward(self.cfg, self.model, batch, self.start_cov_epochs, epoch)

                targets_all.append(torch_to_numpy(targ))
                preds_all.append(torch_to_numpy(pred))
                preds_cov_all.append(torch_to_numpy(pred_cov))
                losses_all.append(np.mean(torch_to_numpy(loss)))

        if len(targets_all) == 0:
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

    def train_step(self, data_loader, epoch):
        train_targets, train_preds, train_preds_cov, train_losses = [], [], [], []
        self.model.train()

        for bid, batch in tqdm(enumerate(data_loader)):
            batch = [t.to(self.device) for t in batch]
            self.optimizer.zero_grad()

            pred, pred_cov, targ, loss = \
                function.fun_train_forward(self.cfg, self.model, batch, self.start_cov_epochs, epoch)

            train_targets.append(torch_to_numpy(targ))
            train_preds.append(torch_to_numpy(pred))
            train_preds_cov.append(torch_to_numpy(pred_cov))
            train_losses.append(np.mean(torch_to_numpy(loss)))

            loss.backward()
            self.optimizer.step()

        if len(train_targets) == 0:
            return None

        train_targets = np.concatenate(train_targets, axis=0)
        train_preds = np.concatenate(train_preds, axis=0)
        train_preds_cov = np.concatenate(train_preds_cov, axis=0)
        train_attr_dict = {
            "targets": train_targets,
            "preds": train_preds,
            "preds_cov": train_preds_cov,
            "losses": train_losses,
        }
        return train_attr_dict

    def train(self, train_loader, val_loader=None, test_loader=None):
        best_val_loss = np.inf
        best_train_loss = np.inf
        val_loss, val_mse, test_loss = [], [], []
        train_loss, train_mse, test_mse = [], [], []

        for epoch in range(self.start_epoch + 1, self.epochs):
            logging.info(f"-------------- Training, Epoch {epoch} ---------------")
            start_t = time.time()

            train_attr_dict = self.train_step(train_loader, epoch)

            if train_attr_dict is None:
                logging.warning("Training set is empty for this epoch!")
                continue

            write_summary(self.summary_writer, train_attr_dict, epoch, self.optimizer, "train")
            end_t = time.time()
            logging.info(f"time usage: {end_t - start_t:.3f}s, lr {self.optimizer.param_groups[0]['lr']}")

            train_loss.append(np.average(train_attr_dict["losses"]))
            train_mse.append(np.mean((train_attr_dict["targets"] - train_attr_dict["preds"]) ** 2))

            # Validation
            if val_loader is not None:
                val_attr_dict = self.inference_step(val_loader, epoch)

                if val_attr_dict is not None:
                    current_val_loss = np.mean(val_attr_dict["losses"])
                    self.scheduler.step(np.average(val_attr_dict["losses"]))

                    write_summary(self.summary_writer, val_attr_dict, epoch, self.optimizer, "val")

                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        save_model(self.out_dir, epoch, self.model, self.optimizer)
                    elif np.mean(train_attr_dict["losses"]) < best_train_loss:
                        best_train_loss = np.mean(train_attr_dict["losses"])
                        save_model(osp.join(self.out_dir, "best_train"), epoch, self.model, self.optimizer)

                    val_loss.append(np.average(val_attr_dict["losses"]))
                    val_mse.append(np.mean((val_attr_dict["targets"] - val_attr_dict["preds"]) ** 2))
            else:
                # No Validation, use Train loss
                self.scheduler.step(np.average(train_attr_dict["losses"]))
                if np.mean(train_attr_dict["losses"]) < best_train_loss:
                    best_train_loss = np.mean(train_attr_dict["losses"])
                    save_model(self.out_dir, epoch, self.model, self.optimizer)

            # Test (Monitor)
            if test_loader is not None:
                test_attr_dict = self.inference_step(test_loader, epoch)
                if test_attr_dict is not None:
                    write_summary(self.summary_writer, test_attr_dict, epoch, self.optimizer, "test")
                    test_loss.append(np.average(test_attr_dict["losses"]))
                    test_mse.append(np.mean((test_attr_dict["targets"] - test_attr_dict["preds"]) ** 2))

            if self.optimizer.param_groups[0]['lr'] < 1.1e-6:
                logging.info("Learning rate too small, early stopping.")
                break

        # Plotting
        train_loss = np.array(train_loss)
        train_mse = np.array(train_mse)

        if len(train_loss) == 0: return

        fig = plt.figure(num="loss", dpi=90, figsize=(16, 9))
        plt.plot(range(train_loss.shape[0]), train_loss, "-b", linewidth=0.5, label="train_loss")
        loss_all = train_loss[:, np.newaxis]
        mse_all = train_mse[:, np.newaxis]

        if val_loader is not None and len(val_loss) > 0:
            val_loss = np.array(val_loss)
            val_mse = np.array(val_mse)
            if len(val_loss) == len(train_loss):
                plt.plot(range(val_loss.shape[0]), val_loss, "-r", linewidth=0.5, label="val_loss")
                loss_all = np.concatenate((loss_all, val_loss[:, np.newaxis]), axis=1)
                mse_all = np.concatenate((mse_all, val_mse[:, np.newaxis]), axis=1)

        if test_loader is not None and len(test_loss) > 0:
            test_loss = np.array(test_loss)
            test_mse = np.array(test_mse)
            if len(test_loss) == len(train_loss):
                plt.plot(range(test_loss.shape[0]), test_loss, "-g", linewidth=0.5, label="test_loss")
                loss_all = np.concatenate((loss_all, test_loss[:, np.newaxis]), axis=1)
                mse_all = np.concatenate((mse_all, test_mse[:, np.newaxis]), axis=1)

        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend()
        try:
            fig.savefig(osp.join(self.out_dir, "epoch_loss.png"))
        except:
            pass

        np.savetxt(osp.join(self.out_dir, "epoch_loss.txt"), loss_all, delimiter=',', )
        np.savetxt(osp.join(self.out_dir, "epoch_mse.txt"), mse_all, delimiter=',', )
        logging.info("Training complete.")