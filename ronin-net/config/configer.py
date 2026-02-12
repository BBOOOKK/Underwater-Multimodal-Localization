import yaml
import os
import re
import train, test
from model import model_lstm
import logging
import torch
import torch.nn as nn


# General config
def load_config(path):
    ''' Loads config file. '''
    path = path.strip()
    if not os.path.exists(path):
        print(f"Config path not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    # 🔥 强制关闭多卡配置，防止意外
    if 'train' in cfg_special:
        cfg_special['train']['use_multi_gpu'] = False

    return cfg_special


def update_recursive(dict1, dict2):
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def build_model(args, cfg):
    # 🔥 单卡模式：直接使用 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(cfg['model']['model_yaml'], 'r', encoding='utf-8') as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)
    update_recursive(cfg, cfg_special)

    model_name = cfg['model']['model_name']
    if model_name == 'resnet_lstm':
        model = model_lstm.ResNetLSTMSeqNet(cfg)

    # 🔥 移除所有 DDP 逻辑，直接转到设备
    network = model.to(device)
    total_params = network.get_num_params()

    logging.info(f'Network "{model_name}" loaded to device {device}')
    logging.info(f"Total number of parameters: {total_params}")

    return network


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]


def GetBestModel(path):
    if not os.path.exists(path):
        return None

    names = sorted(os.listdir(path), key=str2int)
    files = []
    for name in names:
        if os.path.isfile(os.path.join(os.path.abspath(path), name)):
            files.append(name)

    if len(files) == 0:
        return None

    model = os.path.join(os.path.abspath(path), files[-1])
    logging.info(f"load model: {model}")
    return model


def build_trainer(args, cfg, model, **kwargs):
    start_epoch = 0
    optim = torch.optim.Adam if cfg['train']['optimizer']['method'] == 'Adam' else torch.optim.SGD
    optimizer = optim(model.parameters(), cfg['train']['optimizer']['learning_rate'],
                      weight_decay=cfg['train']['optimizer']['weight_decay'])

    ckpt_path = os.path.join(cfg['train']['out_dir'], "checkpoints")
    best_model_path = GetBestModel(ckpt_path)

    if cfg['train']['use_pretrain_model'] and best_model_path is not None:
        logging.info(f"Resuming training from checkpoint: {best_model_path}")
        checkpoint = torch.load(best_model_path)
        start_epoch = checkpoint.get("epoch", 0)

        # 🔥 单卡直接加载，不需要 module. 前缀处理
        model.load_state_dict(checkpoint.get("model_state_dict"))

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint.get("optimizer_state_dict"))

        logging.info(f"Continue from epoch {start_epoch}")

    return train.trainer(args, cfg, model=model, optimizer=optimizer, start_epoch=start_epoch)


def build_tester(args, cfg, model, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(cfg['train']['out_dir'], "checkpoints")
    best_model_path = GetBestModel(ckpt_path)

    if best_model_path is None:
        logging.warning(f"No checkpoint found in {ckpt_path}, testing with random weights!")
    else:
        checkpoint = torch.load(best_model_path, map_location=device)
        # 🔥 单卡直接加载
        model.load_state_dict(checkpoint.get("model_state_dict"))

    model.eval()
    tester = test.tester(args, cfg, model)
    return tester