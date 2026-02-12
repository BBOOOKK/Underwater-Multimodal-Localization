"""
* This file is part of RNIN-VIO
* (Single-GPU Version: Adapted for Pre-generated MoE Data)
"""
import matplotlib
matplotlib.use('Agg') # 无界面模式
import os
import sys
import numpy as np
import logging
import torch
import random
import shutil
from functools import partial
from config import configer
from dataloader import dataset as dataset_utils
from torch.utils.data import DataLoader

# ================= CONFIGURATION =================
# 指向 generate_ronin_data.py 生成的目录
RONIN_DATA_DIR = r"/ronin-net\denoise\output_csv_moe"

# 定义测试集序列（其余自动作为训练集）
TEST_SEQS = ['01', '02', '03', '04', '05']
# =================================================

def setup_logging():
    logging.basicConfig(
        stream=sys.stdout,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        level=logging.INFO,
    )

def GetDataPath(path):
    """获取目录下所有子文件夹的完整路径"""
    if not os.path.exists(path):
        logging.error(f"Path not found: {path}")
        return []
    names = os.listdir(path)
    folders=[]
    for name in names:
        data_path = os.path.join(os.path.abspath(path), name)
        if os.path.isdir(data_path):
            folders.append(data_path)
    folders.sort()
    return folders

def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_output_dir(out_dir):
    if out_dir is not None:
        os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
        logging.info(f"Training output writes to {out_dir}")

def global_worker_init_fn(worker_id, seed):
    np.random.seed(seed + worker_id)

def train_load_data(cfg):
    """
    自动从 output_csv_moe 加载数据并根据序列名划分 Train/Test
    """
    logging.info(f"Loading data from: {RONIN_DATA_DIR}")
    all_folders = GetDataPath(RONIN_DATA_DIR)

    if not all_folders:
        raise ValueError(f"No data found in {RONIN_DATA_DIR}. Please run generate_ronin_data.py first.")

    train_paths = []
    test_paths = []

    for folder in all_folders:
        seq_name = os.path.basename(folder)
        if seq_name in TEST_SEQS:
            test_paths.append(folder)
        else:
            train_paths.append(folder)

    logging.info(f"Train Sequences ({len(train_paths)}): {[os.path.basename(p) for p in train_paths]}")
    logging.info(f"Test Sequences ({len(test_paths)}): {[os.path.basename(p) for p in test_paths]}")

    init_fn = partial(global_worker_init_fn, seed=int(cfg['seeds']['id']))

    # 1. 训练集
    train_basic_data = dataset_utils.BasicSequenceData(cfg, train_paths, mode="train")
    train_dataset = dataset_utils.SeqToSeqDataset(cfg, train_basic_data, train_basic_data.get_merged_index_map(), mode="train")

    # 2. 验证集 (这里简单起见，如果配置了随机划分，可以从训练集中切分，否则直接用部分训练数据做验证)
    # 为了简化，我们直接用训练集的一个子集或者如果 config 里有设置 split
    val_loader = None
    # 单卡 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=cfg['train']['n_workers'],
        worker_init_fn=init_fn
    )

    # 3. 测试集 (作为验证集监控)
    test_loader_monitor = None
    if len(test_paths) > 0:
        test_basic_data = dataset_utils.BasicSequenceData(cfg, test_paths, mode="test")
        test_dataset = dataset_utils.SeqToSeqDataset(cfg, test_basic_data, test_basic_data.get_merged_index_map(), mode="test")
        test_loader_monitor = DataLoader(test_dataset, batch_size=cfg['data'].get('test_batch_size', 32), shuffle=False)

    return train_loader, val_loader, test_loader_monitor, test_paths

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # 移除 --local_rank，不再需要
    parser.add_argument("--yaml", type=str, default="./config/default.yaml")
    args = parser.parse_args()

    setup_logging()

    # 加载配置
    cfg = configer.load_config(args.yaml)

    # 🔥 强制覆盖配置为单卡模式
    cfg['train']['use_multi_gpu'] = False

    # 确保输出目录存在
    if not os.path.exists(cfg['train']['out_dir']):
        os.makedirs(cfg['train']['out_dir'])

    if cfg['seeds']['use_seeds']:
        set_seeds(cfg['seeds']['id'])

    # 1. 构建模型
    # 伪造一个 args.local_rank = 0 传给 build_model，防止它报错
    class MockArgs:
        local_rank = 0
    model = configer.build_model(MockArgs(), cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 2. 加载数据
    train_loader, val_loader, test_loader_monitor, test_data_paths = train_load_data(cfg)

    # 3. 训练流程
    if cfg['schemes']['train']:
        create_output_dir(cfg['train']['out_dir'])
        shutil.copy(args.yaml, os.path.join(cfg['train']['out_dir'], "default.yaml"))

        # 传递 MockArgs 确保兼容性
        trainer = configer.build_trainer(MockArgs(), cfg, model)
        trainer.train(train_loader, val_loader, test_loader_monitor)

    # 4. 测试流程 (使用独立的测试逻辑)
    if cfg['schemes']['test']:
        logging.info("Starting testing...")
        if len(test_data_paths) > 0:
            tester = configer.build_tester(MockArgs(), cfg, model)
            tester.test(test_data_paths)
        else:
            logging.warning("No test data found in the output directory!")