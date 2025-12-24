import torch
import os
import pickle
import yaml
import numpy as np


def pload(*f_names):
    """Pickle load"""
    f_name = os.path.join(*f_names)
    with open(f_name, "rb") as f:
        pickle_dict = pickle.load(f)
    return pickle_dict


def pdump(pickle_dict, *f_names):
    """Pickle dump"""
    f_name = os.path.join(*f_names)
    # 确保保存路径的父文件夹存在，否则会报错
    parent_dir = os.path.dirname(f_name)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    with open(f_name, "wb") as f:
        pickle.dump(pickle_dict, f)


def mkdir(*paths):
    '''Create a directory if not existing.'''
    path = os.path.join(*paths)
    # exist_ok=True 防止目录已存在时报错
    # makedirs 可以递归创建目录 (比如创建 a/b/c)
    os.makedirs(path, exist_ok=True)


def yload(*f_names):
    """YAML load"""
    f_name = os.path.join(*f_names)
    with open(f_name, 'r') as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict


def ydump(yaml_dict, *f_names):
    """YAML dump"""
    f_name = os.path.join(*f_names)
    parent_dir = os.path.dirname(f_name)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    with open(f_name, 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)


def bmv(mat, vec):
    """batch matrix vector product"""
    return torch.einsum('bij, bj -> bi', mat, vec)


def bbmv(mat, vec):
    """double batch matrix vector product"""
    return torch.einsum('baij, baj -> bai', mat, vec)


def bmtv(mat, vec):
    """batch matrix transpose vector product"""
    return torch.einsum('bji, bj -> bi', mat, vec)


def bmtm(mat1, mat2):
    """batch matrix transpose matrix product"""
    return torch.einsum("bji, bjk -> bik", mat1, mat2)


def bmmt(mat1, mat2):
    """batch matrix matrix transpose product"""
    return torch.einsum("bij, bkj -> bik", mat1, mat2)


# [新增] 通用积分函数 (备用)
def integrate(omega, q0, dt):
    """
    Integrate angular velocity to orientation (quaternion).
    omega: (N, 3)
    q0: (1, 4) initial quaternion
    dt: scalar
    """
    from src.lie_algebra import SO3
    # 确保在同一设备
    if omega.device != q0.device:
        q0 = q0.to(omega.device)

    N = omega.shape[0]
    qs = torch.zeros(N, 4, device=omega.device, dtype=omega.dtype)

    # 转换为旋转增量四元数
    # exp(w * dt)
    dqs = SO3.qexp(omega * dt)

    # 累乘
    qs[0] = q0
    current_q = q0
    for i in range(1, N):
        current_q = SO3.qnorm(SO3.qmul(current_q, dqs[i].unsqueeze(0)))
        qs[i] = current_q

    return qs