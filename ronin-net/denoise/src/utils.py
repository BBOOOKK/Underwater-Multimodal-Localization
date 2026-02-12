import torch
import os
import pickle
import yaml
import numpy as np


# =============================================================================
# 📂 1. 文件处理与目录管理
# =============================================================================

def pload(*f_names):
    """Pickle load"""
    f_name = os.path.join(*f_names)
    with open(f_name, "rb") as f:
        pickle_dict = pickle.load(f)
    return pickle_dict


def pdump(pickle_dict, *f_names):
    """Pickle dump (Safe CPU Save)"""
    f_name = os.path.join(*f_names)
    parent_dir = os.path.dirname(f_name)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    def move_to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.cpu()
        elif isinstance(obj, dict):
            return {k: move_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_to_cpu(v) for v in obj]
        return obj

    safe_dict = move_to_cpu(pickle_dict)

    with open(f_name, "wb") as f:
        pickle.dump(safe_dict, f)


def mkdir(*paths):
    """Create a directory if not existing."""
    path = os.path.join(*paths)
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


# =============================================================================
# 🔢 2. 矩阵运算与批处理
# =============================================================================

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


# =============================================================================
# 📐 3. 李代数核心库 (SO3)
# =============================================================================

class SO3:
    # tolerance criterion
    TOL = 1e-8
    Id = torch.eye(3)

    @classmethod
    def exp(cls, phi):
        angle = phi.norm(dim=1, keepdim=True)
        mask = angle[:, 0] < cls.TOL
        dim_batch = phi.shape[0]

        Id = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(dim_batch, 3, 3)

        axis = phi[~mask] / angle[~mask]
        c = angle[~mask].cos().unsqueeze(2)
        s = angle[~mask].sin().unsqueeze(2)

        Rot = phi.new_empty(dim_batch, 3, 3)
        Rot[mask] = Id[mask] + SO3.wedge(phi[mask])
        Rot[~mask] = c * Id[~mask] + \
                     (1 - c) * cls.bouter(axis, axis) + s * cls.wedge(axis)
        return Rot

    @classmethod
    def log(cls, Rot):
        dim_batch = Rot.shape[0]
        tr = cls.btrace(Rot)
        cos_angle = (0.5 * tr - 0.5).clamp(-1. + cls.TOL, 1. - cls.TOL)
        angle = cos_angle.acos()

        mask = angle < 1e-4

        phi = Rot.new_empty(dim_batch, 3)
        if (~mask).any():
            idx = (~mask)
            s = angle[idx].sin()
            factor = 0.5 * angle[idx] / s
            phi[idx] = cls.vee((Rot[idx] - Rot[idx].transpose(1, 2)) * factor.unsqueeze(1).unsqueeze(2))
        if mask.any():
            phi[mask] = 0.5 * cls.vee(Rot[mask] - Rot[mask].transpose(1, 2))

        return phi

    @staticmethod
    def vee(Phi):
        return torch.stack((Phi[:, 2, 1],
                            Phi[:, 0, 2],
                            Phi[:, 1, 0]), dim=1)

    @staticmethod
    def wedge(phi):
        dim_batch = phi.shape[0]
        zero = phi.new_zeros(dim_batch)
        return torch.stack((zero, -phi[:, 2], phi[:, 1],
                            phi[:, 2], zero, -phi[:, 0],
                            -phi[:, 1], phi[:, 0], zero), 1).view(dim_batch, 3, 3)

    @classmethod
    def from_rpy(cls, roll, pitch, yaw):
        return cls.rotz(yaw).bmm(cls.roty(pitch).bmm(cls.rotx(roll)))

    @classmethod
    def rotx(cls, angle_in_radians):
        c = angle_in_radians.cos()
        s = angle_in_radians.sin()
        mat = c.new_zeros((c.shape[0], 3, 3))
        mat[:, 0, 0] = 1
        mat[:, 1, 1] = c
        mat[:, 2, 2] = c
        mat[:, 1, 2] = -s
        mat[:, 2, 1] = s
        return mat

    @classmethod
    def roty(cls, angle_in_radians):
        c = angle_in_radians.cos()
        s = angle_in_radians.sin()
        mat = c.new_zeros((c.shape[0], 3, 3))
        mat[:, 1, 1] = 1
        mat[:, 0, 0] = c
        mat[:, 2, 2] = c
        mat[:, 0, 2] = s
        mat[:, 2, 0] = -s
        return mat

    @classmethod
    def rotz(cls, angle_in_radians):
        c = angle_in_radians.cos()
        s = angle_in_radians.sin()
        mat = c.new_zeros((c.shape[0], 3, 3))
        mat[:, 2, 2] = 1
        mat[:, 0, 0] = c
        mat[:, 1, 1] = c
        mat[:, 0, 1] = -s
        mat[:, 1, 0] = s
        return mat

    @classmethod
    def isclose(cls, x, y):
        return (x - y).abs() < cls.TOL

    @classmethod
    def to_rpy(cls, Rots):
        pitch = torch.atan2(-Rots[:, 2, 0],
                            torch.sqrt(Rots[:, 0, 0] ** 2 + Rots[:, 1, 0] ** 2))
        yaw = pitch.new_empty(pitch.shape)
        roll = pitch.new_empty(pitch.shape)

        near_pi_over_two_mask = cls.isclose(pitch, np.pi / 2.)
        near_neg_pi_over_two_mask = cls.isclose(pitch, -np.pi / 2.)
        remainder_inds = ~(near_pi_over_two_mask | near_neg_pi_over_two_mask)

        yaw[near_pi_over_two_mask] = 0
        roll[near_pi_over_two_mask] = torch.atan2(
            Rots[near_pi_over_two_mask, 0, 1],
            Rots[near_pi_over_two_mask, 1, 1])

        yaw[near_neg_pi_over_two_mask] = 0.
        roll[near_neg_pi_over_two_mask] = -torch.atan2(
            Rots[near_neg_pi_over_two_mask, 0, 1],
            Rots[near_neg_pi_over_two_mask, 1, 1])

        sec_pitch = 1 / pitch[remainder_inds].cos()
        remainder_mats = Rots[remainder_inds]
        yaw[remainder_inds] = torch.atan2(remainder_mats[:, 1, 0] * sec_pitch,
                                          remainder_mats[:, 0, 0] * sec_pitch)
        roll[remainder_inds] = torch.atan2(remainder_mats[:, 2, 1] * sec_pitch,
                                           remainder_mats[:, 2, 2] * sec_pitch)
        rpys = torch.cat([roll.unsqueeze(dim=1),
                          pitch.unsqueeze(dim=1),
                          yaw.unsqueeze(dim=1)], dim=1)
        return rpys

    @classmethod
    def from_quaternion(cls, quat, ordering='wxyz'):
        if ordering == 'xyzw':
            qx = quat[:, 0];
            qy = quat[:, 1];
            qz = quat[:, 2];
            qw = quat[:, 3]
        elif ordering == 'wxyz':
            qw = quat[:, 0];
            qx = quat[:, 1];
            qy = quat[:, 2];
            qz = quat[:, 3]

        mat = quat.new_empty(quat.shape[0], 3, 3)
        qx2 = qx * qx;
        qy2 = qy * qy;
        qz2 = qz * qz

        mat[:, 0, 0] = 1. - 2. * (qy2 + qz2)
        mat[:, 0, 1] = 2. * (qx * qy - qw * qz)
        mat[:, 0, 2] = 2. * (qw * qy + qx * qz)

        mat[:, 1, 0] = 2. * (qw * qz + qx * qy)
        mat[:, 1, 1] = 1. - 2. * (qx2 + qz2)
        mat[:, 1, 2] = 2. * (qy * qz - qw * qx)

        mat[:, 2, 0] = 2. * (qx * qz - qw * qy)
        mat[:, 2, 1] = 2. * (qw * qx + qy * qz)
        mat[:, 2, 2] = 1. - 2. * (qx2 + qy2)
        return mat

    @classmethod
    def to_quaternion(cls, Rots, ordering='wxyz'):
        tmp = 1 + Rots[:, 0, 0] + Rots[:, 1, 1] + Rots[:, 2, 2]
        tmp[tmp < 0] = 0
        qw = 0.5 * torch.sqrt(tmp)
        qx = qw.new_empty(qw.shape[0])
        qy = qw.new_empty(qw.shape[0])
        qz = qw.new_empty(qw.shape[0])

        near_zero_mask = qw.abs() < cls.TOL

        if near_zero_mask.sum() > 0:
            cond1_mask = near_zero_mask * (Rots[:, 0, 0] > Rots[:, 1, 1]) * (Rots[:, 0, 0] > Rots[:, 2, 2])
            cond1_inds = cond1_mask.nonzero()
            if len(cond1_inds) > 0:
                cond1_inds = cond1_inds.squeeze()
                if cond1_inds.ndim == 0: cond1_inds = cond1_inds.unsqueeze(0)
                R_cond1 = Rots[cond1_inds].view(-1, 3, 3)
                d = 2. * torch.sqrt(1. + R_cond1[:, 0, 0] - R_cond1[:, 1, 1] - R_cond1[:, 2, 2]).view(-1)
                qw[cond1_inds] = (R_cond1[:, 2, 1] - R_cond1[:, 1, 2]) / d
                qx[cond1_inds] = 0.25 * d
                qy[cond1_inds] = (R_cond1[:, 1, 0] + R_cond1[:, 0, 1]) / d
                qz[cond1_inds] = (R_cond1[:, 0, 2] + R_cond1[:, 2, 0]) / d

            cond2_mask = near_zero_mask * (Rots[:, 1, 1] > Rots[:, 2, 2])
            cond2_inds = cond2_mask.nonzero()
            if len(cond2_inds) > 0:
                cond2_inds = cond2_inds.squeeze()
                if cond2_inds.ndim == 0: cond2_inds = cond2_inds.unsqueeze(0)
                R_cond2 = Rots[cond2_inds].view(-1, 3, 3)
                d = 2. * torch.sqrt(1. + R_cond2[:, 1, 1] - R_cond2[:, 0, 0] - R_cond2[:, 2, 2]).squeeze()
                tmp = (R_cond2[:, 0, 2] - R_cond2[:, 2, 0]) / d
                qw[cond2_inds] = tmp
                qx[cond2_inds] = (R_cond2[:, 1, 0] + R_cond2[:, 0, 1]) / d
                qy[cond2_inds] = 0.25 * d
                qz[cond2_inds] = (R_cond2[:, 2, 1] + R_cond2[:, 1, 2]) / d

            cond3_mask = near_zero_mask & cond1_mask.logical_not() & cond2_mask.logical_not()
            cond3_inds = cond3_mask
            if len(cond3_inds.nonzero()) > 0:
                R_cond3 = Rots[cond3_inds].view(-1, 3, 3)
                d = 2. * torch.sqrt(1. + R_cond3[:, 2, 2] - R_cond3[:, 0, 0] - R_cond3[:, 1, 1]).squeeze()
                qw[cond3_inds] = (R_cond3[:, 1, 0] - R_cond3[:, 0, 1]) / d
                qx[cond3_inds] = (R_cond3[:, 0, 2] + R_cond3[:, 2, 0]) / d
                qy[cond3_inds] = (R_cond3[:, 2, 1] + R_cond3[:, 1, 2]) / d
                qz[cond3_inds] = 0.25 * d

        far_zero_mask = near_zero_mask.logical_not()
        far_zero_inds = far_zero_mask
        if len(far_zero_inds.nonzero()) > 0:
            R_fz = Rots[far_zero_inds]
            d = 4. * qw[far_zero_inds]
            qx[far_zero_inds] = (R_fz[:, 2, 1] - R_fz[:, 1, 2]) / d
            qy[far_zero_inds] = (R_fz[:, 0, 2] - R_fz[:, 2, 0]) / d
            qz[far_zero_inds] = (R_fz[:, 1, 0] - R_fz[:, 0, 1]) / d

        if ordering == 'xyzw':
            quat = torch.stack([qx, qy, qz, qw], dim=1)
        elif ordering == 'wxyz':
            quat = torch.stack([qw, qx, qy, qz], dim=1)
        return quat

    @classmethod
    def normalize(cls, Rots):
        U, _, V = torch.svd(Rots)
        S = torch.eye(3, device=Rots.device, dtype=Rots.dtype).unsqueeze(0).repeat(Rots.shape[0], 1, 1)
        S[:, 2, 2] = torch.det(U) * torch.det(V)
        return U.bmm(S).bmm(V.transpose(1, 2))

    @classmethod
    def qmul(cls, q, r, ordering='wxyz'):
        terms = cls.bouter(r, q)
        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        xyz = torch.stack((x, y, z), dim=1)
        xyz[w < 0] *= -1
        w[w < 0] *= -1
        if ordering == 'wxyz':
            q = torch.cat((w.unsqueeze(1), xyz), dim=1)
        else:
            q = torch.cat((xyz, w.unsqueeze(1)), dim=1)
        return q / q.norm(dim=1, keepdim=True)

    @staticmethod
    def sinc(x):
        return x.sin() / x

    @classmethod
    def qexp(cls, xi, ordering='wxyz'):
        theta = xi.norm(dim=1, keepdim=True)
        w = (0.5 * theta).cos()
        xyz = 0.5 * cls.sinc(0.5 * theta) * xi
        return torch.cat((w, xyz), 1)

    @classmethod
    def qlog(cls, q, ordering='wxyz'):
        """
        [Math Fix] 修正后的四元数对数映射，增加了极小值保护
        """
        # 提取虚部 norm，即 sin(theta/2)
        q_imag_norm = torch.norm(q[:, 1:], p=2, dim=1, keepdim=True)

        # 防止除以 0
        q_imag_norm = torch.clamp(q_imag_norm, min=1e-8)

        # acos(w) = theta / 2
        half_theta = torch.acos(torch.clamp(q[:, :1], min=-1.0, max=1.0))

        # 简单的泰勒展开保护 (虽然 clamp 已经起了作用，但这样更符合数学定义)
        # 当 theta 趋近于 0 时，sin(theta/2) / (theta/2) -> 1
        # 所以 r = v
        # 这里为了保持梯度流，我们直接使用 clamp 后的除法
        r = (q[:, 1:] / q_imag_norm) * (2 * half_theta)

        return r

    @classmethod
    def qinv(cls, q, ordering='wxyz'):
        r = torch.empty_like(q)
        if ordering == 'wxyz':
            r[:, 1:4] = -q[:, 1:4]
            r[:, 0] = q[:, 0]
        else:
            r[:, :3] = -q[:, :3]
            r[:, 3] = q[:, 3]
        return r

    @classmethod
    def qnorm(cls, q):
        return q / q.norm(dim=1, keepdim=True)

    @classmethod
    def slerp(cls, q0, q1, tau, DOT_THRESHOLD=0.9995):
        dot = (q0 * q1).sum(dim=1)
        q1[dot < 0] = -q1[dot < 0]
        dot[dot < 0] = -dot[dot < 0]

        q = torch.zeros_like(q0)
        tmp = q0 + tau.unsqueeze(1) * (q1 - q0)
        tmp = tmp[dot > DOT_THRESHOLD]
        q[dot > DOT_THRESHOLD] = tmp / tmp.norm(dim=1, keepdim=True)

        theta_0 = dot.acos()
        sin_theta_0 = theta_0.sin()
        theta = theta_0 * tau
        sin_theta = theta.sin()
        s0 = (theta.cos() - dot * sin_theta / sin_theta_0).unsqueeze(1)
        s1 = (sin_theta / sin_theta_0).unsqueeze(1)
        q[dot < DOT_THRESHOLD] = ((s0 * q0) + (s1 * q1))[dot < DOT_THRESHOLD]
        return q / q.norm(dim=1, keepdim=True)

    @staticmethod
    def bouter(vec1, vec2):
        return torch.einsum('bi, bj -> bij', vec1, vec2)

    @staticmethod
    def btrace(mat):
        return torch.einsum('bii -> b', mat)


class CPUSO3(SO3):
    pass


# =============================================================================
# 🧭 4. 核心积分与姿态处理逻辑
# =============================================================================

def integrate_orientation(omega, q0, dt):
    """
    Standard Quaternion Integration using 0-order hold (exponential map)
    """
    N = omega.shape[0]
    delta_rot = SO3.qexp(omega * dt)  # (N, 4)

    # 2. 顺序累积
    qs = torch.empty_like(delta_rot)
    curr_q = q0.view(1, 4)

    for i in range(N):
        curr_q = SO3.qmul(curr_q, delta_rot[i:i + 1])
        # 🔥 [Fix] 显式处理维度，防止广播警告
        qs[i] = curr_q.squeeze(0)

    return SO3.qnorm(qs).float()


def unwrap_rpy(rpy_deg):
    rpy_rad = np.deg2rad(rpy_deg)
    rpy_unwrapped = np.unwrap(rpy_rad, axis=0)
    return np.rad2deg(rpy_unwrapped)


def get_orientation_error(R_pred, R_gt):
    dR = bmtm(R_pred, R_gt)
    error_rad = SO3.log(dR).norm(dim=1)
    return error_rad * 180 / np.pi