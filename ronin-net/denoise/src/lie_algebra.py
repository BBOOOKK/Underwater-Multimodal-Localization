import torch
import numpy as np

class SO3:
    # 容差标准
    TOL = 1e-8

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
        Rot[mask] = Id[mask] + cls.wedge(phi[mask])
        Rot[~mask] = c * Id[~mask] + \
                     (1 - c) * cls.bouter(axis, axis) + s * cls.wedge(axis)
        return Rot

    @classmethod
    def log(cls, Rot):
        dim_batch = Rot.shape[0]
        Id = torch.eye(3, device=Rot.device, dtype=Rot.dtype).expand(dim_batch, 3, 3)

        cos_angle = (0.5 * cls.btrace(Rot) - 0.5).clamp(-1., 1.)
        angle = cos_angle.acos()
        mask = angle < cls.TOL
        
        phi = Rot.new_zeros(dim_batch, 3)
        
        if mask.any():
            phi[mask] = cls.vee(Rot[mask] - Id[mask])
            
        if (~mask).any():
            angle_rem = angle[~mask].unsqueeze(1).unsqueeze(2)
            Rot_rem = Rot[~mask]
            phi[~mask] = cls.vee((0.5 * angle_rem / angle_rem.sin()) * (Rot_rem - Rot_rem.transpose(1, 2)))
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
    def to_rpy(cls, Rots):
        """将旋转矩阵转换为 RPY 欧拉角"""
        pitch = torch.atan2(-Rots[:, 2, 0],
                            torch.sqrt(Rots[:, 0, 0] ** 2 + Rots[:, 1, 0] ** 2))
        yaw = pitch.new_empty(pitch.shape)
        roll = pitch.new_empty(pitch.shape)

        near_pi_over_two_mask = (pitch - np.pi/2).abs() < cls.TOL
        near_neg_pi_over_two_mask = (pitch + np.pi/2).abs() < cls.TOL

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
        return torch.stack([roll, pitch, yaw], dim=1)

    @classmethod
    def from_quaternion(cls, quat, ordering='wxyz'):
        """从四元数生成旋转矩阵"""
        if ordering == 'xyzw':
            qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        else:
            qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        mat = quat.new_empty(quat.shape[0], 3, 3)
        qx2, qy2, qz2 = qx*qx, qy*qy, qz*qz
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
    def qmul(cls, q, r, ordering='wxyz'):
        """四元数汉密尔顿乘法"""
        if ordering == 'wxyz':
            w1, x1, y1, z1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            w2, x2, y2, z2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        else:
            x1, y1, z1, w1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            x2, y2, z2, w2 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        res = torch.stack((w, x, y, z), dim=1) if ordering == 'wxyz' else torch.stack((x, y, z, w), dim=1)
        return res / res.norm(dim=1, keepdim=True)

    @staticmethod
    def sinc(x):
        res = torch.ones_like(x)
        mask = x.abs() > 1e-8
        res[mask] = x[mask].sin() / x[mask]
        return res

    @classmethod
    def qexp(cls, xi, ordering='wxyz'):
        """指数映射：旋转向量 -> 四元数"""
        theta = xi.norm(dim=1, keepdim=True)
        w = (0.5 * theta).cos()
        xyz = 0.5 * cls.sinc(0.5 * theta) * xi
        return torch.cat((w, xyz), 1) if ordering == 'wxyz' else torch.cat((xyz, w), 1)

    @classmethod
    def qnorm(cls, q):
        return q / q.norm(dim=1, keepdim=True)

    @classmethod
    def qinv(cls, q, ordering='wxyz'):
        r = q.clone()
        if ordering == 'wxyz':
            r[:, 1:4] = -q[:, 1:4]
        else:
            r[:, :3] = -q[:, :3]
        return r

    @classmethod
    def qlog(cls, q, ordering='wxyz'):
        """对数映射：四元数 -> 旋转向量"""
        if ordering == 'wxyz':
            w, xyz = q[:, 0:1], q[:, 1:4]
        else:
            w, xyz = q[:, 3:4], q[:, 0:3]
        
        norm_xyz = xyz.norm(dim=1, keepdim=True)
        theta = 2 * torch.atan2(norm_xyz, w)
        # 处理多圈旋转，归一化到 [-pi, pi]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        
        res = torch.zeros_like(xyz)
        mask = norm_xyz.flatten() > cls.TOL
        res[mask] = (theta[mask] / norm_xyz[mask]) * xyz[mask]
        return res

    @staticmethod
    def bouter(vec1, vec2):
        return torch.einsum('bi, bj -> bij', vec1, vec2)

    @staticmethod
    def btrace(mat):
        return torch.einsum('bii -> b', mat)

# =============================================================================
# [修复点] 重新定义 CPUSO3 类，防止 learning.py 报错
# =============================================================================
class CPUSO3(SO3):
    """
    兼容性类：继承自 SO3。
    现在的 SO3 类已经可以处理 CPU 张量，保留此类仅为了满足代码导入依赖。
    """
    pass