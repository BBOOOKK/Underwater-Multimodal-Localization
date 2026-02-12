# debug_moe.py (诊断专用 - 终极修复版 V3 - 修正打印)
import os
import torch
import numpy as np
import src.networks as sn
import src.dataset as ds
from src.utils import bmtm, SO3
import shutil
import tempfile
import time

# ================= CONFIGURATION =================
ROOT_DIR = r"/ronin-net\denoise"
DATA_DIR = os.path.join(ROOT_DIR, "..", "data", "all_data")
CACHE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'src', 'data', 'UNDERWATER_CACHE')
FUSION_RES_DIR = os.path.join(ROOT_DIR, "output", "moe_fusion_results")

# 网络参数
NET_PARAMS = {
    'in_dim': 6, 'out_dim': 3, 'c0': 32,
    'num_experts': 4, 'top_k': None,
    'dropout': 0.0,
    'ks': [11, 11, 11, 11], 'ds': [4, 4, 4], 'momentum': 0.1,
    'gyro_std': [0.017, 0.035, 0.087],
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def robust_load(path):
    print(f"📂 尝试加载: {os.path.basename(path)}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    # 尝试直接加载
    try:
        return torch.load(path, map_location=device)
    except PermissionError:
        print("⚠️ 文件被占用 (PermissionError)，尝试复制到临时目录加载...")
    except Exception as e:
        print(f"⚠️ 直接加载出错 ({e})，尝试复制加载...")

    # 备选方案：复制到临时文件夹再加载
    try:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_model_{int(time.time())}.pt")
        shutil.copy2(path, temp_path)
        time.sleep(0.1)  # 让文件系统喘口气
        state_dict = torch.load(temp_path, map_location=device)
        try:
            os.remove(temp_path)
        except:
            pass
        return state_dict
    except Exception as e2:
        raise RuntimeError(f"❌ 彻底加载失败，请检查文件是否被其他程序锁定。\n错误信息: {e2}")


def find_latest_model():
    if not os.path.exists(FUSION_RES_DIR): raise FileNotFoundError(f"目录不存在: {FUSION_RES_DIR}")

    # 递归搜索所有子目录
    candidates = []
    for root, dirs, files in os.walk(FUSION_RES_DIR):
        for file in files:
            if file == "model_last.p" or file == "weights.pt":
                candidates.append(os.path.join(root, file))

    if not candidates: raise FileNotFoundError("未找到任何模型文件 (model_last.p 或 weights.pt)")

    # 按修改时间排序，找最新的
    latest_model = max(candidates, key=os.path.getmtime)
    print(f"📍 锁定最新模型文件: {latest_model}")
    return latest_model


def fast_integrate(gyro, q0, dt):
    N = gyro.shape[0]
    if isinstance(q0, np.ndarray): q0 = torch.from_numpy(q0)
    q0 = q0.to(device).view(1, 4).float()
    if isinstance(gyro, np.ndarray):
        w = torch.from_numpy(gyro)
    else:
        w = gyro
    w = w.to(device).float()

    dqs = SO3.qexp(w * dt)
    qs = torch.zeros(N, 4, device=device)
    qs[0] = q0
    curr_q = q0
    dqs_list = list(torch.unbind(dqs, dim=0))
    res_list = [q0]
    for i in range(N - 1):
        curr_q = SO3.qmul(curr_q, dqs_list[i].view(1, 4))
        res_list.append(curr_q)
    return torch.cat(res_list, 0)


def calc_error_accelerated(pred_us, gt_qs, stride=10):
    dt_raw = 0.01
    N_raw = min(len(pred_us), len(gt_qs))
    N_crop = (N_raw // stride) * stride
    if N_crop == 0: return 0.0

    if isinstance(pred_us, np.ndarray): pred_us = torch.from_numpy(pred_us)
    pred_us = pred_us[:N_crop].to(device).float()
    pred_chunks = pred_us.view(-1, stride, 3)
    pred_agg = pred_chunks.mean(dim=1)

    if isinstance(gt_qs, np.ndarray): gt_qs = torch.from_numpy(gt_qs)
    gt_qs = gt_qs.to(device).float()
    gt_qs_sub = gt_qs[::stride][:len(pred_agg)]

    q0 = gt_qs[0].view(1, 4)
    q_pred = fast_integrate(pred_agg, q0, dt_raw * stride)

    min_len = min(len(q_pred), len(gt_qs_sub))
    q_pred = q_pred[:min_len]
    gt_qs_sub = gt_qs_sub[:min_len]

    R_pred = SO3.from_quaternion(q_pred)
    R_gt = SO3.from_quaternion(gt_qs_sub)
    dR = bmtm(R_pred, R_gt)
    angle = SO3.log(dR).norm(dim=1) * 180 / np.pi
    return torch.sqrt((angle ** 2).mean()).item()


def inspect_experts_internal_state(model):
    print("\n" + "=" * 60)
    print("🔍 [核心体检] Expert 内部状态检查 (Bias Removal Capability)")
    print("=" * 60)
    print(f"{'Expert':<10} | {'Mean_U (Avg)':<15} | {'Std_U (Avg)':<15} | {'Status':<10}")
    print("-" * 60)

    all_good = True
    for i, expert in enumerate(model.experts):
        m_val = expert.mean_u.abs().mean().item()
        s_val = expert.std_u.mean().item()

        if s_val > 0.9:
            status = "❌ RAW"
            all_good = False
        else:
            status = "✅ STAT"

        print(f"Expert {i:<3} | {m_val:<15.6f} | {s_val:<15.6f} | {status}")

    print("-" * 60)
    if not all_good:
        print("⚠️  警告：部分专家处于 RAW 模式 (Std~1.0)，无法去除零偏！")
    else:
        print("🎉 完美：所有专家都加载了统计参数，具备去偏置能力。")
    print("=" * 60 + "\n")


def diagnose():
    model_path = find_latest_model()

    # 重新构建模型结构
    model = sn.MoEGyroNet(**NET_PARAMS).to(device)

    # 加载权重
    checkpoint = robust_load(model_path)
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        state_dict = checkpoint['net']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("无法识别的模型文件格式")

    # 清洗 state_dict 键名
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 🔥 第一步：检查内部参数
    inspect_experts_internal_state(model)

    # 准备测试数据，覆盖所有可能的目标序列
    test_seqs = ['01', '02', '03', '04', '05']
    dataset = ds.EUROCDataset(DATA_DIR, CACHE_DIR, [], [], test_seqs, 'test', 2048, 16, 32)

    print(
        f"{'Seq':<5} | {'Type':<5} | {'GyroMag':<8} | {'Router Conf':<11} | {'Selection Probabilities':<35} | {'FusedErr':<8} | {'BiasEst(E0)'}")
    print("-" * 110)

    for i, seq in enumerate(dataset.sequences):
        if seq not in test_seqs: continue

        data = dataset.load_seq(i)
        gt = dataset.load_gt(i)
        us = data['us'].to(device)

        with torch.no_grad():
            gyro_mag = torch.norm(us[:, :3], dim=1).mean().item()

            # Router Forward
            us_batch = us.unsqueeze(0)
            _, (gates, _) = model(us_batch)
            gates = gates.squeeze(0)  # (Seq, 4)

            # 统计选择
            max_idx = torch.argmax(gates, dim=1)
            counts = torch.bincount(max_idx, minlength=4).float()
            probs = counts / counts.sum()

            # 赢家和置信度
            winner = torch.argmax(probs).item()
            confidence = probs[winner].item() * 100

            prob_str = f"E0:{probs[0]:.2f} E1:{probs[1]:.2f} E2:{probs[2]:.2f} E3:{probs[3]:.2f}"

            # 计算各专家误差 & 偏置估计
            STRIDE = 20
            expert_errs = []
            expert_bias = []

            for k in range(4):
                exp_out = model.experts[k](us_batch).squeeze(0)
                err = calc_error_accelerated(exp_out, gt['qs'], stride=STRIDE)
                expert_errs.append(err)
                bias_norm = exp_out.mean(dim=0).norm().item()
                expert_bias.append(bias_norm)

            # 融合误差
            hard_mask = torch.zeros_like(gates).scatter_(1, max_idx.unsqueeze(1), 1.0)
            all_exp_outs = torch.stack([e(us_batch) for e in model.experts], dim=1).squeeze(0).permute(1, 0, 2)
            fused_out = torch.sum(all_exp_outs * hard_mask.unsqueeze(-1), dim=1)
            fused_err = calc_error_accelerated(fused_out, gt['qs'], stride=STRIDE)

            seq_type = "DYN" if gyro_mag > 0.1 else "STA"
            conf_str = f"{confidence:.0f}% (E{winner})"
            e0_bias_str = f"{expert_bias[0]:.5f}"

            print(
                f"{seq:<5} | {seq_type:<5} | {gyro_mag:<8.4f} | {conf_str:<11} | {prob_str:<35} | {fused_err:<8.2f} | {e0_bias_str}")
            # 🔥 [修复] 打印所有 4 个专家的误差
            print(
                f"      [Loss Breakdown] Fused: {fused_err:.2f} | E0(Stat): {expert_errs[0]:.2f} | E1(Dyn): {expert_errs[1]:.2f} | E2(Mix): {expert_errs[2]:.2f} | E3(Rob): {expert_errs[3]:.2f}")

            if seq == '02' and winner != 0:
                print("      🔴 [FAIL] Router 选错了！静止序列应该选 E0。")
            if seq == '02' and expert_errs[0] > 10.0:
                print("      🔴 [FAIL] E0 还是不行 (误差>10)！Bias Removal 失败。")

            print("-" * 110)


if __name__ == "__main__":
    diagnose()