#!/usr/bin/env python3
"""旁路机制探针：直接在测试集上量化 dual 旁路到底在做什么。

对每个 (dataset, L, seed) 的已训 checkpoint 测量：
  1. bypass 增益：MSE(y) vs MSE(y_main)（归一化空间，直接证据）
  2. 修正量幅度：||α·corr|| / ||residual||，corr 与 residual 的逐变量相关
  3. S 路消融：S=0 前后修正量变化（S 信号对预测的净贡献）
  4. K 矩阵结构：offdiag 均值/std、秩（top-1 特征值占比）、与真实跨变量相关的 Spearman

用法：
  python analyze_bypass.py --ds etth1 --L 96 --seed 42 --tag old
  python analyze_bypass.py --ds electricity --L 720 --seed 42 --tag new
  python analyze_bypass.py --ds pems_bay --L 336 --seed 42 --tag both
"""
import argparse
import glob
import os
import random
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from data.dataloader import build_standard_loaders
from model.qcc_mamba import QCCMamba
from engine.train import compute_global_stats, set_global_stats

DATA_DIR = os.path.join(ROOT, "..", "ts_quantum", "datasets")
N_BATCHES = 8


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_model(cfg, num_var):
    m = cfg["model"]
    return QCCMamba(
        num_var=num_var,
        lookback=cfg.get("lookback", 96),
        horizon=cfg.get("horizon", 96),
        d_token=m.get("d_token", 512),
        n_qubits=m.get("n_qubits", 10),
        n_layers=m.get("n_layers", 2),
        entangle_topo=m.get("entangle_topo", "linear"),
        kernel_fn=m.get("kernel_fn", "quantum"),
        use_fmap=m.get("use_fmap", True),
        alpha0=m.get("alpha0", 0.1),
        theta_S_scale0=m.get("theta_S_scale0", 0.5),
        beta=m.get("beta", 0.1),
        use_periodic_feat=m.get("use_periodic_feat", True),
        revin_affine=m.get("revin_affine", True),
        use_spectrum=m.get("use_spectrum", True),
        spectrum_M=m.get("spectrum_M", 32),
        spectrum_range=m.get("spectrum_range", "0_2"),
        spectrum_amp_normalize=m.get("spectrum_amp_normalize", False),
        spectrum_time_align=m.get("spectrum_time_align", True),
        spectrum_freq_align=m.get("spectrum_freq_align", True),
        use_H=m.get("use_H", True),
        use_S=m.get("use_S", True),
        use_bypass=m.get("use_bypass", True),
        reupload_source=m.get("reupload_source", "S"),
        angle_norm=m.get("angle_norm", "clamp"),
        angle_radius=m.get("angle_radius", 1.0),
        corr_use_S=m.get("corr_use_S", False),
        corr_gate_S=m.get("corr_gate_S", False),
        corr_use_ymain=m.get("corr_use_ymain", False),
        corr_use_Hp=m.get("corr_use_Hp", False),
        corr_mlp=m.get("corr_mlp", False),
    )


def load_runs(cfg_path, ckpt_path):
    cfg = yaml.safe_load(open(cfg_path))
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_standard_loaders(
        dataset_name=cfg["dataset"],
        data_dir=cfg.get("data_dir", DATA_DIR),
        lookback=cfg.get("lookback", 96),
        horizon=cfg.get("horizon", 96),
        batch_size=cfg["train"].get("batch_size", 32),
        stride=cfg["train"].get("stride", 1),
        num_workers=0,
    )
    inner = getattr(loaders["train"].dataset, "dataset", None)
    train_data = getattr(inner, "data", None)
    if train_data is not None:
        arr = train_data.detach().cpu().numpy()
        gmean, gstd = compute_global_stats(arr.reshape(-1, arr.shape[-1]))
        set_global_stats(gmean, gstd)

    _inner = getattr(loaders["train"].dataset, "dataset", loaders["train"].dataset)
    _data = getattr(_inner, "data", None)
    num_var = _data.shape[1]

    model = make_model(cfg, num_var).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        alpha = model.qcc.alpha.item() if getattr(model, "qcc", None) is not None else None
        gamma = model.qcc.gamma.item() if getattr(model, "qcc", None) is not None else None

    return model, loaders["test"], device, {"alpha": alpha, "gamma": gamma}


def collect(model, loader, device):
    """收集 N_BATCHES 个 test batch 的归一化空间量。"""
    y_ss = 0.0; y_main_ss = 0.0; r_ss = 0.0; c_ss = 0.0
    corr_acc = []
    s_corr_d = 0.0   # S=0 与 S 的修正量差（平方和）
    corr_ss = 0.0    # ||corr||²
    n = 0
    k_offdiag_acc = []
    k_eigs_acc = []
    k_spearman_acc = []
    true_corr_acc = []
    for i, batch in enumerate(loader):
        if i >= N_BATCHES:
            break
        x, y_true = batch[0].to(device), batch[1].to(device)
        x_mark = batch[2].to(device) if len(batch) == 4 else None
        with torch.no_grad():
            y, y_main, K, y_norm, y_main_norm, corr = model(x, x_mark=x_mark, return_norm=True)
            x_mean = model.revin.mean
            x_stdev = model.revin.stdev
            y_true_norm = (y_true - x_mean) / x_stdev
            if model.revin.affine:
                y_true_norm = y_true_norm * model.revin.affine_weight + model.revin.affine_bias

            x_norm = model.revin(x, mode="norm")  # 供 S 消融与 K-真实相关用
            B, H, V = y_norm.shape
            residual = y_true_norm - y_main_norm  # (B,H,V)
            corr_a = model.qcc.alpha * corr if model.qcc is not None else None

            y_ss += torch.sum((y_norm - y_true_norm) ** 2).item()
            y_main_ss += torch.sum((y_main_norm - y_true_norm) ** 2).item()
            r_ss += torch.sum(residual ** 2).item()
            if corr_a is not None:
                c_ss += torch.sum(corr_a ** 2).item()
                corr_ss += torch.sum(corr ** 2).item()
                # 逐变量 Pearson：corr 与 residual
                cflat = corr.permute(0, 2, 1).reshape(-1, corr.shape[-1])  # (B*V, H) 换轴
                rflat = residual.permute(0, 2, 1).reshape(-1, residual.shape[-1])
                c_m = cflat - cflat.mean(0, keepdim=True)
                r_m = rflat - rflat.mean(0, keepdim=True)
                denom = c_m.norm(2, 0) * r_m.norm(2, 0)
                rho = (c_m * r_m).sum(0) / denom.clamp_min(1e-8)
                corr_acc.append(rho.mean().item())
                # S=0 消融：修正量对 S 信号的敏感性（S→0，其余不变）
                if model.spectrum is not None and model.qcc.use_S:
                    S0 = torch.zeros(B, V, model.spectrum.M * 2, device=device)
                    H_tok = model.backbone(model._prepare_input(x_norm, x_mark)).H
                    _, _, corr0 = model.qcc(H_tok.detach(), y_main_norm.detach(), S0)
                    s_corr_d += torch.sum((corr - corr0) ** 2).item()
            # K 结构（第一个 batch 采样统计）
            if K is not None and i == 0:
                Km = K.float().mean(0)  # (V,V)
                off = Km - torch.diag(torch.diag(Km))
                k_offdiag_acc.append(off.mean().item())
                k_offdiag_std = off.std().item()
                # 特征值谱：top-1 占比（秩塌缩指标）
                ev = torch.linalg.eigvalsh(Km.double())
                ev = ev.clamp(min=0)
                k_eigs_acc.append((ev[-1] / ev.sum()).item())
                # K offdiag vs 真实跨变量相关
                xflat = x_norm.reshape(-1, V).double()  # (B*L, V)
                xc = torch.corrcoef(xflat.T)  # (V,V)
                v_idx = torch.triu_indices(V, V, 1)
                kv = off.flatten()[v_idx[0] * V + v_idx[1]].double()
                xv = xc.flatten()[v_idx[0] * V + v_idx[1]].double()
                m = kv.numel()
                rng = np.random.RandomState(0)
                pick = rng.choice(m, min(3000, m), replace=False)
                kv_s = kv[pick]; xv_s = xv[pick]
                kv_r = kv_s - kv_s.mean(); xv_r = xv_s - xv_s.mean()
                sp = (kv_r * xv_r).sum() / (kv_r.norm() * xv_r.norm()).clamp_min(1e-12)
                k_spearman_acc.append(sp.item())
            n += y_true.numel()
    out = dict(
        mse_y=(y_ss / n), mse_y_main=(y_main_ss / n),
        corr_residual_ratio=(c_ss / r_ss) if r_ss > 0 else None,
        corr_magnitude_ratio=(corr_ss / r_ss) if r_ss > 0 else None,
        corr_pearson=float(np.mean(corr_acc)) if corr_acc else None,
        s_ablation_delta=(s_corr_d / corr_ss) if corr_ss > 0 else None,
        k_offdiag=np.mean(k_offdiag_acc) if k_offdiag_acc else None,
        k_offdiag_std=k_offdiag_std,
        k_top1_eigfrac=np.mean(k_eigs_acc) if k_eigs_acc else None,
        k_truecorr_spearman=np.mean(k_spearman_acc) if k_spearman_acc else None,
    )
    return out


def analyze(cfg_path, ckpt_path, label):
    model, loader, device, w = load_runs(cfg_path, ckpt_path)
    r = collect(model, loader, device)
    print(f"\n===== {label} =====")
    print(f"  α={w['alpha']:.4f}  γ={w['gamma']:.4f}" if w["alpha"] is not None else "  无旁路")
    print(f"  MSE_norm(y)      = {r['mse_y']:.4f}")
    print(f"  MSE_norm(y_main) = {r['mse_y_main']:.4f}   ← bypass 增益 {(r['mse_y']-r['mse_y_main']):+.4f}")
    print(f"  ||α·corr||²/||residual||² = {r['corr_residual_ratio']:.4f}  (修正量占残差的比例)")
    print(f"  ||corr||²/||residual||²   = {r['corr_magnitude_ratio']:.4f}  (未缩放修正量)")
    print(f"  Pearson(corr, residual)   = {r['corr_pearson']:.4f}  (修正指向残差的程度)")
    print(f"  S=0 消融 Δ||corr||²/||corr||² = {r['s_ablation_delta']:.4f}  (S 信号对修正量的净贡献)")
    print(f"  K offdiag mean={r['k_offdiag']:.4f} std={r['k_offdiag_std']:.4f}  top1_eig={r['k_top1_eigfrac']:.3f}")
    print(f"  Spearman(K_offdiag, 真实变量相关) = {r['k_truecorr_spearman']:.4f}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", choices=["old", "new", "both", "arch"], default="both")
    args = ap.parse_args()
    ds, L, seed = args.ds, args.L, args.seed

    if args.tag in ("old", "both"):
        for cfg_name, ck_name in [
            ("dual", "dual"),
            ("h_only", "h_only"),
            ("s_only", "s_only"),
        ]:
            cp = f"configs/generated/p1f_{ds}_{cfg_name}_{L}_{seed}.yaml"
            ck = f"results/matrix/p1f_{L}/p1f_{ds}_{cfg_name}_{L}_{seed}_best.pt"
            if os.path.exists(cp) and os.path.exists(ck):
                analyze(cp, ck, f"OLD-P1F {ds} L={L} seed={seed} {cfg_name}")
    if args.tag in ("new", "both"):
        cp = f"configs/fix_matrix/fix_{ds}_{L}_{seed}.yaml"
        ck = f"results/fix_matrix/fix_{ds}_{L}_{seed}_best.pt"
        if os.path.exists(cp) and os.path.exists(ck):
            analyze(cp, ck, f"NEW-fix {ds} L={L} seed={seed} β=1.0 α0=0.3")
    if args.tag == "arch":
        for v in ["s_concat", "hp_channel", "combo"]:
            cp = f"configs/archfix/arch_{v}_{ds}_{L}_{seed}.yaml"
            ck = f"results/archfix/arch_{v}_{ds}_{L}_{seed}_best.pt"
            if os.path.exists(cp) and os.path.exists(ck):
                analyze(cp, ck, f"ARCH {ds} L={L} seed={seed} {v}")


if __name__ == "__main__":
    main()
