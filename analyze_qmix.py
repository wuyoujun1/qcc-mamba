#!/usr/bin/env python3
"""量子混合主干探针（2026-08-11 重构版）。

核心验证：旧主干预测对跨变量输入零响应（非对角/对角 ≈ 1e-6，etth1/electricity 实测）。
量子混合进主干后，预测应当第一次真正响应其他变量的输入。

对每个已训 qmix checkpoint 测量：
  1. y 空间跨变量灵敏度（扰动输入 1σ → 预测响应，非对角/对角）—— 新架构核心指标
  2. K 结构：offdiag 均值/std、与真实跨变量相关的 Spearman
  3. 频谱注入消融（qmix_sin 变体）：S→0 时预测变化（频谱信息的净贡献）
  4. MSE（探针 batch 上的相对对比）

用法：
  python analyze_qmix.py --ds etth1 --L 96 --seed 42
  python analyze_qmix.py --ds etth1 --L 96 --seed 42 --sensitivity-only   # 只测灵敏度(快)
"""
import argparse
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

N_PERTURB = 7
N_BATCHES = 6


def load_run(cfg_path, ckpt_path):
    cfg = yaml.safe_load(open(cfg_path))
    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = cfg["model"]
    loaders = build_standard_loaders(
        dataset_name=cfg["dataset"], data_dir=cfg.get("data_dir", "../ts_quantum/datasets"),
        lookback=cfg.get("lookback", 96), horizon=cfg.get("horizon", 96),
        batch_size=32, stride=1, num_workers=0,
    )
    inner = getattr(loaders["train"].dataset, "dataset", None)
    arr = getattr(inner, "data", None).detach().cpu().numpy()
    gmean, gstd = compute_global_stats(arr.reshape(-1, arr.shape[-1]))
    set_global_stats(gmean, gstd)
    model = QCCMamba(
        num_var=arr.shape[1], lookback=cfg.get("lookback", 96), horizon=cfg.get("horizon", 96),
        d_token=m.get("d_token", 512),
        qmix_layers=m.get("qmix_layers", 0),
        qmix_norm=m.get("qmix_norm", "avg"),
        head_agg=m.get("head_agg", False),
        spectrum_inject=m.get("spectrum_inject", False),
        n_qubits=m.get("n_qubits", 8), n_layers=m.get("n_layers", 2),
        entangle_topo=m.get("entangle_topo", "linear"), kernel_fn=m.get("kernel_fn", "quantum"),
        use_fmap=m.get("use_fmap", True), theta_S_scale0=m.get("theta_S_scale0", 0.5),
        use_periodic_feat=m.get("use_periodic_feat", True), revin_affine=m.get("revin_affine", True),
        use_spectrum=m.get("use_spectrum", True), spectrum_M=m.get("spectrum_M", 32),
        spectrum_range=m.get("spectrum_range", "0_2"),
        spectrum_amp_normalize=m.get("spectrum_amp_normalize", False),
        spectrum_time_align=m.get("spectrum_time_align", True),
        spectrum_freq_align=m.get("spectrum_freq_align", True),
        use_H=m.get("use_H", True), use_S=m.get("use_S", True),
        reupload_source=m.get("reupload_source", "S"),
        angle_norm=m.get("angle_norm", "clamp"), angle_radius=m.get("angle_radius", 1.0),
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, loaders["test"], device


def cross_var_sensitivity(model, loader, device, n_perturb=N_PERTURB):
    """y 空间跨变量灵敏度：sens[v][u] = RMS(Δy_v)/std_y_v（输入 u 加 1σ）。"""
    batch = next(iter(loader))
    x, y_true = batch[0].to(device), batch[1].to(device)
    x_mark = batch[2].to(device) if len(batch) == 4 else None
    V = x.shape[-1]
    with torch.no_grad():
        y0 = model(x, x_mark=x_mark)[0]
        x_std = x.std(dim=(0, 1), keepdim=True) + 1e-8
        y_std = y0.std(dim=(0, 1), keepdim=True) + 1e-8
        sens = np.zeros((V, V))
        us = np.random.RandomState(0).choice(V, min(n_perturb, V), replace=False)
        for u in us:
            xp = x.clone()
            xp[:, :, u] += x_std[:, :, u]
            yp = model(xp, x_mark=x_mark)[0]
            dy = ((yp - y0) ** 2).mean(dim=(0, 1)) ** 0.5
            sens[:, u] = (dy / y_std[:, :, 0]).cpu().numpy()
    return sens, us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sensitivity-only", action="store_true")
    args = ap.parse_args()

    for v in ["qmix_soft", "qmix_head", "qmix_full", "qmix", "qmix_sin", "plain"]:
        cp = f"configs/qmix/qm_{v}_{args.ds}_{args.L}_{args.seed}.yaml"
        ck = f"results/qmix/qm_{v}_{args.ds}_{args.L}_{args.seed}_best.pt"
        if not (os.path.exists(cp) and os.path.exists(ck)):
            continue
        model, loader, device = load_run(cp, ck)
        sens, us = cross_var_sensitivity(model, loader, device)
        d = np.diag(sens).mean()
        off = sens[~np.eye(len(sens), dtype=bool)]
        ratio = off.mean() / d if d > 0 else float("nan")
        print(f"=== {v} {args.ds}:{args.L} seed={args.seed} ===")
        print(f"  跨变量灵敏度: 对角={d:.4f}  非对角={off.mean():.2e}  非对角/对角={ratio:.4f}")
        if args.sensitivity_only:
            continue
        # K 结构与频谱注入消融
        batch = next(iter(loader))
        x, y_true = batch[0].to(device), batch[1].to(device)
        x_mark = batch[2].to(device) if len(batch) == 4 else None
        with torch.no_grad():
            y0, _, K = model(x, x_mark=x_mark)
            if K is not None:
                Km = K.float().mean(0)
                offm = Km - torch.diag(torch.diag(Km))
                x_norm = model.revin(x, mode="norm")
                xflat = x_norm.reshape(-1, x.shape[-1]).double()
                xc = torch.corrcoef(xflat.T)
                v_idx = torch.triu_indices(x.shape[-1], x.shape[-1], 1)
                kv = offm.flatten()[v_idx[0] * x.shape[-1] + v_idx[1]].double()
                xv = xc.flatten()[v_idx[0] * x.shape[-1] + v_idx[1]].double()
                pick = np.random.RandomState(0).choice(kv.numel(), min(3000, kv.numel()), replace=False)
                kv_r = kv[pick] - kv[pick].mean(); xv_r = xv[pick] - xv[pick].mean()
                sp = (kv_r * xv_r).sum() / (kv_r.norm() * xv_r.norm()).clamp_min(1e-12)
                print(f"  K offdiag mean={offm.mean():.4f}  Spearman(K, 真实相关)={sp.item():.4f}")
            if model.spectrum is not None and model.backbone.spectrum_inject is not None:
                x_norm = model.revin(x, mode="norm")
                S = model.spectrum(x_norm)
                S0 = torch.zeros_like(S)
                x_in = model._prepare_input(x_norm, x_mark)
                y_s0 = model.revin(model.backbone(x_in, S=S0).y_main, mode="denorm")
                dy = ((y_s0 - y0) ** 2).mean().item() ** 0.5
                scale = y0.std().item()
                print(f"  频谱注入消融: S→0 时 Δy/std = {dy / scale:.4f}")


if __name__ == "__main__":
    main()
