#!/usr/bin/env python3
"""C1/C2 前提判别实验（2026-08-13）。

问题：跨变量信息（残差 59% 可被其他变量线性解释）到底能不能转化为 val 上的真实预测收益？
- C1 前提成立：可以 → 量子混合层有真实目标，值得继续砸
- C2 前提崩塌：不可以（信息不泛化）→ 所有运输补丁都是白费

方法：用已训 plain checkpoint（冻结，不训任何网络），拟合跨变量线性残差修正器
      r_v(t) ≈ ridge(其他变量的输入窗口)，在 val 上测修正后 MSE 是否真下降。

多 cell 覆盖（3 数据集 × 4 档位 × 双 seed 优先），单 cell 单 seed 不下结论。

用法：
  python premise_discriminator.py                      # 全部已有 plain checkpoint
  python premise_discriminator.py --ds etth1 --L 96    # 单 cell（调试用）
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from data.dataloader import build_standard_loaders
from model.qcc_mamba import QCCMamba
from engine.train import compute_global_stats, set_global_stats

MAX_BATCHES = 150   # 每 split 采样上限（4800 窗口）
RIDGE_LAMBDA = 1.0
V_FULLWINDOW_MAX = 21  # V ≤ 21 用全窗口特征，否则用末步值


def load_plain(ds, L, seed):
    cfg = yaml.safe_load(open(f"configs/qmix/qm_plain_{ds}_{L}_{seed}.yaml"))
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
        qmix_layers=m.get("qmix_layers", 0), qmix_norm=m.get("qmix_norm", "avg"),
        head_agg=m.get("head_agg", False), spectrum_inject=m.get("spectrum_inject", False),
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
    ck = torch.load(f"results/qmix/qm_plain_{ds}_{L}_{seed}_best.pt",
                    map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, loaders, device


@torch.no_grad()
def collect(model, loader, device, cap=MAX_BATCHES):
    """返回 X 原始全窗 (N,L,V)、X ReVIN 归一化全窗 (N,L,V)、残差 r (N,H,V)。"""
    xw, xn, rs = [], [], []
    for i, batch in enumerate(loader):
        if i >= cap:
            break
        x = batch[0].to(device)
        y = batch[1].to(device)
        x_mark = batch[2].to(device) if len(batch) == 4 else None
        y_hat = model(x, x_mark=x_mark)[0]
        xw.append(x.cpu())
        xn.append(model.revin(x, mode="norm").cpu())
        rs.append((y - y_hat).cpu())
    return (torch.cat(xw), torch.cat(xn), torch.cat(rs))


def ridge_fit(F, R, lam=RIDGE_LAMBDA):
    """多输出 ridge：beta (F_dim, H)。特征已标准化。"""
    FtF = F.T @ F
    reg = torch.eye(FtF.shape[0]) * lam
    return torch.linalg.solve(FtF + reg, F.T @ R)


def run_cell(ds, L, seed):
    model, loaders, device = load_plain(ds, L, seed)
    V = model.num_var if hasattr(model, "num_var") else None
    if V is None:
        x0 = next(iter(loaders["train"]))[0]
        V = x0.shape[-1]
    full_window = V <= V_FULLWINDOW_MAX

    Xw_tr, Xn_tr, R_tr = collect(model, loaders["train"], device)
    Xw_va, Xn_va, R_va = collect(model, loaders["val"], device)
    N, Lb, Vv = Xw_tr.shape
    H = R_tr.shape[1]

    # 特征构造：排除 v 的其他变量输入（对每个 v；Xw=原始, Xn=ReVIN 归一化）
    def make_feats(Xw, Xn):
        out = []
        for v in range(Vv):
            mask = torch.ones(Vv, dtype=bool)
            mask[v] = False
            if full_window:
                out.append(Xw[:, :, mask].reshape(Xw.shape[0], -1))
            else:
                out.append(Xn[:, -1, mask])   # 大 V 用归一化末步值
        return out

    F_tr_raw, F_va_raw = make_feats(Xw_tr, Xn_tr), make_feats(Xw_va, Xn_va)
    F_tr_nrm, F_va_nrm = make_feats(Xn_tr, Xn_tr), make_feats(Xn_va, Xn_va)

    # 对照 1：用 val 自己拟合（val 样本内），分离"无信号"与"不迁移"
    corr_va_in = torch.zeros_like(R_va)
    for v in range(Vv):
        mu, sd = F_va_nrm[v].mean(0), F_va_nrm[v].std(0).clamp_min(1e-8)
        beta = ridge_fit((F_va_nrm[v] - mu) / sd, R_va[:, :, v])
        corr_va_in[:, :, v] = ((F_va_nrm[v] - mu) / sd) @ beta

    # 对照 2：自身变量输入作特征（文档称相关 0.50 < 他变量 0.765，应复现该序）
    corr_own = torch.zeros_like(R_va)
    for v in range(Vv):
        f = Xw_tr[:, :, v].reshape(N, -1)
        mu, sd = f.mean(0), f.std(0).clamp_min(1e-8)
        beta = ridge_fit((f - mu) / sd, R_tr[:, :, v])
        fv = Xw_va[:, :, v].reshape(Xw_va.shape[0], -1)
        corr_own[:, :, v] = ((fv - mu) / sd) @ beta

    # 主实验（双特征域）：fit 域 → eval 域评估（eval 用 FIT 域统计量标准化）
    def transfer_r2(F_fit, F_eval, R_fit, R_eval):
        corr = torch.zeros_like(R_eval)
        for v in range(Vv):
            mu, sd = F_fit[v].mean(0), F_fit[v].std(0).clamp_min(1e-8)
            beta = ridge_fit((F_fit[v] - mu) / sd, R_fit[:, :, v])
            corr[:, :, v] = ((F_eval[v] - mu) / sd) @ beta
        mse_plain = (R_eval ** 2).mean().item()
        return 1 - ((R_eval - corr) ** 2).mean().item() / max(mse_plain, 1e-12)

    train_r2_raw = transfer_r2(F_tr_raw, F_tr_raw, R_tr, R_tr)  # 样本内（原始域）
    val_r2_raw = transfer_r2(F_tr_raw, F_va_raw, R_tr, R_va)    # 迁移（原始域）
    val_r2_nrm = transfer_r2(F_tr_nrm, F_va_nrm, R_tr, R_va)    # 迁移（ReVIN 域）

    mse_plain = (R_va ** 2).mean().item()
    mse_corr_in = ((R_va - corr_va_in) ** 2).mean().item()
    mse_own = ((R_va - corr_own) ** 2).mean().item()
    val_r2_in = 1 - mse_corr_in / max(mse_plain, 1e-12)
    own_r2 = 1 - mse_own / max(mse_plain, 1e-12)
    return dict(ds=ds, L=L, seed=seed, V=Vv, full=full_window, n_tr=N, n_va=Xw_va.shape[0],
                train_r2=train_r2_raw, val_r2=val_r2_raw, val_r2_nrm=val_r2_nrm,
                val_r2_in=val_r2_in, own_r2=own_r2, mse_plain=mse_plain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=None)
    ap.add_argument("--L", type=int, default=None)
    args = ap.parse_args()

    cells = []
    for ck in sorted(glob.glob("results/qmix/qm_plain_*_best.pt")):
        name = os.path.basename(ck).replace("qm_plain_", "").replace("_best.pt", "")
        ds, L, seed = name.rsplit("_", 2)
        if args.ds and ds != args.ds:
            continue
        if args.L and int(L) != args.L:
            continue
        cells.append((ds, int(L), int(seed)))

    print(f"共 {len(cells)} 个 plain checkpoint cell（判别实验，冻结模型 + 线性修正器）", flush=True)
    print(f"特征: {'全窗口' if False else '末步'}+全窗口(V≤{V_FULLWINDOW_MAX})  ridge λ={RIDGE_LAMBDA}", flush=True)
    print(f"{'cell':16s} V   n_tr  train_R²  valR²原始  valR²ReVIN  valR²样本内 own_R²", flush=True)

    rows = []
    for ds, L, seed in cells:
        t0 = time.time()
        try:
            r = run_cell(ds, L, seed)
        except Exception as e:
            print(f"{ds}:{L} seed={seed}  FAILED: {e}", flush=True)
            continue
        rows.append(r)
        print(f"{ds}:{L:<4d} s{seed:<4d} {r['V']:<3d} {r['n_tr']:<5d} "
              f"{r['train_r2']:.4f}  {r['val_r2']:+.4f}  {r['val_r2_nrm']:+.4f}  "
              f"{r['val_r2_in']:+.4f}  {r['own_r2']:+.4f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    if not rows:
        print("无结果"); return

    print("\n=== 汇总 ===")
    imp_raw = sum(1 for r in rows if r["val_r2"] > 0.01)
    imp_nrm = sum(1 for r in rows if r["val_r2_nrm"] > 0.01)
    print(f"迁移为正的 cell: 原始域 {imp_raw}/{len(rows)}  ReVIN域 {imp_nrm}/{len(rows)}")
    print(f"train_R² 均值 = {np.mean([r['train_r2'] for r in rows]):.4f}  "
          f"val_R²(原始域迁移) = {np.mean([r['val_r2'] for r in rows]):+.4f}  "
          f"val_R²(ReVIN域迁移) = {np.mean([r['val_r2_nrm'] for r in rows]):+.4f}  "
          f"val_R²(样本内) = {np.mean([r['val_r2_in'] for r in rows]):+.4f}  "
          f"own_R² = {np.mean([r['own_r2'] for r in rows]):+.4f}")
    verdict = "ReVIN域前提成立" if imp_nrm > len(rows) * 0.6 else "前提不成立/需自适应"
    print(f"\n判别: {verdict}  (判据: ReVIN 域迁移为正的 cell > 60%)")


if __name__ == "__main__":
    main()
