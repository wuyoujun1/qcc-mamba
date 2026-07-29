"""标准 benchmark / 超长序列实验一键运行。

支持 E2（标准 L=96/192/336/720）和 E3（超长 L=1440/8760/17520）。

用法：
    PYTHONPATH=. python run_benchmark.py --config configs/e2_standard.yaml --gpu 0
    PYTHONPATH=. python run_benchmark.py --config configs/e3_longterm.yaml --gpu 0
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data.dataloader import build_e1_loaders, DATASET_FILES
from data.dataset import SplitConfig
from engine.evaluate import metric_table
from engine.train import fit, set_global_stats, compute_global_stats
from model.qcc_mamba import QCCMamba
from qcc import make_kernel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, device: torch.device, seed: int = 0) -> QCCMamba:
    """根据配置构造 QCCMamba（E2/E3 只跑一个 method）."""
    mcfg = cfg["model"]
    tcfg = cfg["training"]
    method_cfg = cfg.get("method", {})

    if method_cfg.get("use_qcc", True):
        kernel_name = method_cfg.get("kernel", "quantum")
        kernel_fn = None
        if kernel_name not in ("quantum", "none"):
            kernel_fn = make_kernel(
                kernel_name,
                d_token=mcfg["d_token"],
                D_rff=method_cfg.get("D_rff", 256),
                cache_key=f"{kernel_name}_{method_cfg.get('D_rff', 256)}",
                seed=seed,
            )
        model = QCCMamba(
            num_var=mcfg["num_var"],
            lookback=cfg["lookback"],
            horizon=cfg["horizon"],
            d_token=mcfg["d_token"],
            use_qcc=True,
            n_qubits=method_cfg.get("n_qubits", 8),
            n_layers=method_cfg.get("n_layers", 2),
            entangle_topo=method_cfg.get("entangle_topo", "linear"),
            encode_gate=method_cfg.get("encode_gate", "R_Y"),
            kernel_fn=kernel_fn,
            use_fmap=method_cfg.get("use_fmap", True),
            alpha0=tcfg["alpha0"],
            beta=tcfg["beta"],
            use_periodic_feat=mcfg.get("use_periodic_feat", True),
            revin_affine=mcfg.get("revin_affine", True),
        )
    else:
        # QCCMamba 内部已根据 use_qcc=False / bond_dim 构造 MPSBypass，不再覆盖
        model = QCCMamba(
            num_var=mcfg["num_var"],
            lookback=cfg["lookback"],
            horizon=cfg["horizon"],
            d_token=mcfg["d_token"],
            use_qcc=False,
            bond_dim=method_cfg.get("bond_dim", 8),
            alpha0=tcfg["alpha0"],
            beta=tcfg["beta"],
            use_periodic_feat=mcfg.get("use_periodic_feat", True),
            revin_affine=mcfg.get("revin_affine", True),
        )

    return model.to(device)


def run_single_setting(cfg: dict, lookback: int, horizon: int, seed: int, device: torch.device):
    """跑一个 (lookback, horizon, seed) 设置。"""
    set_seed(seed)
    cfg = dict(cfg)
    cfg["lookback"] = lookback
    cfg["horizon"] = horizon

    loaders = build_e1_loaders(
        dataset_name=cfg["dataset"],
        lookback=lookback,
        horizon=horizon,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        split_cfg=cfg.get("split"),
    )
    model = build_model(cfg, device, seed=seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )

    experiment = cfg.get("experiment", "benchmark")
    method_cfg = cfg.get("method", {})
    if method_cfg.get("use_qcc", True):
        method_tag = method_cfg.get("kernel", "qcc")
    else:
        method_tag = "mps"
    run_name = f"{experiment}_{cfg['dataset']}_L{lookback}H{horizon}_{method_tag}_s{seed}"
    history = fit(
        model,
        loaders,
        optimizer,
        scheduler=scheduler,
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"]["patience"],
        device=device,
        save_dir="checkpoints",
        run_name=run_name,
    )
    mse_norm = history.get("test_mse_norm", [None])[-1]
    mae_norm = history.get("test_mae_norm", [None])[-1]
    return history["test_mse"][-1], history["test_mae"][-1], mse_norm, mae_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="benchmark 配置文件")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", default="results", help="结果输出目录")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.out, exist_ok=True)
    experiment_name = cfg.get("experiment", "benchmark")
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Dataset: {cfg['dataset']}, Method: {cfg['method']}")
    print(f"{'='*60}")

    # 计算全局每变量标准化统计量（论文标准做法）
    ds_name = cfg["dataset"].lower()
    fname = DATASET_FILES.get(ds_name, f"{ds_name}.csv")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ts_quantum", "datasets")
    csv_path = os.path.join(data_dir, fname)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        raw = df.iloc[:, 1:].values.astype(np.float32)
        split_cfg_raw = cfg.get("split", SplitConfig())
        n_train = int(len(raw) * split_cfg_raw.get("train_ratio", 0.7))
        train_raw = raw[:n_train]
        gm, gs = compute_global_stats(train_raw)
        set_global_stats(gm, gs)
        print(f"全局标准化统计量: {train_raw.shape[0]} 个训练样本, "
              f"{train_raw.shape[1]} 变量, 平均std={train_raw.std(axis=0).mean():.2f}")
    else:
        print(f"⚠️ 未找到数据集 {csv_path}，不计算归一化 MSE")

    seeds = [cfg["training"]["seed_base"] + i for i in range(cfg["training"].get("n_seeds", 3))]
    rows = []
    for lookback, horizon in cfg["settings"]:
        print(f"\n>>> Setting: L={lookback}, H={horizon}")
        mse_list, mae_list, mn_list, man_list = [], [], [], []
        for seed in seeds:
            out = run_single_setting(cfg, lookback, horizon, seed, device)
            if len(out) == 4:
                mse, mae, mse_norm, mae_norm = out
            else:
                mse, mae = out
                mse_norm, mae_norm = None, None
            mse_list.append(mse)
            mae_list.append(mae)
            mn_list.append(mse_norm)
            man_list.append(mae_norm)
            msg = f"  seed {seed}: MSE={mse:.6f}, MAE={mae:.6f}"
            if mse_norm is not None:
                msg += f", MSE_norm={mse_norm:.6f}"
            if mae_norm is not None:
                msg += f", MAE_norm={mae_norm:.6f}"
            print(msg)
        r = {
            "lookback": lookback,
            "horizon": horizon,
            "mse_mean": float(np.mean(mse_list)),
            "mse_std": float(np.std(mse_list)),
            "mae_mean": float(np.mean(mae_list)),
            "mae_std": float(np.std(mae_list)),
        }
        valid_mn = [x for x in mn_list if x is not None]
        valid_man = [x for x in man_list if x is not None]
        if valid_mn:
            r["mse_norm"] = float(np.mean(valid_mn))
        if valid_man:
            r["mae_norm"] = float(np.mean(valid_man))
        rows.append(r)

    df = pd.DataFrame(rows)
    out_path = os.path.join(args.out, f"{experiment_name}_{cfg['dataset']}.csv")
    df.to_csv(out_path, index=False)
    print("\n" + "=" * 60)
    print("Benchmark 结果")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
