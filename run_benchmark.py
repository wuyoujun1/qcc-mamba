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

from data.dataloader import build_e1_loaders
from data.dataset import SplitConfig
from engine.evaluate import metric_table
from engine.train import fit
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

    history = fit(
        model,
        loaders,
        optimizer,
        scheduler=scheduler,
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"]["patience"],
        device=device,
    )
    return history["test_mse"][-1], history["test_mae"][-1]


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

    seeds = [cfg["training"]["seed_base"] + i for i in range(cfg["training"].get("n_seeds", 3))]
    rows = []
    for lookback, horizon in cfg["settings"]:
        print(f"\n>>> Setting: L={lookback}, H={horizon}")
        mse_list, mae_list = [], []
        for seed in seeds:
            mse, mae = run_single_setting(cfg, lookback, horizon, seed, device)
            mse_list.append(mse)
            mae_list.append(mae)
            print(f"  seed {seed}: MSE={mse:.6f}, MAE={mae:.6f}")
        rows.append({
            "lookback": lookback,
            "horizon": horizon,
            "mse_mean": float(np.mean(mse_list)),
            "mse_std": float(np.std(mse_list)),
            "mae_mean": float(np.mean(mae_list)),
            "mae_std": float(np.std(mae_list)),
        })

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
