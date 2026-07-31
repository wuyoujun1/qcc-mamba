"""一键运行 E1 决定性实验。

用法：
    python run_e1.py --config configs/e1_kernel_decisive.yaml --gpu 0
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
from qcc.mps_kernel import MPSBypass


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, method_cfg: dict, device: torch.device, seed: int = 0) -> QCCMamba:
    """根据 method 配置构造模型。"""
    mcfg = cfg["model"]
    if method_cfg.get("use_qcc", True):
        kernel_name = method_cfg.get("kernel", "quantum")
        if kernel_name == "quantum":
            kernel_fn = None  # QCCBlock 默认 quantum_kernel
        else:
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
            alpha0=cfg["training"]["alpha0"],
            beta=cfg["training"]["beta"],
            use_periodic_feat=mcfg.get("use_periodic_feat", True),
            revin_affine=mcfg.get("revin_affine", True),
        )
    else:
        # MPS bypass
        model = QCCMamba(
            num_var=mcfg["num_var"],
            lookback=cfg["lookback"],
            horizon=cfg["horizon"],
            d_token=mcfg["d_token"],
            use_qcc=False,
            bond_dim=method_cfg.get("bond_dim", 8),
            kernel_type=method_cfg.get("kernel_type", "rbf"),
            alpha0=cfg["training"]["alpha0"],
            beta=cfg["training"]["beta"],
            use_periodic_feat=mcfg.get("use_periodic_feat", True),
            revin_affine=mcfg.get("revin_affine", True),
        )

    return model.to(device)


def run_method(cfg: dict, method_name: str, method_cfg: dict, seed: int, device: torch.device):
    """跑一个 method 的一个 seed。"""
    set_seed(seed)
    loaders = build_e1_loaders(
        dataset_name=cfg["dataset"],
        lookback=cfg["lookback"],
        horizon=cfg["horizon"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    model = build_model(cfg, method_cfg, device, seed=seed)

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

    # 论文标准：仅报告 MSE_norm（全局每变量标准化）
    # 不报告 raw MSE（跨变量尺度大，不具可比性）
    norm_values = history.get("test_mse_norm", [])
    if norm_values and norm_values[-1] is not None and norm_values[-1] != float("inf"):
        return norm_values[-1]
    else:
        # fallback: 若全局统计量没设，也返回 raw
        return history["test_mse"][-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e1_kernel_decisive.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=None, help="只跑指定方法")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 预计算全局每变量标准化统计量（论文标准 MSE_norm）
    ds_name = cfg["dataset"].lower()
    fname = DATASET_FILES.get(ds_name, f"{ds_name}.csv")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ts_quantum", "datasets")
    csv_path = os.path.join(data_dir, fname)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        raw = df.iloc[:, 1:].values.astype(np.float32)
        sc = cfg.get("split", SplitConfig(train_ratio=0.7))
        if isinstance(sc, dict):
            sc = SplitConfig(**sc)
        n_train = int(len(raw) * sc.train_ratio)
        train_raw = raw[:n_train]
        gm, gs = compute_global_stats(train_raw)
        set_global_stats(gm, gs)
        print(f"全局标准化统计量: {train_raw.shape[0]} 训练样本, "
              f"{train_raw.shape[1]} 变量, mean_std={gs.mean():.2f}")
    else:
        print(f"⚠️ 数据集 {csv_path} 未找到，无法计算 MSE_norm")

    methods = cfg["methods"]
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in args.methods}

    seeds = [cfg["training"]["seed_base"] + i for i in range(cfg["training"].get("n_seeds", 1))]
    results = {name: [] for name in methods}

    for method_name, method_cfg in methods.items():
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")
        for seed in seeds:
            mse_norm = run_method(cfg, method_name, method_cfg, seed, device)
            results[method_name].append(mse_norm)
            print(f"  seed {seed}: MSE_norm={mse_norm:.6f}")

    print("\n" + "=" * 60)
    print("E1 决定性实验结果 (MSE_norm ↓)")
    print("=" * 60)
    print(metric_table(results, baseline_name="none"))


if __name__ == "__main__":
    main()
