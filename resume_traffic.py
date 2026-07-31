"""E2 Traffic 基线断点续训脚本。
从 CPU 跑了一半的 checkpoint 恢复，在 GPU 上跑完。
"""
from __future__ import annotations

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataloader import build_e1_loaders
from engine.train import fit, set_global_stats, compute_global_stats


def build_model(cfg, device):
    mcfg = cfg["model"]
    tcfg = cfg["training"]
    method_cfg = cfg.get("method", {})

    from model.qcc_mamba import QCCMamba

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
        kernel_fn=None,
        use_fmap=method_cfg.get("use_fmap", False),
        alpha0=tcfg["alpha0"],
        beta=tcfg["beta"],
        use_periodic_feat=mcfg.get("use_periodic_feat", True),
        revin_affine=mcfg.get("revin_affine", True),
    )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 全局标准化统计量
    ds_name = cfg["dataset"].lower()
    fname = {"traffic": "traffic.csv"}.get(ds_name, f"{ds_name}.csv")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ts_quantum", "datasets")
    csv_path = os.path.join(data_dir, fname)
    df = pd.read_csv(csv_path)
    raw = df.iloc[:, 1:].values.astype(np.float32)
    split_raw = cfg.get("split", {})
    n_train = int(len(raw) * split_raw.get("train_ratio", 0.7))
    train_raw = raw[:n_train]
    gm, gs = compute_global_stats(train_raw)
    set_global_stats(gm, gs)
    print(f"全局统计量: {train_raw.shape[0]} 样本, {train_raw.shape[1]} 变量")

    seed = cfg["training"]["seed_base"]
    method_tag = "none"  # config 中 kernel: none

    for lookback, horizon in cfg["settings"]:
        cfg["lookback"] = lookback
        cfg["horizon"] = horizon

        run_name = f"{cfg['experiment']}_{cfg['dataset']}_L{lookback}H{horizon}_{method_tag}_s{seed}"
        ckpt_path = os.path.join("checkpoints", f"{run_name}_best.pt")

        print(f"\n>>> L={lookback}, H={horizon}")
        print(f"   Checkpoint: {ckpt_path}")

        if not os.path.exists(ckpt_path):
            print(f"   ⚠️ 没有 checkpoint，跳过")
            continue

        # 加载数据
        loaders = build_e1_loaders(
            dataset_name=cfg["dataset"],
            lookback=lookback,
            horizon=horizon,
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            split_cfg=cfg.get("split"),
        )
        print(f"   数据加载完成: {len(loaders['train'])} batches")

        # 构建模型
        model = build_model(cfg, device)
        print(f"   模型构建完成")

        # 加载 checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        initial_epoch = ckpt["epoch"]
        print(f"   🔄 恢复自 epoch {initial_epoch} (val_mse={ckpt.get('val_mse', '?'):.6f})")

        # 优化器 + 调度器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"]
        )

        # 训练
        print(f"   开始训练 ({cfg['training']['epochs'] - initial_epoch} epochs)...")
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
            initial_epoch=initial_epoch,
        )

        # 结果
        best_idx = int(np.argmin(history["test_mse"]))
        print(f"   ✅ 完成! 最佳 test_mse = {history['test_mse'][best_idx]:.6f} "
              f"(epoch {best_idx + 1})")

    print("\n全部完成!")


if __name__ == "__main__":
    main()
