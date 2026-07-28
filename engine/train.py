"""训练循环。

提供：
- train_one_epoch：一个 epoch 的训练。
- evaluate：在 val/test 上评估，返回 MSE/MAE。
- fit：完整训练 + early stopping + lr scheduling。

对应文档：experiment-design.md §八
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """训练一个 epoch。

    假设 model 是 QCCMamba，数据项为 (x, y, x_mark, y_mark) 或 (x, y)。
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        if len(batch) == 4:
            x, y_true, x_mark, _ = batch
            x_mark = x_mark.to(device)
        else:
            x, y_true = batch
            x_mark = None

        x = x.to(device)
        y_true = y_true.to(device)

        optimizer.zero_grad()

        # 前向（返回归一化空间用于损失）
        y_pred, y_main, _, y_norm, y_main_norm, correction_norm = model(
            x, x_mark=x_mark, return_norm=True
        )

        # 目标也需要在归一化空间，但必须使用 x 的 RevIN 统计量（与 y_norm 一致）
        with torch.no_grad():
            x_mean = model.revin.mean
            x_stdev = model.revin.stdev
            y_true_norm = (y_true - x_mean) / x_stdev
            if model.revin.affine:
                y_true_norm = y_true_norm * model.revin.affine_weight + model.revin.affine_bias

        loss = model.compute_loss(
            y_pred, y_main, y_true, y_norm, y_main_norm, y_true_norm, correction_norm
        )
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """评估，返回 MSE 与 MAE（反归一化空间）。"""
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n_samples = 0

    for batch in loader:
        if len(batch) == 4:
            x, y_true, x_mark, _ = batch
            x_mark = x_mark.to(device)
        else:
            x, y_true = batch
            x_mark = None

        x = x.to(device)
        y_true = y_true.to(device)

        y_pred, _, _ = model(x, x_mark=x_mark, return_norm=False)

        mse = nn.functional.mse_loss(y_pred, y_true, reduction="sum")
        mae = torch.abs(y_pred - y_true).sum()

        mse_sum += mse.item()
        mae_sum += mae.item()
        n_samples += y_true.numel()

    return {
        "mse": mse_sum / n_samples,
        "mae": mae_sum / n_samples,
        "rmse": (mse_sum / n_samples) ** 0.5,
    }


def fit(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epochs: int = 100,
    patience: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, list]:
    """完整训练 + early stopping。

    Returns:
        history: {"train_loss": [...], "val_mse": [...], "test_mse": [...]}
    """
    best_val = float("inf")
    wait = 0
    history = {"train_loss": [], "val_mse": [], "test_mse": []}

    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, device)
        val_metrics = evaluate(model, loaders["val"], device)
        test_metrics = evaluate(model, loaders["test"], device)

        history["train_loss"].append(train_metrics["loss"])
        history["val_mse"].append(val_metrics["mse"])
        history["test_mse"].append(test_metrics["mse"])

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch + 1}/{epochs}  "
            f"train_loss={train_metrics['loss']:.6f}  "
            f"val_mse={val_metrics['mse']:.6f}  "
            f"test_mse={test_metrics['mse']:.6f}"
        )

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return history


__all__ = ["train_one_epoch", "evaluate", "fit"]
