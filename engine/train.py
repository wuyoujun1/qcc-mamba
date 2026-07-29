"""训练循环。

提供：
- train_one_epoch：一个 epoch 的训练。
- evaluate：在 val/test 上评估，返回 MSE/MAE。
- fit：完整训练 + early stopping + lr scheduling。

对应文档：experiment-design.md §八
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# 全局标准化统计量（在 dataset 上预计算，供 evaluate 使用）
# 论文标准做法：对整个训练集按变量求 mean/std，用固定值标准化
_GLOBAL_STATS_CACHE: dict = {}


def set_global_stats(mean: torch.Tensor, std: torch.Tensor):
    """设置全局标准化参数（每个变量在整个训练集上的 mean/std）。"""
    _GLOBAL_STATS_CACHE["mean"] = mean
    _GLOBAL_STATS_CACHE["std"] = std


def compute_global_stats(data: np.ndarray) -> tuple:
    """从 numpy 数据计算每变量 mean/std。"""
    mean = torch.from_numpy(data.mean(axis=0)).float()
    std = torch.from_numpy(data.std(axis=0)).float()
    return mean, std


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
    """评估，返回 MSE 与 MAE（反归一化空间）。

    额外返回 mse_norm：论文标准归一化 MSE（全局每变量标准化），
    值域 0.1~0.3（electricity 数据集），可与其他论文直接对比。
    """
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n_samples = 0
    mse_norm_sum = 0.0
    mae_norm_sum = 0.0
    n_norm = 0

    # 取全局标准化参数（预计算）
    global_mean = _GLOBAL_STATS_CACHE.get("mean", None)
    global_std = _GLOBAL_STATS_CACHE.get("std", None)

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

        # 论文标准归一化 MSE/MAE：用全局每变量 mean/std 标准化
        if global_mean is not None and global_std is not None:
            gs = global_std.to(device).clamp(min=1e-5)
            gm = global_mean.to(device)
            y_pred_norm = (y_pred - gm) / gs
            y_true_norm = (y_true - gm) / gs
            mse_norm_sum += nn.functional.mse_loss(y_pred_norm, y_true_norm, reduction="sum").item()
            mae_norm_sum += torch.abs(y_pred_norm - y_true_norm).sum().item()
            n_norm += y_true.numel()

    if n_samples == 0:
        return {"mse": float("inf"), "mae": float("inf"), "rmse": float("inf")}

    result = {
        "mse": mse_sum / n_samples,
        "mae": mae_sum / n_samples,
        "rmse": (mse_sum / n_samples) ** 0.5,
    }
    if n_norm > 0:
        result["mse_norm"] = mse_norm_sum / n_norm
        result["mae_norm"] = mae_norm_sum / n_norm
    return result


def fit(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epochs: int = 100,
    patience: int = 10,
    device: torch.device = torch.device("cpu"),
    save_dir: Optional[str] = None,
    run_name: str = "model",
) -> Dict[str, list]:
    """完整训练 + early stopping + checkpoint。

    Args:
        save_dir: 最佳模型保存目录（传 None 则不保存）。
        run_name: checkpoint 文件名前缀。

    Returns:
        history: {"train_loss": [...], "val_mse": [...], "test_mse": [...]}
    """
    best_val = float("inf")
    wait = 0
    history = {"train_loss": [], "val_mse": [], "test_mse": [],
               "val_mae": [], "test_mae": [],
               "val_mse_norm": [], "test_mse_norm": [],
               "val_mae_norm": [], "test_mae_norm": []}
    best_state = None

    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, device)
        val_metrics = evaluate(model, loaders["val"], device)
        test_metrics = evaluate(model, loaders["test"], device)

        history["train_loss"].append(train_metrics["loss"])
        history["val_mse"].append(val_metrics["mse"])
        history["test_mse"].append(test_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])
        history["test_mae"].append(test_metrics["mae"])
        history["val_mse_norm"].append(val_metrics.get("mse_norm", float("inf")))
        history["test_mse_norm"].append(test_metrics.get("mse_norm", float("inf")))
        history["val_mae_norm"].append(val_metrics.get("mae_norm", float("inf")))
        history["test_mae_norm"].append(test_metrics.get("mae_norm", float("inf")))

        # 记录旁路权重 α（QCCBlock 有 alpha 参数）
        alpha_val = None
        if hasattr(model, 'qcc') and hasattr(model.qcc, 'alpha'):
            alpha_val = model.qcc.alpha.item()
        history.setdefault("alpha", []).append(alpha_val)

        if scheduler is not None:
            scheduler.step()

        mn = test_metrics.get("mse_norm")
        man = test_metrics.get("mae_norm")
        extra = ""
        if mn is not None:
            extra += f"  MSE_norm={mn:.6f}"
        if man is not None:
            extra += f"  MAE_norm={man:.6f}"
        if alpha_val is not None:
            extra += f"  α={alpha_val:.4f}"
        print(
            f"Epoch {epoch + 1}/{epochs}  "
            f"train_loss={train_metrics['loss']:.6f}  "
            f"val_mse={val_metrics['mse']:.6f}  "
            f"test_mse={test_metrics['mse']:.6f}{extra}"
        )

        # 验证集早停 + 保存最佳 checkpoint
        val_ok = val_metrics["mse"] != float("inf")
        if val_ok and val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            wait = 0
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                ckpt_path = os.path.join(save_dir, f"{run_name}_best.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "val_mse": best_val,
                        "test_mse": test_metrics["mse"],
                        "mse_norm": test_metrics.get("mse_norm", None),
                    },
                    ckpt_path,
                )
                print(f"  ✅ 保存最佳模型: {ckpt_path} (epoch {epoch+1}, val_mse={best_val:.4f})")
        elif val_ok:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 无论早停与否，最终再保存一次最佳
    if save_dir:
        print(f"  最佳 val_mse = {best_val:.4f}, checkpoint 已保存")

    return history


__all__ = ["train_one_epoch", "evaluate", "fit"]
