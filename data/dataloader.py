"""DataLoader 一站式工厂：按 dataset 名快速构造 train/val/test loader。

支持：electricity / etth1 / etth2 / ettm1 / ettm2 / traffic / weather / solar / exchange

数据集默认从项目根 ../ts_quantum/datasets/ 读取（保持与现有仓库一致）。
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .dataset import (
    ECLDataset,
    ETDataset,
    SplitConfig,
    build_dataloader,
)

# 数据集文件名映射
DATASET_FILES = {
    "electricity": "electricity.csv",
    "etth1": "ETTh1.csv",
    "etth2": "ETTh2.csv",
    "ettm1": "ETTm1.csv",
    "ettm2": "ETTm2.csv",
    "traffic": "traffic.csv",
    "weather": "weather.csv",
    "exchange": "exchange_rate.csv",
    "solar": "solar.csv",
}


def _default_dataset_dir() -> str:
    """默认数据集目录：qcc_mamba/../ts_quantum/datasets。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(here, "..", "..", "ts_quantum", "datasets"))
    return candidate if os.path.isdir(candidate) else os.getcwd()


def build_e1_loaders(
    dataset_name: str = "electricity",
    lookback: int = 720,
    horizon: int = 96,
    batch_size: int = 32,
    stride: int = 1,
    num_workers: int = 0,
    data_dir: Optional[str] = None,
    split_cfg: Optional[SplitConfig] = None,
) -> Dict[str, DataLoader]:
    """为 E1（决定性实验）构造 train/val/test DataLoader。

    Args:
        dataset_name: 数据集名（见 DATASET_FILES）。
        lookback: 输入窗口 L。
        horizon: 预测步长 H。
        batch_size: 批大小（L=8760 时建议 8-16 + 梯度累积）。
        stride: 滑窗步长。
        num_workers: DataLoader worker 数。
        data_dir: 显式数据集目录。
        split_cfg: 划分配置。

    Returns:
        {"train": DL, "val": DL, "test": DL}
    """
    name = dataset_name.lower()
    if name not in DATASET_FILES:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    fname = DATASET_FILES[name]
    d = data_dir or _default_dataset_dir()
    csv_path = os.path.join(d, fname)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    # 兼容：从 yaml 传进来的 split 配置可能是 dict，自动包成 SplitConfig
    if isinstance(split_cfg, dict):
        split_cfg = SplitConfig(**split_cfg)
    sc = split_cfg or SplitConfig(train_ratio=0.7, test_ratio=0.2)
    # ECL/Traffic/Weather/Solar/Exchange 用 ECL 格式；ETT 系列用 ET
    if name.startswith("ett"):
        ds_cls = ETDataset
        kwargs = dict(freq="h" if name.startswith("etth") else "t")
    else:
        ds_cls = ECLDataset
        kwargs = dict()
    common = dict(
        csv_path=csv_path,
        lookback=lookback,
        horizon=horizon,
        split_cfg=sc,
        stride=stride,
    )
    train_ds = ds_cls(split="train", **{**common, **kwargs})
    val_ds = ds_cls(split="val", **{**common, **kwargs})
    test_ds = ds_cls(split="test", **{**common, **kwargs})
    return {
        "train": build_dataloader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, drop_last=True),
        "val":   build_dataloader(val_ds,   batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, drop_last=False),
        "test":  build_dataloader(test_ds,  batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, drop_last=False),
    }


__all__ = ["build_e1_loaders", "DATASET_FILES"]
