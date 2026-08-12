"""DataLoader 一站式工厂：按 dataset 名快速构造 train/val/test loader。

支持：
- 原始数据集：electricity / etth1 / etth2 / ettm1 / ettm2 / traffic / weather / solar / exchange
- 新增数据集：chinaaqi / metr_la / pems_bay / ili

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
    ILIDataset,
    SplitConfig,
    build_dataloader,
)

# 数据集文件名映射
DATASET_FILES = {
    # 原始数据集
    "electricity": "electricity.csv",
    "etth1": "ETTh1.csv",
    "etth2": "ETTh2.csv",
    "ettm1": "ETTm1.csv",
    "ettm2": "ETTm2.csv",
    "traffic": "traffic.csv",
    "weather": "weather.csv",
    "exchange": "exchange_rate.csv",
    "solar": "solar.csv",
    # 新增数据集（build 脚本产出带下划线文件名）
    "chinaaqi": "china_aqi.csv",
    "metr_la": "metr_la.csv",
    "pems_bay": "pems_bay.csv",
    "ili": "national_illness.csv",
}

# 数据集频率映射（影响 add_time_features）
DATASET_FREQ = {
    # 小时频
    "electricity": "h",
    "traffic": "h",
    "chinaaqi": "h",
    "etth1": "h",
    "etth2": "h",
    # 15 分钟频（ETTm 官方为 5 列含 minute//15）
    "ettm1": "t",
    "ettm2": "t",
    # 10 分钟频
    "weather": "t",
    "solar": "t",
    # 5 分钟频（METR_LA/PEMS_BAY 转换后）
    "metr_la": "5min",
    "pems_bay": "5min",
    # 日频
    "exchange": "d",
    # 周频
    "ili": "W",
}

# 数据集类别映射
DATASET_CLASS = {
    # ECL 风格（CSV，第一列时间戳）
    "electricity": ECLDataset,
    "traffic": ECLDataset,
    "weather": ECLDataset,
    "solar": ECLDataset,
    "exchange": ECLDataset,
    "chinaaqi": ECLDataset,
    "metr_la": ECLDataset,
    "pems_bay": ECLDataset,
    # ETT 风格（12mo:4mo:4mo 划分）
    "etth1": ETDataset,
    "etth2": ETDataset,
    "ettm1": ETDataset,
    "ettm2": ETDataset,
    # ILI 风格（周频）
    "ili": ILIDataset,
}


def _default_dataset_dir() -> str:
    """默认数据集目录：qcc_mamba/../ts_quantum/datasets。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(here, "..", "..", "ts_quantum", "datasets"))
    return candidate if os.path.isdir(candidate) else os.getcwd()


def build_standard_loaders(
    dataset_name: str = "electricity",
    lookback: int = 720,
    horizon: int = 96,
    batch_size: int = 32,
    stride: int = 1,
    num_workers: int = 0,
    data_dir: Optional[str] = None,
    split_cfg: Optional[SplitConfig] = None,
) -> Dict[str, DataLoader]:
    """构造 train/val/test DataLoader（统一接口）。

    Args:
        dataset_name: 数据集名（见 DATASET_FILES）。
        lookback: 输入窗口 L。
        horizon: 预测步长 H。
        batch_size: 批大小（大 V 数据集建议 8-16 + 梯度累积）。
        stride: 滑窗步长。
        num_workers: DataLoader worker 数。
        data_dir: 显式数据集目录。
        split_cfg: 划分配置（ETT 默认 12mo:4mo:4mo，其他默认 7:1:2）。

    Returns:
        {"train": DL, "val": DL, "test": DL}
    """
    name = dataset_name.lower()
    if name not in DATASET_FILES:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_FILES.keys())}")
    fname = DATASET_FILES[name]
    d = data_dir or _default_dataset_dir()
    csv_path = os.path.join(d, fname)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    # 兼容：从 yaml 传进来的 split 配置可能是 dict，自动包成 SplitConfig
    if isinstance(split_cfg, dict):
        split_cfg = SplitConfig(**split_cfg)

    # 获取数据集类别和频率
    ds_cls = DATASET_CLASS[name]
    freq = DATASET_FREQ.get(name, "h")

    # 构造通用参数
    common = dict(
        csv_path=csv_path,
        lookback=lookback,
        horizon=horizon,
        stride=stride,
    )

    # 根据数据集类型添加特定参数
    if ds_cls == ETDataset:
        # ETT 默认 12mo:4mo:4mo
        if split_cfg is None:
            split_cfg = SplitConfig(use_months=True, train_months=12, val_months=4, test_months=4)
        kwargs = dict(freq=freq, split_cfg=split_cfg)
    elif ds_cls == ILIDataset:
        # ILI 周频
        kwargs = dict(split_cfg=split_cfg)
    else:
        # ECL 风格
        kwargs = dict(freq=freq, split_cfg=split_cfg)

    # 构造三个数据集
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


# 向后兼容别名
build_e1_loaders = build_standard_loaders


__all__ = ["build_standard_loaders", "build_e1_loaders", "DATASET_FILES", "DATASET_FREQ", "DATASET_CLASS"]
