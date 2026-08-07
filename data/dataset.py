"""时间序列滑窗 Dataset。

支持：
- Electricity (ECL) 风格 CSV：第一列为时间戳，其余列为变量
- ETTh1/h2/m1/m2 风格 CSV：同 ECL，支持 12mo:4mo:4mo 划分
- ChinaAQI / METR_LA / PEMS_BAY：统一 CSV 格式（第一列时间戳）
- ILI：周频，L=H=24/36/48/60 特殊配置
- 自定义 numpy 数组

对应文档：IDEA_DualAE_QCC.md §5 / DATASET_SPEC_9.md / TSL_STANDARDS.md
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocess import add_time_features


@dataclass
class SplitConfig:
    """数据集划分配置。

    支持两种模式：
    1. 比例模式（默认）：train_ratio + test_ratio，验证 = 1 - train - test
    2. 月份模式（ETT 12mo:4mo:4mo）：use_months=True，按月划分
    """
    train_ratio: float = 0.7
    test_ratio: float = 0.2
    use_months: bool = False
    train_months: int = 12
    val_months: int = 4
    test_months: int = 4

    def split_lengths(self, total: int) -> Tuple[int, int, int]:
        """比例模式划分。"""
        if self.use_months:
            raise ValueError("use_months=True 时请用 split_by_months()")
        n_test = int(total * self.test_ratio)
        n_train = int(total * self.train_ratio)
        n_val = total - n_train - n_test
        return n_train, n_val, n_test

    def split_by_months(self, time_index: pd.DatetimeIndex) -> Tuple[int, int, int]:
        """月份模式划分（ETT 12mo:4mo:4mo）。

        返回 (n_train, n_val, n_test) 长度。
        """
        if not self.use_months:
            raise ValueError("use_months=False 时请用 split_lengths()")

        # 找到训练集结束点（前 train_months 个月）
        train_end = time_index[0] + pd.DateOffset(months=self.train_months)
        # 找到验证集结束点（再 val_months 个月）
        val_end = train_end + pd.DateOffset(months=self.val_months)

        n_train = (time_index < train_end).sum()
        n_val = ((time_index >= train_end) & (time_index < val_end)).sum()
        n_test = (time_index >= val_end).sum()

        return int(n_train), int(n_val), int(n_test)


# ---------------------------------------------------------------------- #
# 基础滑窗 Dataset
# ---------------------------------------------------------------------- #
class WindowedTimeSeriesDataset(Dataset):
    """滑窗 (L, H) 时间序列数据集。

    输入 x: (T, V)，输出 (B, L, V) 与 (B, H, V)。
    """

    def __init__(
        self,
        data: np.ndarray,  # (T, V) float32
        lookback: int,
        horizon: int,
        stride: int = 1,
    ):
        if data.ndim != 2:
            raise ValueError(f"data must be (T, V), got {data.shape}")
        self.data = torch.from_numpy(data).float()
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.T = data.shape[0]
        self.n_windows = max(0, (self.T - lookback - horizon) // stride + 1)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = idx * self.stride
        e = s + self.lookback
        x = self.data[s:e]                    # (L, V)
        y = self.data[e:e + self.horizon]    # (H, V)
        return x, y


# ---------------------------------------------------------------------- #
# 通用 Dataset：可同时返回 (x, y, x_mark, y_mark)
# ---------------------------------------------------------------------- #
class TimeSeriesDataset(Dataset):
    """TimeSeriesDataset：包含时间特征标记。"""

    def __init__(
        self,
        data: np.ndarray,            # (T, V)
        time_feats: np.ndarray,     # (T, F_t) — hour/dow
        lookback: int,
        horizon: int,
        stride: int = 1,
        with_time: bool = True,
    ):
        if len(data) != len(time_feats):
            raise ValueError("data and time_feats must have same length T")
        self.data = torch.from_numpy(data).float()
        self.time = torch.from_numpy(time_feats).float() if time_feats is not None else None
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.T = data.shape[0]
        self.with_time = with_time
        self.n_windows = max(0, (self.T - lookback - horizon) // stride + 1)

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx):
        s = idx * self.stride
        e = s + self.lookback
        x = self.data[s:e]
        y = self.data[e:e + self.horizon]
        if self.with_time and self.time is not None:
            x_mark = self.time[s:e]
            y_mark = self.time[e:e + self.horizon]
            return x, y, x_mark, y_mark
        return x, y


# ---------------------------------------------------------------------- #
# 文件级加载：CSV（第一列日期）
# ---------------------------------------------------------------------- #
def _load_csv_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 第一列为时间
    first = df.columns[0]
    if first.lower() in ("date", "datetime", "time", "timestamp"):
        df[first] = pd.to_datetime(df[first])
        df = df.set_index(first)
    return df


# ---------------------------------------------------------------------- #
# ECL 风格 Dataset（ChinaAQI / METR_LA / PEMS_BAY 统一格式）
# ---------------------------------------------------------------------- #
class ECLDataset(Dataset):
    """Electricity (ECL) / Traffic / Solar / ChinaAQI / METR_LA / PEMS_BAY 等：
    CSV 形式，多变量时间序列，第一列时间戳。

    支持比例划分（默认 7:1:2）或月份划分（use_months=True）。
    """

    def __init__(
        self,
        csv_path: str,
        lookback: int,
        horizon: int,
        split: str = "train",
        split_cfg: Optional[Union[SplitConfig, dict]] = None,
        stride: int = 1,
        freq: str = "h",
    ):
        df = _load_csv_dataframe(csv_path)
        data = df.values.astype(np.float32)  # (T, V)
        time_feats = add_time_features(df.index, freq=freq)  # (T, F)
        self.time_index = df.index

        # 兼容 dict 形式的 split 配置（来自 yaml）
        if isinstance(split_cfg, dict):
            split_cfg = SplitConfig(**split_cfg)
        split_cfg = split_cfg or SplitConfig()

        # 划分
        if split_cfg.use_months:
            n_train, n_val, n_test = split_cfg.split_by_months(df.index)
        else:
            n_train, n_val, n_test = split_cfg.split_lengths(len(data))

        if split == "train":
            data = data[:n_train]
            time_feats = time_feats[:n_train]
        elif split == "val":
            data = data[n_train:n_train + n_val]
            time_feats = time_feats[n_train:n_train + n_val]
            stride = 1  # 验证集密集采样
        elif split == "test":
            data = data[n_train + n_val:]
            time_feats = time_feats[n_train + n_val:]
            stride = 1  # 测试集密集采样
        else:
            raise ValueError(f"split must be train/val/test, got {split}")

        self.dataset = TimeSeriesDataset(
            data, time_feats, lookback, horizon, stride=stride, with_time=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# ---------------------------------------------------------------------- #
# ETT Dataset（支持 12mo:4mo:4mo 划分）
# ---------------------------------------------------------------------- #
class ETDataset(Dataset):
    """ETTh1/h2/m1/m2：与 ECL 同结构，freq 决定是否带 minute。

    默认使用 12mo:4mo:4mo 月份划分（TSL 标准）。
    """

    def __init__(
        self,
        csv_path: str,
        lookback: int,
        horizon: int,
        split: str = "train",
        split_cfg: Optional[Union[SplitConfig, dict]] = None,
        stride: int = 1,
        freq: str = "h",
    ):
        df = _load_csv_dataframe(csv_path)
        data = df.values.astype(np.float32)
        time_feats = add_time_features(df.index, freq=freq)

        # 兼容 dict 形式的 split 配置
        if isinstance(split_cfg, dict):
            split_cfg = SplitConfig(**split_cfg)

        # 默认使用 12mo:4mo:4mo 月份划分
        if split_cfg is None:
            split_cfg = SplitConfig(use_months=True, train_months=12, val_months=4, test_months=4)

        # 划分
        if split_cfg.use_months:
            n_train, n_val, n_test = split_cfg.split_by_months(df.index)
        else:
            n_train, n_val, n_test = split_cfg.split_lengths(len(data))

        if split == "train":
            data = data[:n_train]
            time_feats = time_feats[:n_train]
        elif split == "val":
            data = data[n_train:n_train + n_val]
            time_feats = time_feats[n_train:n_train + n_val]
            stride = 1
        elif split == "test":
            data = data[n_train + n_val:]
            time_feats = time_feats[n_train + n_val:]
            stride = 1
        else:
            raise ValueError(f"split must be train/val/test, got {split}")

        self.dataset = TimeSeriesDataset(
            data, time_feats, lookback, horizon, stride=stride, with_time=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# ---------------------------------------------------------------------- #
# ILI Dataset（周频，L=H=24/36/48/60 特殊配置）
# ---------------------------------------------------------------------- #
class ILIDataset(Dataset):
    """ILI（流感样病例）：周频数据，L=H=24/36/48/60。

    与 ECL 同结构，但 freq="W"（周频）。
    """

    def __init__(
        self,
        csv_path: str,
        lookback: int,
        horizon: int,
        split: str = "train",
        split_cfg: Optional[Union[SplitConfig, dict]] = None,
        stride: int = 1,
    ):
        df = _load_csv_dataframe(csv_path)
        data = df.values.astype(np.float32)
        time_feats = add_time_features(df.index, freq="W")  # 周频

        # 兼容 dict 形式的 split 配置
        if isinstance(split_cfg, dict):
            split_cfg = SplitConfig(**split_cfg)
        split_cfg = split_cfg or SplitConfig()

        # 划分
        if split_cfg.use_months:
            n_train, n_val, n_test = split_cfg.split_by_months(df.index)
        else:
            n_train, n_val, n_test = split_cfg.split_lengths(len(data))

        if split == "train":
            data = data[:n_train]
            time_feats = time_feats[:n_train]
        elif split == "val":
            data = data[n_train:n_train + n_val]
            time_feats = time_feats[n_train:n_train + n_val]
            stride = 1
        elif split == "test":
            data = data[n_train + n_val:]
            time_feats = time_feats[n_train + n_val:]
            stride = 1
        else:
            raise ValueError(f"split must be train/val/test, got {split}")

        self.dataset = TimeSeriesDataset(
            data, time_feats, lookback, horizon, stride=stride, with_time=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# ---------------------------------------------------------------------- #
# METR_LA / PEMS_BAY .h5 文件加载（转换为统一 CSV）
# ---------------------------------------------------------------------- #
def load_h5_to_csv(h5_path: str, csv_path: str, feature_idx: int = 0):
    """将 METR_LA / PEMS_BAY 的 .h5 文件转换为统一 CSV 格式。

    Args:
        h5_path: .h5 文件路径（包含 data 数组 (T, N, features)）。
        csv_path: 输出 CSV 路径。
        feature_idx: 取哪个特征通道（默认 0 = 速度）。
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("需要安装 h5py: pip install h5py")

    with h5py.File(h5_path, "r") as f:
        data = f["data"][:]  # (T, N, features)

    # 取指定特征通道
    data = data[:, :, feature_idx]  # (T, N)

    # 生成时间戳（假设从 2017-01-01 开始，5 分钟间隔）
    T = data.shape[0]
    time_index = pd.date_range(start="2017-01-01", periods=T, freq="5min")

    # 生成列名
    col_names = [f"sensor_{i}" for i in range(data.shape[1])]

    # 保存为 CSV
    df = pd.DataFrame(data, index=time_index, columns=col_names)
    df.index.name = "date"
    df.to_csv(csv_path)
    print(f"已转换 {h5_path} → {csv_path} ({T} 时间步, {data.shape[1]} 传感器)")


# ---------------------------------------------------------------------- #
# DataLoader 工厂
# ---------------------------------------------------------------------- #
def build_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )


__all__ = [
    "SplitConfig",
    "WindowedTimeSeriesDataset",
    "TimeSeriesDataset",
    "ECLDataset",
    "ETDataset",
    "ILIDataset",
    "load_h5_to_csv",
    "build_dataloader",
]
