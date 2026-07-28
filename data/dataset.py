"""时间序列滑窗 Dataset。

支持：
- Electricity (ECL) 风格 CSV：第一列为时间戳，其余列为变量
- ETTh1/h2/m1/m2 风格 CSV：同 ECL
- 自定义 numpy 数组
- 标准 7:1:2 / 6:2:2 划分

对应文档：experiment-design.md §七
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    test_ratio: float = 0.2
    # 验证 = 1 - train - test

    def split_lengths(self, total: int) -> Tuple[int, int, int]:
        n_test = int(total * self.test_ratio)
        n_train = int(total * self.train_ratio)
        n_val = total - n_train - n_test
        return n_train, n_val, n_test


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

    def __getitem__(self, idx: int):
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


class ECLDataset(Dataset):
    """Electricity (ECL) / Traffic / Solar 等：CSV 形式，多变量时间序列。"""

    def __init__(
        self,
        csv_path: str,
        lookback: int,
        horizon: int,
        split: str = "train",
        split_cfg: Optional[SplitConfig] = None,
        stride: int = 1,
    ):
        df = _load_csv_dataframe(csv_path)
        data = df.values.astype(np.float32)  # (T, V)
        time_feats = add_time_features(df.index, freq="h")  # (T, 2)
        self.time_index = df.index
        n_train, n_val, n_test = (split_cfg or SplitConfig()).split_lengths(len(data))
        if split == "train":
            data = data[:n_train]
            time_feats = time_feats[:n_train]
        elif split == "val":
            data = data[n_train:n_train + n_val]
            time_feats = time_feats[n_train:n_train + n_val]
            stride = 1  # 验证集密集采样，避免漏样本
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


class ETDataset(Dataset):
    """ETTh1/h2/m1/m2：与 ECL 同结构，freq 决定是否带 minute。"""

    def __init__(
        self,
        csv_path: str,
        lookback: int,
        horizon: int,
        split: str = "train",
        split_cfg: Optional[SplitConfig] = None,
        stride: int = 1,
        freq: str = "h",
    ):
        df = _load_csv_dataframe(csv_path)
        data = df.values.astype(np.float32)
        time_feats = add_time_features(df.index, freq=freq)
        # 兼容：dict 形式的 split 配置（来自 yaml）自动包成 SplitConfig
        if isinstance(split_cfg, dict):
            split_cfg = SplitConfig(**split_cfg)
        n_train, n_val, n_test = (split_cfg or SplitConfig()).split_lengths(len(data))
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
    "build_dataloader",
]
