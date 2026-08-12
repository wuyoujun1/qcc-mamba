#!/usr/bin/env python3
"""数据加载吞吐微基准：测不同 num_workers/pin_memory/persistent_workers 组合下，
ECL L=96 train loader 的加载吞吐（只加载，不上 GPU）。"""
import sys
import time

import torch
from data.dataloader import build_standard_loaders

CONFIGS = [
    {"num_workers": 0, "pin_memory": True, "persistent_workers": False, "tag": "nw0_pin"},
    {"num_workers": 0, "pin_memory": False, "persistent_workers": False, "tag": "nw0_nopin"},
    {"num_workers": 4, "pin_memory": True, "persistent_workers": True, "tag": "nw4_pin_persist"},
    {"num_workers": 4, "pin_memory": False, "persistent_workers": True, "tag": "nw4_nopin_persist"},
    {"num_workers": 8, "pin_memory": False, "persistent_workers": True, "tag": "nw8_nopin_persist"},
]


def loaders_with(dl_cfg):
    """构造指定 DataLoader 设置的 loaders（临时改 build_dataloader 参数）。"""
    import data.dataloader as dd
    from data.dataset import TimeSeriesDataset, SplitConfig
    from torch.utils.data import DataLoader

    orig = dd.build_dataloader

    def patched(dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False):
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last,
            pin_memory=dl_cfg["pin_memory"],
            persistent_workers=dl_cfg["persistent_workers"],
        )

    dd.build_dataloader = patched
    try:
        loaders = dd.build_standard_loaders(
            dataset_name="electricity", lookback=96, horizon=96,
            batch_size=32, stride=1, num_workers=dl_cfg["num_workers"],
            data_dir="../ts_quantum/datasets",
        )
        return loaders
    finally:
        dd.build_dataloader = orig


def bench_loader(loader, n_batches=300):
    """迭代 n_batches，返回 批次/秒（含 collate，不上 GPU）。"""
    t0 = time.time()
    cnt = 0
    for batch in loader:
        cnt += 1
        if cnt >= n_batches:
            break
    dt = time.time() - t0
    return cnt / dt if dt > 0 else float("inf")


def main():
    print(f"{'tag':>20} | {'b/s':>8} | {'ms/batch':>9} | {'800批/epoch':>12}")
    for cfg in CONFIGS:
        try:
            loaders = loaders_with(cfg)
            train = loaders["train"]
            # 预热：worker fork
            next(iter(train))
            bps = bench_loader(train)
            n_windows = len(train)
            ep_batches = n_windows // 32
            est = ep_batches / bps if bps else float("inf")
            print(f"{cfg['tag']:>20} | {bps:>7.1f} | {1000/bps:>8.1f} | {est:>10.1f}s/epoch (n={ep_batches})")
        except Exception as e:
            print(f"{cfg['tag']:>20} | ERROR: {e}")
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
