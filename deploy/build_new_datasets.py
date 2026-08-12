#!/usr/bin/env python3
"""候选数据集落盘脚本：把原始下载文件转换为 (T,V) CSV（第一列 date 时间戳）。

用法:
    python build_new_datasets.py --task metr_la|pems_bay|china_aqi|all

各任务的原始输入与输出：
    metr_la   : downloads/METR_LA/METR_LA.pkl       -> datasets/metr_la.csv   (T,V) 5min
    pems_bay  : downloads/PEMS_BAY/PEMS_BAY.pkl     -> datasets/pems_bay.csv  (T,V) 5min
    china_aqi : downloads/china_aqi/ 下各站 csv     -> datasets/china_aqi.csv (T,V) 1h
"""
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("/home/wuyoujun/ts_quantum")
DL_DIR = BASE / "downloads"
DATA_DIR = BASE / "datasets"


def from_uctb_pkl(pkl_path: Path, start: str, out_name: str, freq: str):
    """UCTB 格式 pkl -> (T,V) CSV。UTC 数据藏在 obj['Node']['TrafficNode']。"""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    X = np.asarray(obj["Node"]["TrafficNode"], dtype=np.float64)     # (T, V)
    T, V = X.shape
    idx = pd.date_range(start=start, periods=T, freq=freq)
    df = pd.DataFrame(X, index=idx, columns=[f"sensor_{i}" for i in range(V)])
    df.index.name = "date"
    df = df.reset_index()
    out = DATA_DIR / out_name
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"  ✅ {out_name}: shape=(T={T}, V={V}) -> {out}")
    return T, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    args = ap.parse_args()
    if args.task in ("all", "metr_la"):
        from_uctb_pkl(DL_DIR / "METR_LA/METR_LA.pkl", "2012-03-01", "metr_la.csv", "5min")
    if args.task in ("all", "pems_bay"):
        from_uctb_pkl(DL_DIR / "PEMS_BAY/PEMS_BAY.pkl", "2017-01-01", "pems_bay.csv", "5min")
    if args.task in ("all", "china_aqi"):
        print("  ⚠️  china_aqi 需要先下载到 downloads/china_aqi/，再运行")


if __name__ == "__main__":
    main()
