#!/usr/bin/env python3
"""ChinaAQI 构建脚本：CN-AIR 城市级逐小时 CSV → (T,V) 矩阵 china_aqi.csv。

来源：atomgit.com/GewisLab/CN-AIR（国内直连克隆），2024 全年 366 个日文件，
每文件 15 种污染物 type × 375 城市 × 24 小时。取 type=AQI，按小时拼接为 (T≈8784, V=375)。
缺失值按列中位数填充；若某城市整体缺失率过高则剔除。
"""
import pandas as pd
import numpy as np
from pathlib import Path

DL = Path("/home/wuyoujun/ts_quantum/downloads/cnair/CN-AIR")
DATA_DIR = Path("/home/wuyoujun/ts_quantum/datasets")

YEAR_DIR = "城市_20240101-20241231/城市_20240101-20241231"
TYPE = "AQI"


def main():
    files = sorted((DL / YEAR_DIR).glob("china_cities_*.csv"))
    print(f"{len(files)} daily files")
    frames = []
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        sub = df[df["type"] == TYPE].copy()
        sub["date"] = pd.to_datetime(sub["date"].astype(str), format="%Y%m%d") \
            + pd.to_timedelta(sub["hour"], unit="h")
        sub = sub.drop(columns=["hour", "type"]).set_index("date")
        frames.append(sub)
        if i % 60 == 0:
            print(f"  read {i}/{len(files)}")
    X = pd.concat(frames, axis=0)                      # (T, V)
    X = X.sort_index()
    print(f"merged: shape={X.shape}, NaN%={X.isna().mean().mean()*100:.2f}")
    # 重索引到完整小时网格，缺失小时线性插值（散点缺失 1.1%，插值后采样间隔严格 1h）
    full = pd.date_range(X.index.min(), X.index.max(), freq="h")
    X = X.reindex(full)
    X = X.interpolate(method="linear", limit_area="inside").ffill().bfill()
    print(f"after reindex to full hourly grid: shape={X.shape}, remaining NaN={int(X.isna().sum().sum())}")
    # 整体缺失率过高（>20%）的城市剔除
    miss = X.isna().mean()
    keep = miss[miss <= 0.20]
    drop = miss[miss > 0.20]
    X = X[keep.index]
    print(f"after dropping {len(drop)} cities (>20% missing): shape={X.shape}")
    # 剩余缺失按列中位数填充
    for c in X.columns:
        med = X[c].median()
        if np.isnan(med):
            med = 0.0
        X[c] = X[c].fillna(med)
    print(f"after median fill: remaining NaN={int(X.isna().sum().sum())}")
    X.index.name = "date"
    out = DATA_DIR / "china_aqi.csv"
    X.reset_index().to_csv(out, index=False, float_format="%.1f")
    print(f"  ✅ china_aqi.csv: (T={X.shape[0]}, V={X.shape[1]}) "
          f"{X.index.min()} -> {X.index.max()} -> {out}")


if __name__ == "__main__":
    main()
