#!/usr/bin/env python3
"""PJM 合并脚本：把 12 个区域小时负荷 CSV 拼成 (T,V) 矩阵并落盘 pjm.csv。

来源：Kaggle "Hourly Energy Consumption"（robikscube）经 GitHub 镜像
      iamirmasoud/energy_consumption_prediction 落地到 downloads/pjm/。
各文件列：Datetime,<ZONE>_MW。12 列 = AEP COMED DAYTON DEOK DOM DUQ EKPC FE NI PJME PJMW PJM_Load。

合并策略：外连接对齐时间轴 → 线性插值填补区内缺失 → 边缘向前/向后填充。
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("/home/wuyoujun/ts_quantum")
DL = BASE / "downloads/pjm"
DATA_DIR = BASE / "datasets"

ZONES = ["AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ",
         "EKPC", "FE", "NI", "PJME", "PJMW", "PJM_Load"]
EXPECT = {"AEP": 3395509, "COMED": 1842915, "DAYTON": 3274443, "DEOK": 1558965,
          "DOM": 3206580, "DUQ": 3214852, "EKPC": 1220853, "FE": 1701528,
          "NI": 1621599, "PJME": 4070265, "PJMW": 3866578, "PJM_Load": 921109}


def load_zone(zone):
    df = pd.read_csv(DL / f"{zone}_hourly.csv")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").sort_index()
    col = df.columns[0]          # <ZONE>_MW
    s = df[col].astype(float)
    s.name = zone
    # DST 回拨会造成重复小时索引 → 同时间戳取均值去重
    if not s.index.is_unique:
        s = s.groupby(s.index).mean()
    return s


def main():
    # 只使用完整下载的区域；不完整/缺失的跳过并注明
    avail = []
    for z in ZONES:
        p = DL / f"{z}_hourly.csv"
        if p.exists() and p.stat().st_size >= EXPECT[z] * 0.99:
            avail.append(z)
        else:
            print(f"  ⚠️ 跳过（文件缺失/不完整）: {z}")
    series = {z: load_zone(z) for z in avail}
    # 各区域时间范围
    for z, s in series.items():
        print(f"  {z:10s} {s.index.min()} -> {s.index.max()}  n={len(s)}")
    df = pd.concat(series.values(), axis=1)      # 外连接
    print(f"\nmerged shape (outer): {df.shape}, NaN%={df.isna().mean().mean()*100:.2f}")
    # 只保留所有区域都有真实数据的公共区间（避免边缘 ffill 造出常数段污染 FFT）
    df = df.dropna(how="any")
    df = df.interpolate(method="linear", limit_area="inside")
    df = df.ffill().bfill()
    print(f"after common-range fill: shape={df.shape}, remaining NaN={int(df.isna().sum().sum())}")
    out = DATA_DIR / "pjm.csv"
    df.index.name = "date"
    df.reset_index().to_csv(out, index=False, float_format="%.2f")
    print(f"  ✅ pjm.csv: (T={df.shape[0]}, V={df.shape[1]}) -> {out}")


if __name__ == "__main__":
    main()
