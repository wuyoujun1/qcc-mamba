#!/usr/bin/env python3
"""标准公开数据集一键下载脚本。

用法：
    python download_datasets.py --dir ../datasets

支持的 datasets（全部 / 按需选一个或多个）：
    electricity, etth1, etth2, ettm1, ettm2, traffic, weather, solar, exchange

Pecan Street / 多光伏站等需注册的数据集不在此脚本范围内，请手动放置。
"""
from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
import argparse
from pathlib import Path

# ── 数据集 URL 映射 ──────────────────────────────────────────────
DATASETS = {
    "electricity": {
        "url": "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
        "filename": "electricity.zip",
        "extract": True,
        "csv_name": "LD2011_2014.txt",        # 原始格式，后续重命名
        "rename_to": "electricity.csv",
        "note": "UCI ElectricityLoadDiagrams 2011-2014",
    },
    "etth1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "filename": "ETTh1.csv",
        "extract": False,
        "note": "ETTh1 (hourly)",
    },
    "etth2": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        "filename": "ETTh2.csv",
        "extract": False,
        "note": "ETTh2 (hourly)",
    },
    "ettm1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "filename": "ETTm1.csv",
        "extract": False,
        "note": "ETTm1 (15-minute)",
    },
    "ettm2": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
        "filename": "ETTm2.csv",
        "extract": False,
        "note": "ETTm2 (15-minute)",
    },
    "traffic": {
        "url": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz",
        "filename": "traffic.txt.gz",
        "extract": False,    # 需要额外处理
        "note": "Traffic (需额外处理，建议从 TSL 仓库取现成 csv)",
    },
    "weather": {
        "url": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/weather/weather.txt.gz",
        "filename": "weather.txt.gz",
        "extract": False,
        "note": "Weather",
    },
    "solar": {
        "url": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar_AL/solar.txt.gz",
        "filename": "solar.txt.gz",
        "extract": False,
        "note": "Solar-Energy",
    },
    "exchange": {
        "url": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
        "filename": "exchange_rate.txt.gz",
        "extract": False,
        "note": "Exchange Rate",
    },
}


def download_file(url: str, dest: str) -> None:
    """下载单个文件，带进度信息。"""
    print(f"  ⬇️  下载: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        print(f"      ✅ 完成 ({size / 1024:.1f} KB)")
    except Exception as e:
        print(f"      ❌ 失败: {e}")
        if os.path.exists(dest):
            os.remove(dest)


def main():
    parser = argparse.ArgumentParser(description="下载标准时序数据集")
    parser.add_argument("--dir", default="../datasets", help="数据集目标目录")
    parser.add_argument(
        "--datasets", nargs="+",
        default=list(DATASETS.keys()),
        help=f"要下载的数据集，可选: {list(DATASETS.keys())}"
    )
    args = parser.parse_args()

    target_dir = Path(args.dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂 数据集目录: {target_dir.resolve()}\n")

    for name in args.datasets:
        if name not in DATASETS:
            print(f"  ⚠️ 跳过未知数据集: {name}")
            continue

        info = DATASETS[name]
        dest = target_dir / info["filename"]
        if dest.exists():
            print(f"  ✅ {name}: 已存在 ({dest.name}), 跳过")
            continue

        print(f"🌐 {name} ({info['note']})")
        if info.get("extract"):
            # 下载 zip
            zip_dest = dest
            download_file(info["url"], zip_dest)
            if zip_dest.exists() and zipfile.is_zipfile(zip_dest):
                with zipfile.ZipFile(zip_dest, "r") as zf:
                    # 找到 csv 或 txt
                    csv_members = [m for m in zf.namelist() if m.endswith(".csv") or m.endswith(".txt")]
                    if csv_members:
                        zf.extract(csv_members[0], target_dir)
                        extracted = target_dir / csv_members[0]
                        if csv_members[0] != info.get("rename_to"):
                            extracted.rename(target_dir / info["rename_to"])
                        print(f"      ✅ 解压重命名: {info.get('rename_to', csv_members[0])}")
                zip_dest.unlink()  # 删 zip
        else:
            download_file(info["url"], dest)

    print(f"\n✅ 完成！数据集目录: {target_dir.resolve()}")
    print("⚠️  注意: Pecan Street / 多光伏站等需注册的数据集需手动下载放置。")


if __name__ == "__main__":
    main()
