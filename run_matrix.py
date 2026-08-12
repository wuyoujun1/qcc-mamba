#!/usr/bin/env python3
"""DualAE-QCC 24h 实验矩阵批跑器。

按优先级 P0→P1→P2→P3 顺序执行作业；每个作业生成独立 yaml 到 _batch_configs/，
stdout 落盘 _batch_logs/<id>.log，指标汇总写入 results/overnight_matrix.csv。

特性：
- 断点续跑：已完成作业（CSV 中已有记录）自动跳过。
- 全局 deadline：默认 24h，到时不再启动新作业（当前作业跑完即停）。
- 冒烟作业单独跑（不在此矩阵内）。

用法：
    python run_matrix.py [--deadline 24] [--only P1,P2]
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(ROOT, "configs")
BATCH_CFG_DIR = os.path.join(ROOT, "_batch_configs")
LOG_DIR = os.path.join(ROOT, "_batch_logs")
RESULT_CSV = os.path.join(ROOT, "results", "overnight_matrix.csv")

# ---------------------------------------------------------------------- #
# 排期参数（可命令行覆盖）
# ---------------------------------------------------------------------- #
# 主线规范 epochs=100, patience=10。夜间筛选用缩减档提速（结果按同一档位对比，公平）。
SCREEN_EPOCHS = 60
SCREEN_PATIENCE = 8
EVAL_TEST_EVERY_EPOCH = False  # 提速：每 epoch 只评 val，test 留到结尾

P1_SEEDS = [42, 2026]   # ECL 编码消融 2 seed
P2_SEEDS = [42]         # 跨数据集（生死线）1 seed 起步，时间允许再加
P3_SEEDS = [42]

# 数据集元信息：(loader 名, 显示名, 异质度 δ̂std% 或 None 待回填, 大变量)
DATASETS = {
    "electricity": ("electricity", "ECL", 20.5, True),
    "etth1":       ("etth1",       "ETTh1", 28.6, False),
    "weather":     ("weather",     "Weather", 17.0, False),
    "chinaaqi":    ("chinaaqi",    "ChinaAQI", None, True),
    "metr_la":     ("metr_la",     "METR_LA", None, True),
    "pems_bay":    ("pems_bay",    "PEMS_BAY", None, True),
}

# P1：ECL L=H=96 编码消融（含重上传消融）
P1_CONFIGS = [
    ("P1", "phase1",        "dual_phase1.yaml",                "dual_stage",    {"use_bypass": None}),
    ("P1", "baseline",      "dual_baseline_smamba.yaml",       "smamba_base",   {"use_bypass": None}),
    ("P1", "h_only",        "dual_ablation_h_only.yaml",       "h_only",        {"use_bypass": None}),
    ("P1", "s_only",        "dual_ablation_s_only.yaml",       "s_only",        {"use_bypass": None}),
    ("P1", "reupload_H",    "dual_ablation_reupload_H.yaml",   "reupload_H",    {"use_bypass": None}),
    ("P1", "reupload_alt",  "dual_ablation_reupload_alternate.yaml", "reupload_alt", {"use_bypass": None}),
]

# P2：跨数据集 phase1 vs no_align（L=H=96）
P2_CONFIGS = [
    ("phase1", "dual_phase1.yaml"),
    ("no_align", "dual_ablation_no_align.yaml"),
]

# P3a：对齐消融（ECL）
P3_ALIGN = [
    ("time_only", "dual_ablation_time_only.yaml"),
    ("freq_only", "dual_ablation_freq_only.yaml"),
    ("no_align",  "dual_ablation_no_align.yaml"),
]


# ---------------------------------------------------------------------- #
# 作业定义
# ---------------------------------------------------------------------- #
def build_jobs():
    jobs = []

    # P1：ECL 编码消融（2 seed）
    for tag, name, cfg_file, cfg_name, _ in P1_CONFIGS:
        for seed in P1_SEEDS:
            jobs.append({
                "id": f"P1_{name}_s{seed}",
                "priority": 1,
                "config_file": cfg_file,
                "dataset": "electricity",
                "display_dataset": "ECL",
                "lookback": 96, "horizon": 96,
                "seed": seed,
                "run_name": f"P1_{cfg_name}_s{seed}",
            })

    # P2：跨数据集 生死线（phase1 vs no_align）
    for ds_key, (loader, disp, hetero, _big) in DATASETS.items():
        for tag, cfg_file in P2_CONFIGS:
            for seed in P2_SEEDS:
                jobs.append({
                    "id": f"P2_{disp}_{tag}_s{seed}",
                    "priority": 2,
                    "config_file": cfg_file,
                    "dataset": loader,
                    "display_dataset": disp,
                    "lookback": 96, "horizon": 96,
                    "seed": seed,
                    "run_name": f"P2_{disp}_{tag}_s{seed}",
                })

    # P3a：对齐消融（ECL，phase1 已在 P1）
    for name, cfg_file in P3_ALIGN:
        for seed in P3_SEEDS:
            jobs.append({
                "id": f"P3a_{name}_s{seed}",
                "priority": 3,
                "config_file": cfg_file,
                "dataset": "electricity",
                "display_dataset": "ECL",
                "lookback": 96, "horizon": 96,
                "seed": seed,
                "run_name": f"P3a_{name}_s{seed}",
            })

    return jobs


# ---------------------------------------------------------------------- #
# yaml 生成
# ---------------------------------------------------------------------- #
def load_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(cfg, path):
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)


def make_job_yaml(job) -> str:
    """基于基线 yaml 生成作业专属配置。返回写入的路径。"""
    base = load_yaml(os.path.join(CFG_DIR, job["config_file"]))
    cfg = copy.deepcopy(base)

    cfg["dataset"] = job["dataset"]
    cfg["lookback"] = job["lookback"]
    cfg["horizon"] = job["horizon"]
    cfg["seed"] = job["seed"]

    # 训练档位
    cfg["train"]["epochs"] = SCREEN_EPOCHS
    cfg["train"]["patience"] = SCREEN_PATIENCE
    cfg["train"]["eval_test_every_epoch"] = EVAL_TEST_EVERY_EPOCH

    # 保存目录/run_name（含优先级与 seed，避免覆盖）
    save_dir = os.path.join(ROOT, "results", "matrix", f"P{job['priority']}")
    cfg["save_dir"] = save_dir
    cfg["run_name"] = job["run_name"]

    # 注释附加数据集异质度（便于人工核对）
    out_path = os.path.join(BATCH_CFG_DIR, f"{job['id']}.yaml")
    dump_yaml(cfg, out_path)
    return out_path


# ---------------------------------------------------------------------- #
# 结果解析
# ---------------------------------------------------------------------- #
def parse_metrics(log_text: str, config_path: str):
    """从 run_dual_ae.py 的 stdout 解析最终指标。"""
    m = re.search(r"Test MSE: ([\d.eE+-]+)", log_text)
    mse = float(m.group(1)) if m else None
    m = re.search(r"Test MAE: ([\d.eE+-]+)", log_text)
    mae = float(m.group(1)) if m else None
    m = re.search(r"Test MSE \(normalized\): ([\d.eE+-]+)", log_text)
    mse_norm = float(m.group(1)) if m else None
    m = re.search(r"Test MAE \(normalized\): ([\d.eE+-]+)", log_text)
    mae_norm = float(m.group(1)) if m else None

    # 最佳 val_mse（早停选择标准）
    m = re.search(r"val_mse=([\d.eE+-]+)", log_text)
    best_val = float(m.group(1)) if m else None

    # 实际训练的 epoch 数
    epochs_run = SCREEN_EPOCHS
    m = re.search(r"Early stopping at epoch (\d+)", log_text)
    if m:
        epochs_run = int(m.group(1))

    # 异常判定
    crashed = "Traceback" in log_text or mse is None
    return {
        "mse": mse, "mae": mae, "mse_norm": mse_norm, "mae_norm": mae_norm,
        "best_val": best_val, "epochs_run": epochs_run, "crashed": crashed,
    }


def write_result_row(job, metrics, elapsed_min, out_path):
    header = ["id", "priority", "config", "dataset", "hetero_dstd_pct",
              "lookback", "horizon", "seed", "epochs_run",
              "mse", "mae", "mse_norm", "mae_norm", "best_val",
              "elapsed_min", "crashed", "log"]
    new_file = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow({
            "id": job["id"],
            "priority": job["priority"],
            "config": job["config_file"].replace("dual_", "").replace(".yaml", ""),
            "dataset": job["display_dataset"],
            "hetero_dstd_pct": DATASETS.get(job["dataset"], (None, None, None, None))[2],
            "lookback": job["lookback"],
            "horizon": job["horizon"],
            "seed": job["seed"],
            "epochs_run": metrics["epochs_run"],
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "mse_norm": metrics["mse_norm"],
            "mae_norm": metrics["mae_norm"],
            "best_val": metrics["best_val"],
            "elapsed_min": round(elapsed_min, 1),
            "crashed": metrics["crashed"],
            "log": f"_batch_logs/{job['id']}.log",
        })


def load_done_ids(out_path):
    if not os.path.exists(out_path):
        return set()
    with open(out_path, newline="", encoding="utf-8") as f:
        return {r["id"] for r in csv.DictReader(f)}


# ---------------------------------------------------------------------- #
# 主循环
# ---------------------------------------------------------------------- #
def main():
    global SCREEN_EPOCHS, SCREEN_PATIENCE, P1_SEEDS, P2_SEEDS, P3_SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", type=float, default=24.0, help="总预算小时数（默认 24）")
    ap.add_argument("--only", type=str, default=None, help="逗号分隔的优先级子集，如 '1,2'")
    ap.add_argument("--epochs", type=int, default=SCREEN_EPOCHS)
    ap.add_argument("--patience", type=int, default=SCREEN_PATIENCE)
    ap.add_argument("--seeds", type=str, default=None, help="覆盖 seed 列表，如 '42,2026'")
    ap.add_argument("--resume", action="store_true", help="跳过 CSV 中已完成作业")
    args = ap.parse_args()

    SCREEN_EPOCHS = args.epochs
    SCREEN_PATIENCE = args.patience
    if args.seeds:
        P1_SEEDS = P2_SEEDS = P3_SEEDS = [int(s) for s in args.seeds.split(",")]

    os.makedirs(BATCH_CFG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "matrix"), exist_ok=True)

    jobs = build_jobs()
    if args.only:
        keep = set(int(x) for x in args.only.split(","))
        jobs = [j for j in jobs if j["priority"] in keep]
    jobs.sort(key=lambda j: (j["priority"], j["id"]))

    done = load_done_ids(RESULT_CSV) if args.resume else set()

    start = time.time()
    deadline = start + args.deadline * 3600
    print(f"[runner] 启动矩阵批跑：{len(jobs)} 作业，deadline={args.deadline}h "
          f"(epochs={SCREEN_EPOCHS}, patience={SCREEN_PATIENCE})", flush=True)

    for i, job in enumerate(jobs, 1):
        if job["id"] in done:
            print(f"[runner] [{i}/{len(jobs)}] 跳过（已完成）: {job['id']}", flush=True)
            continue
        if time.time() > deadline:
            print(f"[runner] ⏰ deadline 已到，停止启动新作业（{job['id']} 未开始）", flush=True)
            break

        cfg_path = make_job_yaml(job)
        log_path = os.path.join(LOG_DIR, f"{job['id']}.log")
        t0 = time.time()
        print(f"[runner] [{i}/{len(jobs)}] ▶ {job['id']} "
              f"({job['config_file']}, {job['display_dataset']}, L={job['lookback']}H={job['horizon']}, s{job['seed']})",
              flush=True)
        try:
            with open(log_path, "w", encoding="utf-8") as flog:
                subprocess.run(
                    [sys.executable, os.path.join(ROOT, "run_dual_ae.py"), "--config", cfg_path],
                    cwd=ROOT, stdout=flog, stderr=subprocess.STDOUT,
                    timeout=max(1.0, deadline - time.time()),
                )
        except subprocess.TimeoutExpired:
            print(f"[runner] ⏰ {job['id']} 被 deadline 中断", flush=True)
            break

        with open(log_path, "r", encoding="utf-8") as f:
            log_text = f.read()
        metrics = parse_metrics(log_text, cfg_path)
        elapsed_min = (time.time() - t0) / 60
        write_result_row(job, metrics, elapsed_min, RESULT_CSV)
        status = "❌ CRASH" if metrics["crashed"] else "✅ ok"
        print(f"[runner]   {job['id']} 完成 {elapsed_min:.1f}min "
              f"MSE_norm={metrics['mse_norm']} {status}", flush=True)

    print(f"[runner] 批跑结束，总耗时 {(time.time()-start)/3600:.1f}h。结果 → {RESULT_CSV}", flush=True)


if __name__ == "__main__":
    main()
