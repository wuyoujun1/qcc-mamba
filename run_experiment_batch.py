#!/usr/bin/env python3
"""DualAE-QCC 36h 无人值守批处理驱动。

对应方案：wholeexperiment.md。按 phase0→phase1→phase2→phase3→phase4 顺序执行，
每实验生成临时 yaml → 调 run_dual_ae.py → 超时/失败继续 → 指标解析进 results/summary.csv
→ done 标记（断点续跑依据）。

与方案的差异（审查后修正，见 wholeexperiment.md 修订说明）：
- baseline 用 use_bypass=false（真·无旁路 S-Mamba），而非 use_fmap=false 四开关
  （后者仍会跑量子核旁路，不是"无旁路"）；
- 全矩阵 use_amp=false（AMP fp16 偶发 NaN，已实测；关掉后稳定）；
- 每 epoch 只评 val（eval_test_every_epoch=false），test 留到结尾，省 ~25% 时间；
- epochs=50 / patience=8（夜间筛选用缩减档，保证 36h 内跑完 P1/P2/P3）。

用法：
    python run_experiment_batch.py [--only phase0_smoke] [--deadline 36] [--resume] [--calib 1.0]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(ROOT, "configs")
GEN_DIR = os.path.join(ROOT, "configs", "generated")
LOG_DIR = os.path.join(ROOT, "logs", "batch")
DONE_DIR = os.path.join(ROOT, "results", ".done")
SUMMARY_CSV = os.path.join(ROOT, "results", "summary.csv")

# ---------------------------------------------------------------------- #
# 全局训练档位（36h 筛选用）
# ---------------------------------------------------------------------- #
SCREEN_EPOCHS = 50
SCREEN_PATIENCE = 8
USE_AMP = False            # AMP fp16 偶发 NaN，关闭保证无人值守稳定
EVAL_TEST_EVERY_EPOCH = False

# P1 全矩阵统一量子比特数（用户指示：N=10→8，QKCS 状态维 2^N 减小 4 倍，
# 复数 cgemm 成本降 ~3/4，实测 ECL L=96 dual 从 ~3min/epoch 降到 ~0.75min/epoch）。
N_QUBITS = 8

# 数据集 loader 名映射（方案里用 ecl / pemsbay，loader 用 electricity / pems_bay）
DS_ALIAS = {"ecl": "electricity", "pemsbay": "pems_bay", "pems_bay": "pems_bay"}

# 数据集异质强度（δ̂ std % 主导周期），用于 summarize；None=待回填
HETEROGENEITY = {
    "electricity": 20.5, "etth1": 28.6, "etth2": 20.9, "weather": 17.0,
    "chinaaqi": None, "metr_la": None, "pems_bay": None,
    "ettm1": 6.9, "ettm2": 7.6, "traffic": 2.4, "exchange": 12.6,
}

# 数据集类别（决定是否大变量/小变量，用于时间预算）
BIG_V = {"electricity", "chinaaqi", "metr_la", "pems_bay", "traffic"}

# 基准每 epoch 估算（分钟）。QCC 为主、L 无关；backbone 随 L 近似线性。
# 实测 ECL L=96 dual N=10 ≈ 3min/epoch，baseline ≈ 0.3min/epoch。
# QCC 成本 ∝ 状态维 2^N，故乘 (2^(N-10)) 因子（N=8 → 0.25）。
def _epoch_est_min(dataset: str, L: int, config: str, n_qubits: int = 10) -> float:
    big = dataset in BIG_V
    base = 3.0 if big else 0.35          # L=96 每 epoch 分钟（N=10）
    scale = max(1.0, L / 96.0) * 0.5 + 0.5   # backbone 随 L 放大，QCC 不变
    if config == "baseline":
        return 0.3 * scale                # 无 QCC，n_qubits 无关
    return base * (0.6 + 0.4 * scale) * (2 ** (n_qubits - 10))


# ---------------------------------------------------------------------- #
# 实验矩阵（对应 wholeexperiment.md §3）
# ---------------------------------------------------------------------- #
# 每个作业：id, phase, ds, L, H, seed, base_config, model_overrides
PHASES = {}


def _add(jobs, pid, ds, L, H, seed, base, overrides=None, epochs=SCREEN_EPOCHS, patience=SCREEN_PATIENCE, phase="phase0"):
    jobs.append({
        "id": pid, "phase": phase, "ds": DS_ALIAS.get(ds, ds),
        "L": L, "H": H, "seed": seed, "base": base,
        "overrides": overrides or {},
        "epochs": epochs, "patience": patience,
        "seq": len(jobs),     # 构造顺序，排序用（同 phase 内按此执行）
    })


def build_jobs():
    jobs = []

    # ---- phase0_smoke ----
    _add(jobs, "smoke_ecl_96_dual", "ecl", 96, 96, 42, "dual_phase1.yaml",
         epochs=1, patience=1, phase="phase0_smoke")

    # ---- phase1：P1 定方向（ECL × L=96 × 2 seed）----
    for s in (42, 2024):
        _add(jobs, f"p1_ecl_baseline_{s}", "ecl", 96, 96, s, "dual_baseline_smamba.yaml",
             {"use_bypass": False}, phase="phase1")           # 修正：真·无旁路
        _add(jobs, f"p1_ecl_h_{s}", "ecl", 96, 96, s, "dual_ablation_h_only.yaml",
             {}, phase="phase1")
        _add(jobs, f"p1_ecl_s_{s}", "ecl", 96, 96, s, "dual_ablation_s_only.yaml",
             {}, phase="phase1")
        _add(jobs, f"p1_ecl_dual_{s}", "ecl", 96, 96, s, "dual_phase1.yaml",
             {}, phase="phase1")

    # ---- phase2：P2 生死线（6 数据集 × {dual, no_align} × L=96 × seed42）----
    for ds in ("ecl", "chinaaqi", "metr_la", "pems_bay", "etth1", "weather"):
        _add(jobs, f"p2_{ds}_dual_42", ds, 96, 96, 42, "dual_phase1.yaml", {}, phase="phase2")
        _add(jobs, f"p2_{ds}_no_align_42", ds, 96, 96, 42, "dual_ablation_no_align.yaml",
             {"spectrum_time_align": False, "spectrum_freq_align": False}, phase="phase2")

    # ---- phase3：P3 对齐 + 参数消融（ECL × L=96 × seed42）----
    _add(jobs, "p3_ecl_time_only_42", "ecl", 96, 96, 42, "dual_ablation_time_only.yaml", {}, phase="phase3")
    _add(jobs, "p3_ecl_freq_only_42", "ecl", 96, 96, 42, "dual_ablation_freq_only.yaml", {}, phase="phase3")
    _add(jobs, "p3_ecl_N8_42", "ecl", 96, 96, 42, "dual_phase1.yaml", {"n_qubits": 8}, phase="phase3")
    _add(jobs, "p3_ecl_N12_42", "ecl", 96, 96, 42, "dual_phase1.yaml", {"n_qubits": 12}, phase="phase3")
    _add(jobs, "p3_ecl_M16_42", "ecl", 96, 96, 42, "dual_phase1.yaml", {"spectrum_M": 16}, phase="phase3")
    _add(jobs, "p3_ecl_M48_42", "ecl", 96, 96, 42, "dual_phase1.yaml", {"spectrum_M": 48}, phase="phase3")

    # ---- phase4：E2 主档位（长 L × 双阶段 × seed42）----
    _add(jobs, "p4_ecl_336_42", "ecl", 336, 336, 42, "dual_phase1.yaml", {}, phase="phase4")
    _add(jobs, "p4_ecl_720_42", "ecl", 720, 720, 42, "dual_phase1.yaml", {}, phase="phase4")
    _add(jobs, "p4_chinaaqi_336_42", "chinaaqi", 336, 336, 42, "dual_phase1.yaml", {}, phase="phase4")
    _add(jobs, "p4_pemsbay_336_42", "pems_bay", 336, 336, 42, "dual_phase1.yaml", {}, phase="phase4")
    _add(jobs, "p4_metrla_336_42", "metr_la", 336, 336, 42, "dual_phase1.yaml", {}, phase="phase4")
    _add(jobs, "p4_weather_720_42", "weather", 720, 720, 42, "dual_phase1.yaml", {}, phase="phase4")
    _add(jobs, "p4_etth1_336_42", "etth1", 336, 336, 42, "dual_phase1.yaml", {}, phase="phase4")

    # ---- 缓冲批（INCLUDE_BUFFER=1 时启用，随主批跑完后追加，--resume 续跑）----
    # p2s2：P2 生死线补第 2 seed（核心判据稳定性，2026-08-08 因 P2 信号偏弱而建议）
    # p3s2：§2 缓冲优先 1——phase3 对齐消融补 1 seed
    # e2na：§2 缓冲优先 4——E2 no_align 长档（异质度图第二个 L 点）
    buf_tiers = (os.environ.get("INCLUDE_BUFFER", "") or "").split(",")
    buf_tiers = {t.strip() for t in buf_tiers if t.strip()}
    if buf_tiers:
        print(f"[batch] 启用缓冲批: {sorted(buf_tiers)}", flush=True)
    if "p2s2" in buf_tiers:
        for ds in ("ecl", "chinaaqi", "metr_la", "pems_bay", "etth1", "weather"):
            _add(jobs, f"p2_{ds}_dual_2024", ds, 96, 96, 2024, "dual_phase1.yaml", {}, phase="buffer_p2s2")
            _add(jobs, f"p2_{ds}_no_align_2024", ds, 96, 96, 2024, "dual_ablation_no_align.yaml",
                 {"spectrum_time_align": False, "spectrum_freq_align": False}, phase="buffer_p2s2")
    if "p3s2" in buf_tiers:
        _add(jobs, "p3_ecl_time_only_2024", "ecl", 96, 96, 2024, "dual_ablation_time_only.yaml", {}, phase="buffer_p3s2")
        _add(jobs, "p3_ecl_freq_only_2024", "ecl", 96, 96, 2024, "dual_ablation_freq_only.yaml", {}, phase="buffer_p3s2")
        _add(jobs, "p3_ecl_N8_2024", "ecl", 96, 96, 2024, "dual_phase1.yaml", {"n_qubits": 8}, phase="buffer_p3s2")
        _add(jobs, "p3_ecl_N12_2024", "ecl", 96, 96, 2024, "dual_phase1.yaml", {"n_qubits": 12}, phase="buffer_p3s2")
        _add(jobs, "p3_ecl_M16_2024", "ecl", 96, 96, 2024, "dual_phase1.yaml", {"spectrum_M": 16}, phase="buffer_p3s2")
        _add(jobs, "p3_ecl_M48_2024", "ecl", 96, 96, 2024, "dual_phase1.yaml", {"spectrum_M": 48}, phase="buffer_p3s2")
    if "e2na" in buf_tiers:
        # 与 phase4 的 dual 长档一一对应的 no_align 对照组（第二个 L 点）
        _add(jobs, "p4_ecl_no_align_336_42", "ecl", 336, 336, 42, "dual_ablation_no_align.yaml",
             {"spectrum_time_align": False, "spectrum_freq_align": False}, phase="buffer_e2na")
        _add(jobs, "p4_ecl_no_align_720_42", "ecl", 720, 720, 42, "dual_ablation_no_align.yaml",
             {"spectrum_time_align": False, "spectrum_freq_align": False}, phase="buffer_e2na")
        _add(jobs, "p4_chinaaqi_no_align_336_42", "chinaaqi", 336, 336, 42, "dual_ablation_no_align.yaml",
             {"spectrum_time_align": False, "spectrum_freq_align": False}, phase="buffer_e2na")
        _add(jobs, "p4_pemsbay_no_align_336_42", "pems_bay", 336, 336, 42, "dual_ablation_no_align.yaml",
             {"spectrum_time_align": False, "spectrum_freq_align": False}, phase="buffer_e2na")
        _add(jobs, "p4_metrla_no_align_336_42", "metr_la", 336, 336, 42, "dual_ablation_no_align.yaml",
             {"spectrum_time_align": False, "spectrum_freq_align": False}, phase="buffer_e2na")
        _add(jobs, "p4_weather_no_align_720_42", "weather", 720, 720, 42, "dual_ablation_no_align.yaml",
             {"spectrum_time_align": False, "spectrum_freq_align": False}, phase="buffer_e2na")

    return jobs


# ---------------------------------------------------------------------- #
# P1 全矩阵（2026-08-09 重做：6 数据集 × 4 配置 × 4 L × 2 seed = 192）
# 与旧 P1 的差异：全任务 n_qubits=8（用户提速指示）；其余档位不变
# （epochs=50/patience=8/use_amp=false/eval_test_every_epoch=false）。
# 排序：按 L 分波（96→192→336→720，96 先出方向信号）；
# 波内按 数据集(快→慢) × 配置(dual/h_only/s_only/baseline) × seed(42,2024)。
# ---------------------------------------------------------------------- #
P1F_DS_ORDER = ["ecl", "chinaaqi", "metr_la", "etth1", "weather", "pems_bay"]
P1F_CONFIGS = [
    ("dual",     "dual_phase1.yaml",            {}),
    ("h_only",   "dual_ablation_h_only.yaml",   {}),
    ("s_only",   "dual_ablation_s_only.yaml",   {}),
    ("baseline", "dual_baseline_smamba.yaml",   {"use_bypass": False}),
]
P1F_LOOKBACKS = [96, 192, 336, 720]
P1F_SEEDS = [42, 2024]


def build_p1f_jobs(jobs):
    """P1 定方向全矩阵：6 数据集 × {dual, h_only, s_only, baseline} × {96,192,336,720} × {42,2024}。"""
    for L in P1F_LOOKBACKS:
        phase = f"p1f_{L}"
        for ds in P1F_DS_ORDER:
            for cname, base, ov in P1F_CONFIGS:
                for s in P1F_SEEDS:
                    _add(jobs, f"p1f_{ds}_{cname}_{L}_{s}", ds, L, L, s, base,
                         {**ov, "n_qubits": N_QUBITS}, phase=phase)


# ---------------------------------------------------------------------- #
# 临时 yaml 生成
# ---------------------------------------------------------------------- #
def make_job_yaml(job) -> str:
    with open(os.path.join(CFG_DIR, job["base"]), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["dataset"] = job["ds"]
    cfg["lookback"] = job["L"]
    cfg["horizon"] = job["H"]
    cfg["seed"] = job["seed"]

    # 训练档位（全局）
    cfg["train"]["epochs"] = job["epochs"]
    cfg["train"]["patience"] = job["patience"]
    cfg["train"]["use_amp"] = USE_AMP
    cfg["train"]["eval_test_every_epoch"] = EVAL_TEST_EVERY_EPOCH
    # 死锁修复（2026-08-09）：num_workers=4 + persistent_workers=True + 预热。
    # 旧的 num_workers=0（主进程加载）在 N=8 提速后加载占比变可观，且用户要求
    # "numworker开大一点，不要让gpu等待"。配套修复：data/dataset.py 的
    # build_dataloader 加 persistent_workers=(num_workers>0)，run_dual_ae.py 在
    # CUDA 初始化前预热 DataLoader worker（避免 CUDA 初始化后再 fork 导致 epoch
    # 边界死锁）。已用 N=8 4-epoch 冒烟跨 3 次 epoch 边界验证零死锁。
    cfg["train"]["num_workers"] = 4

    # 模型 overrides（含 use_bypass、n_qubits、spectrum_M、对齐开关等）
    for k, v in job["overrides"].items():
        cfg["model"][k] = v
        if k == "spectrum_M":
            # 保证 M ≤ L/2+1
            cfg["model"]["spectrum_M"] = min(v, job["L"] // 2 + 1)

    # 保存目录 / run_name（统一放 results/matrix/<phase>/）
    save_dir = os.path.join(ROOT, "results", "matrix", job["phase"])
    cfg["save_dir"] = save_dir
    cfg["run_name"] = job["id"]

    out = os.path.join(GEN_DIR, f"{job['id']}.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return out


# ---------------------------------------------------------------------- #
# 指标解析
# ---------------------------------------------------------------------- #
def parse_metrics(log_text: str):
    def g(pat):
        m = re.search(pat, log_text)
        return float(m.group(1)) if m else None
    mse, mae = g(r"Test MSE: ([\d.eE+-]+)"), g(r"Test MAE: ([\d.eE+-]+)")
    mse_norm, mae_norm = g(r"Test MSE \(normalized\): ([\d.eE+-]+)"), g(r"Test MAE \(normalized\): ([\d.eE+-]+)")
    m = re.search(r"val_mse=([\d.eE+-]+)", log_text)
    best_val = float(m.group(1)) if m else None
    m = re.search(r"Early stopping at epoch (\d+)", log_text)
    epochs_run = int(m.group(1)) if m else None
    kdiag = g(r"K stats: diag_mean=([\d.eE+-]+)")
    koff = g(r"offdiag_mean=([\d.eE+-]+)")
    crashed = ("Traceback" in log_text) or mse is None
    return {"mse": mse, "mae": mae, "mse_norm": mse_norm, "mae_norm": mae_norm,
            "best_val": best_val, "epochs_run": epochs_run,
            "k_diag": kdiag, "k_offdiag": koff, "crashed": crashed}


def write_summary_row(job, metrics, elapsed_min, status, log_path):
    header = ["id", "phase", "config", "dataset", "hetero_dstd_pct", "L", "H", "seed",
              "epochs_planned", "epochs_run", "best_val", "test_mse", "test_mae",
              "mse_norm", "mae_norm", "k_diag", "k_offdiag", "elapsed_min", "status", "log"]
    new_file = not os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow({
            "id": job["id"], "phase": job["phase"],
            "config": job["base"].replace("dual_", "").replace(".yaml", ""),
            "dataset": job["ds"], "hetero_dstd_pct": HETEROGENEITY.get(job["ds"]),
            "L": job["L"], "H": job["H"], "seed": job["seed"],
            "epochs_planned": job["epochs"],
            "epochs_run": metrics["epochs_run"] or job["epochs"],
            "best_val": metrics["best_val"], "test_mse": metrics["mse"],
            "test_mae": metrics["mae"], "mse_norm": metrics["mse_norm"],
            "mae_norm": metrics["mae_norm"], "k_diag": metrics["k_diag"],
            "k_offdiag": metrics["k_offdiag"], "elapsed_min": round(elapsed_min, 1),
            "status": status, "log": os.path.relpath(log_path, ROOT),
        })


def load_done():
    if not os.path.isdir(DONE_DIR):
        return set()
    return {p[:-5] for p in os.listdir(DONE_DIR) if p.endswith(".done")}


def mark_done(jid):
    os.makedirs(DONE_DIR, exist_ok=True)
    open(os.path.join(DONE_DIR, f"{jid}.done"), "w").close()


# ---------------------------------------------------------------------- #
# 主循环
# ---------------------------------------------------------------------- #
def main():
    global SCREEN_EPOCHS, SCREEN_PATIENCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑某 phase，如 phase1,phase2")
    ap.add_argument("--deadline", type=float, default=36.0, help="总预算小时数（默认 36）")
    ap.add_argument("--calib", type=float, default=1.0, help="时间预算校准系数")
    ap.add_argument("--resume", action="store_true", help="跳过已 done 作业")
    ap.add_argument("--epochs", type=int, default=SCREEN_EPOCHS)
    ap.add_argument("--patience", type=int, default=SCREEN_PATIENCE)
    args = ap.parse_args()

    SCREEN_EPOCHS = args.epochs
    SCREEN_PATIENCE = args.patience

    for d in (GEN_DIR, LOG_DIR, DONE_DIR, os.path.join(ROOT, "results", "matrix")):
        os.makedirs(d, exist_ok=True)

    jobs = build_jobs()
    build_p1f_jobs(jobs)
    if args.only:
        only = set(p.strip() for p in args.only.split(","))
        jobs = [j for j in jobs if j["phase"] in only]
    order = {"phase0_smoke": 0, "phase1": 1, "phase2": 2, "phase3": 3, "phase4": 4,
             "p1f_96": 5, "p1f_192": 6, "p1f_336": 7, "p1f_720": 8}
    jobs.sort(key=lambda j: (order.get(j["phase"], 9), j["seq"]))

    done = load_done() if args.resume else set()

    start = time.time()
    deadline = start + args.deadline * 3600
    print(f"[batch] 启动 {len(jobs)} 作业，deadline={args.deadline}h，"
          f"epochs={SCREEN_EPOCHS}/patience={SCREEN_PATIENCE}, AMP off", flush=True)

    for i, job in enumerate(jobs, 1):
        if job["id"] in done:
            print(f"[batch] [{i}/{len(jobs)}] 跳过（已完成）: {job['id']}", flush=True)
            continue
        if time.time() > deadline:
            print(f"[batch] ⏰ deadline 到，停止启动新作业（{job['id']} 未跑）", flush=True)
            break

        cfg_path = make_job_yaml(job)
        log_path = os.path.join(LOG_DIR, f"{job['id']}.log")
        est = _epoch_est_min(job["ds"], job["L"],
                             "baseline" in job["base"] and "baseline" or "dual",
                             job["overrides"].get("n_qubits", 10))
        timeout_s = max(60.0, job["epochs"] * est * 60 * 3 * args.calib)
        timeout_s = min(timeout_s, max(60.0, deadline - time.time()))  # 不越过全局 deadline

        t0 = time.time()
        print(f"[batch] [{i}/{len(jobs)}] ▶ {job['id']} "
              f"({job['base']}, {job['ds']}, L={job['L']}H={job['H']}, s{job['seed']}, 超时{timeout_s/60:.0f}min)",
              flush=True)
        try:
            with open(log_path, "w", encoding="utf-8") as flog:
                # -u：stdout 重定向到文件时是块缓冲，Epoch 行会积压到进程退出才落盘，
                # 被 SIGKILL 时全部丢失（pems_bay_dual_96_42 就死于"日志无痕迹"）。
                # -u 逐行刷新，监控/诊断可见实时进度。
                subprocess.run(
                    [sys.executable, "-u", os.path.join(ROOT, "run_dual_ae.py"), "--config", cfg_path],
                    cwd=ROOT, stdout=flog, stderr=subprocess.STDOUT, timeout=timeout_s,
                )
            status = "ok"
        except subprocess.TimeoutExpired:
            status = "timeout"
        except Exception as e:
            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(f"\n[driver] subprocess error: {e}\n")
            status = "failed"

        with open(log_path, "r", encoding="utf-8") as f:
            log_text = f.read()
        metrics = parse_metrics(log_text)
        if status == "ok" and metrics["crashed"]:
            status = "failed"
        elapsed_min = (time.time() - t0) / 60
        write_summary_row(job, metrics, elapsed_min, status, log_path)
        if status == "ok":
            mark_done(job["id"])
        icon = {"ok": "✅", "timeout": "⏰", "failed": "❌"}[status]
        print(f"[batch]   {job['id']} {icon} {elapsed_min:.1f}min "
              f"MSE_norm={metrics['mse_norm']} epochs={metrics['epochs_run']}", flush=True)

    print(f"[batch] 结束，总耗时 {(time.time()-start)/3600:.1f}h。汇总 → {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
