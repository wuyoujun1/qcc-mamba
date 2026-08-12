#!/usr/bin/env python3
"""量子混合主干实验运行器（2026-08-11 重构，无旁路）。

架构：量子核进主干 —— 每层 Mamba 后插 QuantumMixLayer（保真度核 K 做跨变量消息传递），
可选频谱输入注入（S → proj → 加到 embedding 输出）。预测 = 主干直接输出，无修正头。

变体（对照三臂 baseline_smamba / h_only / s_only 已存于 results/summary.csv）：
  qmix     : 量子混合层 ×2（每层 Mamba 后一层），无频谱注入
  qmix_sin : 量子混合层 ×2 + 频谱输入注入
  plain    : 纯主干对照（同训练配置，qmix_layers=0）

实验矩阵：6 cell（快 cell 优先）× 3 变体 × 2 seed = 36 runs。
2 并发，断点续跑（.done），汇总 results/qmix_summary.csv。

用法：
  python run_qmix.py
  python run_qmix.py --dry
  python run_qmix.py --scope etth1:96 --variants qmix
"""
import os
import re
import subprocess
import sys
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
YAML_DIR = "configs/qmix"
LOG_DIR = "logs/qmix"
DONE_DIR = "results/qmix_done"
SAVE_DIR = "results/qmix"
OUT_CSV = "results/qmix_summary.csv"

BASE_YAML = """data_dir: ../ts_quantum/datasets
dataset: {ds}
horizon: {L}
lookback: {L}
model:
  qmix_layers: {qmix_layers}
  qmix_norm: {qmix_norm}
  head_agg: {head_agg}
  spectrum_inject: {spectrum_inject}
  kernel_T: {kernel_T}
  topk: {topk}
  n_qubits: {n_qubits}
  d_token: 512
  entangle_topo: linear
  kernel_fn: quantum
  n_layers: 2
  revin_affine: true
  spectrum_M: 32
  spectrum_amp_normalize: false
  spectrum_freq_align: true
  spectrum_range: '0_2'
  spectrum_time_align: true
  theta_S_scale0: 0.5
  use_H: true
  use_S: true
  use_fmap: true
  use_periodic_feat: true
  use_spectrum: true
run_name: {run_id}
save_dir: {save_dir}
seed: {seed}
train:
  accumulation_steps: 1
  batch_size: 32
  epochs: 50
  eval_test_every_epoch: false
  lr: 0.0001
  num_workers: 0
  patience: 8
  proj_weight_decay: 0.0
  stride: 1
  use_amp: false
  weight_decay: 1.0e-05
"""

VARIANTS = {
    # 2026-08-12 选择性修复（正交开关：温度 × n_qubits × topk × 位置）
    "qmix_T10":     dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1,  n_qubits=8),
    "qmix_T10_n4":  dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1,  n_qubits=4),
    "qmix_head_T10":dict(qmix_layers=0, head_agg=True,       kernel_T=0.1,  n_qubits=8),
    "qmix_topk":    dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1,  topk=2, n_qubits=8),
    # 基准（plain 永不变）
    "plain":        dict(qmix_layers=0, qmix_norm="avg",     head_agg=False, spectrum_inject=False,
                         kernel_T=1.0, topk=0, n_qubits=8),
}
# 快速验证集（用户限定）：3 数据集 × 4 档位 × 1 seed
CELLS = [("etth1", 96), ("etth1", 192), ("etth1", 336), ("etth1", 720),
         ("weather", 96), ("weather", 192), ("weather", 336), ("weather", 720),
         ("electricity", 96), ("electricity", 192), ("electricity", 336), ("electricity", 720)]
SEEDS = [42]


def make_yaml(ds, L, seed, variant):
    run_id = f"qm_{variant}_{ds}_{L}_{seed}"
    yaml_path = os.path.join(YAML_DIR, f"{run_id}.yaml")
    flags = VARIANTS[variant]
    text = BASE_YAML.format(
        ds=ds, L=L, seed=seed, run_id=run_id,
        save_dir=os.path.join(ROOT, SAVE_DIR),
        qmix_layers=flags["qmix_layers"],
        qmix_norm=flags.get("qmix_norm", "avg"),
        head_agg="true" if flags.get("head_agg", False) else "false",
        spectrum_inject="true" if flags.get("spectrum_inject", False) else "false",
        kernel_T=flags.get("kernel_T", 1.0),
        topk=flags.get("topk", 0),
        n_qubits=flags.get("n_qubits", 8),
    )
    with open(yaml_path, "w") as f:
        f.write(text)
    return yaml_path, run_id


def parse_log(path):
    txt = open(path).read()
    m = re.findall(r"Test MSE \(normalized\): ([\d.e+-]+)", txt)
    km = re.search(r"K stats: diag_mean=([\d.]+) offdiag_mean=([\d.]+) offdiag_std=([\d.]+)", txt)
    return {
        "mse_norm": float(m[-1]) if m else None,
        "k_off": float(km.group(2)) if km else None,
    }


def run_one(args):
    ds, L, seed, variant = args
    yaml_path, run_id = make_yaml(ds, L, seed, variant)
    done_path = os.path.join(DONE_DIR, f"{run_id}.done")
    if os.path.exists(done_path):
        return run_id, "skip", None
    log_path = os.path.join(LOG_DIR, f"{run_id}.log")
    t0 = time.time()
    # -u 无缓冲输出；timeout 3h（大 V cell 单跑可达 2h+）
    try:
        proc = subprocess.run([sys.executable, "-u", "run_dual_ae.py", "--config", yaml_path],
                              capture_output=True, text=True, timeout=10800)
        stdout = proc.stdout or ""
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or "") + f"\n[runner] TIMEOUT after 10800s\n"
        with open(log_path, "w") as f:
            f.write(stdout)
        return run_id, "timeout", (time.time() - t0) / 60
    except Exception as e:
        with open(log_path, "w") as f:
            f.write(f"[runner] exception: {e}\n")
        return run_id, "failed", (time.time() - t0) / 60
    with open(log_path, "w") as f:
        f.write(stdout + proc.stderr)
    elapsed = (time.time() - t0) / 60
    ok = "Test MSE (normalized):" in stdout
    if ok:
        open(done_path, "w").write("ok")
    return run_id, "ok" if ok else "failed", elapsed


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--scope", default=None, help="如 'etth1:96+720'，逗号分隔数据集")
    ap.add_argument("--variants", default=None, help="变体子集，如 'qmix+qmix_sin'")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    variants = {k: v for k, v in VARIANTS.items() if k in args.variants.split("+")} if args.variants else VARIANTS

    if args.scope:
        jobs = []
        for part in args.scope.split(","):
            ds, _, ls = part.partition(":")
            for L in [int(x) for x in ls.split("+")]:
                for s in SEEDS:
                    for v in variants:
                        jobs.append((ds, L, s, v))
    else:
        jobs = [(ds, L, s, v) for ds, L in CELLS for s in SEEDS for v in variants]
    total = len(jobs) if args.max is None else min(len(jobs), args.max)
    print(f"变体: {list(variants)}；共 {total} 个实验（快 cell 优先），2 并发", flush=True)

    os.makedirs(YAML_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(DONE_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    if args.dry:
        for j in jobs[:total]:
            _, run_id = make_yaml(*j)
            done = "已存在 .done（将跳过）" if os.path.exists(os.path.join(DONE_DIR, f"{run_id}.done")) else "待跑"
            print(f"  {run_id:36s} {done}")
        print(f"[dry] 已生成 {len(jobs[:total])} 个 yaml 到 {YAML_DIR}/")
        return

    results = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs[:total]}
        done = 0
        for fut in as_completed(futs):
            try:
                run_id, status, elapsed = fut.result()
            except Exception as e:
                print(f"[runner] 任务异常: {e}", flush=True)
                continue
            done += 1
            log_path = os.path.join(LOG_DIR, f"{run_id}.log")
            p = parse_log(log_path) if os.path.exists(log_path) else {}
            fmt = lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else "-"
            el_s = f"({elapsed:.1f}min)" if elapsed is not None else ""
            print(f"[{done}/{total}] {run_id}: {status} mse_norm={fmt(p.get('mse_norm'))} "
                  f"k_off={fmt(p.get('k_off'))} {el_s}", flush=True)
            results.append((run_id, status, p.get("mse_norm"), p.get("k_off"), elapsed))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "variant", "dataset", "L", "seed", "status", "mse_norm", "k_offdiag", "elapsed_min"])
        for run_id, status, mn, ko, el in sorted(results, key=lambda r: r[0]):
            # run_id = qm_{variant}_{ds}_{L}_{seed}
            rid = run_id[len("qm_"):]
            variant = next((v for v in VARIANTS if rid.startswith(v + "_")))
            rest = rid[len(variant) + 1:]
            ds, L, s = rest.rsplit("_", 2)
            rnd = lambda x: round(x, 4) if isinstance(x, (int, float)) else ""
            w.writerow([run_id, variant, ds, L, s, status, rnd(mn), rnd(ko),
                        round(el, 1) if el else ""])
    print(f"\n完成。汇总: {OUT_CSV}")


if __name__ == "__main__":
    main()
