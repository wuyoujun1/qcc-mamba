#!/usr/bin/env python3
"""架构修复实验运行器。

探针结论（analyze_bypass.py）：
  - S 信号到达 K（K 变化 33~75%）但死在 LN(H+Hp) 残差结构（修正量只变 2%）
  - corr 与 residual 相关≈0.02~0.11，α 冻结在初值 → dual ≡ h_only

四个修复变体（基座 = fix_matrix 新配置 β=1.0/α0=0.3）：
  s_concat  : corr_use_S=True   —— S_norm 直接拼进修正输入（绕开 K→Hp 死路）
  s_gate    : corr_gate_S=True  —— 频谱门控逐变量缩放修正
  hp_channel: corr_use_Hp=True  —— Hp 独立 LN 通道（不被 H 淹没）
  mlp_head  : corr_mlp=True     —— 修正头 MLP（残差非线性可读）

实验矩阵：3 cell（etth1 96/192, pems_bay 96）× 4 变体 × 2 seed = 24 runs。
2 并发，断点续跑（.done），结果汇总 results/archfix_summary.csv。

用法：
  python run_arch_fix.py                  # 全 24 个
  python run_arch_fix.py --dry            # 只生成 yaml + 打印任务列表，不运行
  python run_arch_fix.py --scope etth1:96 # 只看某个 cell
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
YAML_DIR = "configs/archfix"
LOG_DIR = "logs/archfix"
DONE_DIR = "results/archfix_done"
SAVE_DIR = "results/archfix"
OUT_CSV = "results/archfix_summary.csv"

BASE_YAML = """data_dir: ../ts_quantum/datasets
dataset: {ds}
horizon: {L}
lookback: {L}
model:
  alpha0: 0.3
  beta: 1.0
  corr_use_S: {corr_use_S}
  corr_gate_S: {corr_gate_S}
  corr_use_Hp: {corr_use_Hp}
  corr_mlp: {corr_mlp}
  d_token: 512
  entangle_topo: linear
  kernel_fn: quantum
  n_layers: 2
  n_qubits: 8
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

# 变体 → 模型开关（波次 1.5 筛选后：砍掉 s_gate/mlp_head，机制与 MSE 均无亮点）
VARIANTS = {
    "s_concat":   dict(corr_use_S=True),
    "hp_channel": dict(corr_use_Hp=True),
    "combo":      dict(corr_use_S=True, corr_use_Hp=True),  # S 直达 + Hp 独立通道
}
# 先跑信息量最大的快速 cell，便于中途检查
CELLS = [("etth1", 96), ("pems_bay", 96), ("etth1", 192)]
SEEDS = [42, 2024]


def make_yaml(ds, L, seed, variant):
    run_id = f"arch_{variant}_{ds}_{L}_{seed}"
    yaml_path = os.path.join(YAML_DIR, f"{run_id}.yaml")
    flags = VARIANTS[variant]
    text = BASE_YAML.format(
        ds=ds, L=L, seed=seed, run_id=run_id,
        save_dir=os.path.join(ROOT, SAVE_DIR),
        corr_use_S="true" if flags.get("corr_use_S") else "false",
        corr_gate_S="true" if flags.get("corr_gate_S") else "false",
        corr_use_Hp="true" if flags.get("corr_use_Hp") else "false",
        corr_mlp="true" if flags.get("corr_mlp") else "false",
    )
    with open(yaml_path, "w") as f:
        f.write(text)
    return yaml_path, run_id


def parse_log(path):
    txt = open(path).read()
    m = re.findall(r"Test MSE \(normalized\): ([\d.e+-]+)", txt)
    km = re.search(r"K stats: diag_mean=([\d.]+) offdiag_mean=([\d.]+) offdiag_std=([\d.]+)", txt)
    am = re.findall(r"α=([\d.]+)", txt)
    gm = re.findall(r"γ=([\d.]+)", txt)
    return {
        "mse_norm": float(m[-1]) if m else None,
        "k_off": float(km.group(2)) if km else None,
        "alpha_end": float(am[-1]) if am else None,
        "gamma_end": float(gm[-1]) if gm else None,
    }


def run_one(args):
    ds, L, seed, variant = args
    yaml_path, run_id = make_yaml(ds, L, seed, variant)
    done_path = os.path.join(DONE_DIR, f"{run_id}.done")
    if os.path.exists(done_path):
        return run_id, "skip", None
    log_path = os.path.join(LOG_DIR, f"{run_id}.log")
    t0 = time.time()
    # -u：无缓冲输出，否则 capture_output 下 epoch 打印堵在管道里（超时日志全空）
    # timeout 3h：pems_bay 96 等大 V cell 单跑可达 2h+（s_concat 71min 才早停）
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
    ap.add_argument("--max", type=int, default=None, help="最多跑 N 个完成的任务后停止")
    ap.add_argument("--scope", default=None, help="作用域，如 'etth1:96' 或 'etth1:96+192'；逗号分隔多个数据集")
    ap.add_argument("--variants", default=None, help="变体子集，如 's_concat+hp_channel'；默认全部")
    ap.add_argument("--seeds", default=None, help="seed 子集，如 '42'；默认 42+2024")
    ap.add_argument("--dry", action="store_true", help="只生成 yaml 并打印任务列表，不运行")
    args = ap.parse_args()

    variants = {k: v for k, v in VARIANTS.items() if k in args.variants.split("+")} if args.variants else VARIANTS
    seeds = [int(x) for x in args.seeds.split("+")] if args.seeds else SEEDS
    print(f"变体: {list(variants)}，seeds: {seeds}", flush=True)

    if args.scope:
        jobs = []
        for part in args.scope.split(","):
            ds, _, ls = part.partition(":")
            for L in [int(x) for x in ls.split("+")]:
                for s in seeds:
                    for v in variants:
                        jobs.append((ds, L, s, v))
    else:
        jobs = [(ds, L, s, v) for ds, L in CELLS for s in seeds for v in variants]
    total = len(jobs) if args.max is None else min(len(jobs), args.max)
    print(f"共 {total} 个实验（顺序：{[f'{ds}:{L}' for ds, L in CELLS]}，变体 {list(VARIANTS)}），2 并发", flush=True)

    os.makedirs(YAML_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(DONE_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    if args.dry:
        for j in jobs[:total]:
            yaml_path, run_id = make_yaml(*j)
            done = "已存在 .done（将跳过）" if os.path.exists(os.path.join(DONE_DIR, f"{run_id}.done")) else "待跑"
            print(f"  {run_id:40s} {done}")
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
            # 修复 matrix_fix 的 None 格式化崩溃：None 显示为 '-'
            fmt = lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else "-"
            el_s = f"({elapsed:.1f}min)" if elapsed is not None else ""
            print(f"[{done}/{total}] {run_id}: {status} mse_norm={fmt(p.get('mse_norm'))} "
                  f"k_off={fmt(p.get('k_off'))} α={fmt(p.get('alpha_end'))} {el_s}", flush=True)
            results.append((run_id, status, p.get("mse_norm"), p.get("k_off"),
                            p.get("alpha_end"), p.get("gamma_end"), elapsed))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "variant", "dataset", "L", "seed", "status",
                    "mse_norm", "k_offdiag", "alpha_end", "gamma_end", "elapsed_min"])
        for run_id, status, mn, ko, ae, ge, el in sorted(results, key=lambda r: r[0]):
            # run_id = arch_{variant}_{ds}_{L}_{seed}；variant 名可能含下划线（如 mlp_head），
            # 用已知变体名反查前缀，避免 split 错位（此前 CSV 崩溃的根因）
            rid = run_id[len("arch_"):]
            variant = next((v for v in VARIANTS if rid.startswith(v + "_")), rid)
            rest = rid[len(variant) + 1:]
            ds, L, s = rest.rsplit("_", 2)
            rnd = lambda x: round(x, 4) if isinstance(x, (int, float)) else ""
            w.writerow([run_id, variant, ds, L, s, status, rnd(mn), rnd(ko), rnd(ae), rnd(ge),
                        round(el, 1) if el else ""])
    print(f"\n完成。汇总: {OUT_CSV}")


if __name__ == "__main__":
    main()
