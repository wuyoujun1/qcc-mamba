"""用新配置（β=1.0, α0=0.3）重跑 P1F 矩阵的 dual 部分。

矩阵：6 数据集 × 4 档位（96/192/336/720）× 2 seed（42/2024）= 48 个实验。
2 并发跑 GPU，断点续跑（.done 标记），结果汇总到 results/fix_matrix_summary.csv。
baseline/h_only/s_only 已有数字（results/summary.csv），无需重跑。
"""
import os, re, subprocess, sys, time, csv
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
YAML_DIR = "configs/fix_matrix"
LOG_DIR = "logs/matrix_fix"
DONE_DIR = "results/fix_matrix_done"
SAVE_DIR = "results/fix_matrix"
OUT_CSV = "results/fix_matrix_summary.csv"

# 顺序按信息价值排列：已知赢区（etth1/pems_bay）先行，便于中途检查
DATASETS = ["etth1", "pems_bay", "electricity", "weather", "chinaaqi", "metr_la"]
HORIZONS = [96, 192, 336, 720]
SEEDS = [42, 2024]

os.makedirs(YAML_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DONE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

def make_yaml(ds, L, seed):
    """镜像 p1f 配置，只改 beta=1.0 / alpha0=0.3。"""
    run_id = f"fix_{ds}_{L}_{seed}"
    yaml_path = f"{YAML_DIR}/{run_id}.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""data_dir: ../ts_quantum/datasets
dataset: {ds}
horizon: {L}
lookback: {L}
model:
  alpha0: 0.3
  beta: 1.0
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
save_dir: {os.path.join(ROOT, SAVE_DIR)}
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
""")
    return yaml_path, run_id

def parse_log(path):
    txt = open(path).read()
    m = re.findall(r"Test MSE \(normalized\): ([\d.e+-]+)", txt)
    km = re.search(r"K stats: diag_mean=([\d.]+) offdiag_mean=([\d.]+) offdiag_std=([\d.]+)", txt)
    am = re.findall(r"α=([\d.]+)", txt)
    return {
        "mse_norm": float(m[-1]) if m else None,
        "mae_norm": float(re.findall(r"Test MAE \(normalized\): ([\d.e+-]+)", txt)[-1]) if re.findall(r"Test MAE \(normalized\): ([\d.e+-]+)", txt) else None,
        "k_off": float(km.group(2)) if km else None,
        "alpha_end": float(am[-1]) if am else None,
    }

def run_one(args):
    ds, L, seed = args
    yaml_path, run_id = make_yaml(ds, L, seed)
    done_path = os.path.join(DONE_DIR, f"{run_id}.done")
    if os.path.exists(done_path):
        return run_id, "skip", None
    log_path = f"{LOG_DIR}/{run_id}.log"
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, "run_dual_ae.py", "--config", yaml_path],
                              capture_output=True, text=True, timeout=5400)
        stdout = proc.stdout or ""
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or "") + f"\n[runner] TIMEOUT after 5400s\n"
        with open(log_path, "w") as f:
            f.write(stdout)
        return run_id, "timeout", (time.time() - t0) / 60
    except Exception as e:  # 单任务异常不拖垮整个批
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
    ap.add_argument("--max", type=int, default=None, help="跑到 N 个完成后停止")
    ap.add_argument("--scope", default=None, help="作用域，如 'etth1:96,192,336,720' 或 'etth1*:96,336'，逗号分隔多个；默认全矩阵")
    args = ap.parse_args()
    if args.scope:
        jobs = []
        for part in args.scope.split(","):
            ds, _, ls = part.partition(":")
            for L in [int(x) for x in ls.split("+")]:
                for s in SEEDS:
                    jobs.append((ds, L, s))
    else:
        jobs = [(ds, L, s) for ds in DATASETS for L in HORIZONS for s in SEEDS]
    total = len(jobs) if args.max is None else min(len(jobs), args.max)
    print(f"共 {total} 个实验（顺序：{DATASETS}），2 并发后台运行", flush=True)
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
            log_path = f"{LOG_DIR}/{run_id}.log"
            p = parse_log(log_path) if os.path.exists(log_path) else {}
            results.append((run_id, status, p.get("mse_norm"), p.get("mae_norm"), p.get("k_off"), p.get("alpha_end"), elapsed))
            print(f"[{done}/{total}] {run_id}: {status} mse_norm={p.get('mse_norm')} ({elapsed:.1f}min)", flush=True)
    # 写汇总
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "dataset", "L", "seed", "status", "mse_norm", "mae_norm", "k_off", "alpha_end", "elapsed_min"])
        for run_id, status, mn, man, ko, ae, el in sorted(results):
            _, ds, L, s = run_id.split("_")
            w.writerow([run_id, ds, L, s, status, mn, man, ko, ae, round(el, 1) if el else ""])
    print(f"\n完成。汇总: {OUT_CSV}")

if __name__ == "__main__":
    main()
