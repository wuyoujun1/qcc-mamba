#!/usr/bin/env python3
"""并发标定：同时跑 N 个冒烟作业，采样 GPU 显存/利用率，测单作业耗时。

目的：决定 batch 驱动 --parallel 用几路并发。
方法：用 ECL L=96 epochs=1 dual（~3min）冒烟配置，N 路并发，
     后台线程每 2s 采样 nvidia-smi 的 memory.used / utilization.gpu。
"""
import os
import subprocess
import sys
import threading
import time

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(ROOT, "run_dual_ae.py")

BASE_CFG = {
    "dataset": "electricity",
    "lookback": 96,
    "horizon": 96,
    "model": {
        "d_token": 512, "n_qubits": 10, "n_layers": 2, "entangle_topo": "linear",
        "kernel_fn": "quantum", "use_fmap": True, "alpha0": 0.1, "theta_S_scale0": 0.5,
        "beta": 0.1, "use_periodic_feat": True, "revin_affine": True,
        "use_spectrum": True, "spectrum_M": 32, "spectrum_range": "0_2",
        "spectrum_amp_normalize": False, "spectrum_time_align": True,
        "spectrum_freq_align": True, "use_H": True, "use_S": True,
    },
    "train": {
        "batch_size": 32, "epochs": 1, "patience": 1, "lr": 1e-4,
        "weight_decay": 1e-5, "proj_weight_decay": 0.0, "use_amp": False,
        "accumulation_steps": 1, "stride": 1, "num_workers": 0,
        "eval_test_every_epoch": False,
    },
}


def make_cfg(tag):
    cfg = yaml.safe_load(yaml.safe_dump(BASE_CFG))  # deep copy
    cfg["seed"] = 42
    cfg["save_dir"] = os.path.join(ROOT, "results", "calib", tag)
    cfg["run_name"] = tag
    p = os.path.join(ROOT, "configs", "generated", f"calib_{tag}.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f)
    return p


def gpu_sampler(interval, stop, out):
    """每 interval 秒采样一次 nvidia-smi memory/利用率的 mean/max。"""
    import re
    mems, utils = [], []
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            m = re.findall(r"(\d+),\s*(\d+)", r.stdout)
            if m:
                mems.append(int(m[0][0]))
                utils.append(int(m[0][1]))
        except Exception:
            pass
        time.sleep(interval)
    out["mem_max"] = max(mems) if mems else 0
    out["mem_mean"] = sum(mems) / len(mems) if mems else 0
    out["util_max"] = max(utils) if utils else 0
    out["util_mean"] = sum(utils) / len(utils) if utils else 0


def run_one(cfg_path, log_path):
    with open(log_path, "w") as flog:
        t0 = time.time()
        r = subprocess.run([sys.executable, RUNNER, "--config", cfg_path],
                           stdout=flog, stderr=subprocess.STDOUT)
        return time.time() - t0, r.returncode


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    tag = sys.argv[2] if len(sys.argv) > 2 else f"x{N}"
    stop = threading.Event()
    out = {}
    thr = threading.Thread(target=gpu_sampler, args=(1.5, stop, out), daemon=True)
    thr.start()

    jobs = [(make_cfg(f"{tag}_{i}"), os.path.join(ROOT, "logs", "batch", f"calib_{tag}_{i}.log"))
            for i in range(N)]
    t0 = time.time()
    results = [None] * N
    # 顺序启动，全并发
    procs = []
    for cfgp, logp in jobs:
        with open(logp, "w") as flog:
            procs.append((subprocess.Popen(
                [sys.executable, RUNNER, "--config", cfgp],
                stdout=flog, stderr=subprocess.STDOUT), cfgp, logp))
    for i, (p, cfgp, logp) in enumerate(procs):
        rc = p.wait()
        results[i] = rc
    total = time.time() - t0
    stop.set()
    thr.join()

    print(f"\n=== 并发标定: N={N}, 总墙钟 {total:.1f}s ({total/60:.1f}min) ===")
    print(f"  串行等价时间: {total*N:.0f}s (单作业 × {N})")
    print(f"  加速比(相对串行): {N*total/total:.1f}x  → 实际吞吐 = {N} 作业 / {total/60:.1f}min")
    print(f"  GPU 显存: mean={out['mem_mean']:.0f}MiB max={out['mem_max']:.0f}MiB / 24564MiB")
    print(f"  GPU 利用率: mean={out['util_mean']:.0f}% max={out['util_max']:.0f}%")


if __name__ == "__main__":
    main()
