#!/usr/bin/env python3
"""诊断：单任务训练时，计算到底跑在 GPU 还是 CPU？

采样 GPU 显存/利用率 + 训练进程 CPU%，并在训练结束时检查日志确认 device。
"""
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(ROOT, "configs", "generated", "smoke_ecl_96_dual.yaml")
LOG = os.path.join(ROOT, "logs", "batch", "diag_single.log")

# 1) CUDA 可用性（在主进程检查）
import torch
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device count = {torch.cuda.device_count()}")
    print(f"device 0     = {torch.cuda.get_device_name(0)}")
    print(f"total mem    = {torch.cuda.get_device_properties(0).total_memory / 2**20:.0f} MiB")

# 2) 后台跑冒烟训练
with open(LOG, "w") as f:
    t0 = time.time()
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "run_dual_ae.py"),
                          "--config", CFG], stdout=f, stderr=subprocess.STDOUT)

    gpu_mem, gpu_util, cpu_pct, cpu_threads = [], [], [], []
    while p.poll() is None:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            m = re.findall(r"(\d+),\s*(\d+)", r.stdout)
            if m:
                gpu_mem.append(int(m[0][0])); gpu_util.append(int(m[0][1]))
            # 训练进程 CPU%（ps 累加多线程）
            r2 = subprocess.run(["ps", "-o", "pcpu=", "-C", "python"],
                                capture_output=True, text=True, timeout=5)
            vals = [float(x) for x in r2.stdout.split()]
            cpu_pct.append(sum(vals) / max(1, len(vals)))   # 平均每个 python 进程
            cpu_threads.append(len(vals))
        except Exception:
            pass
        time.sleep(2)
    p.wait()
    wall = time.time() - t0

with open(LOG) as f:
    text = f.read()

dev = re.search(r"Using device: (\S+)", text)
gpu_lines = [l for l in text.splitlines() if "device" in l.lower()]

print(f"\n=== 单任务诊断: 墙钟 {wall:.0f}s ===")
print(f"训练日志 device 行: {dev.group(0) if dev else '未找到'}")
print(f"GPU 显存: mean={sum(gpu_mem)/max(1,len(gpu_mem)):.0f}MiB  max={max(gpu_mem) if gpu_mem else 0}MiB  (共24G)")
print(f"GPU 利用率: mean={sum(gpu_util)/max(1,len(gpu_util)):.0f}%  max={max(gpu_util) if gpu_util else 0}%")
print(f"python进程CPU: mean={sum(cpu_pct)/max(1,len(cpu_pct)):.0f}%  (同时存在的python进程数均值={sum(cpu_threads)/max(1,len(cpu_threads)):.1f})")
low = sum(1 for u in gpu_util if u < 50) / max(1, len(gpu_util))
print(f"GPU util<50% 占比: {low*100:.0f}%")
if gpu_mem and max(gpu_mem) < 2000:
    print(">>> ⚠️ GPU 显存峰值<2G：模型疑似没上 GPU（或在 CPU 上算）")
elif gpu_mem and max(gpu_mem) > 2000:
    print(">>> ✅ GPU 显存峰值 >2G：模型在 GPU 上，显存占用正常")
