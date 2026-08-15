#!/usr/bin/env python3
"""快速负载诊断：扫 /proc 找 CPU 大户、D 状态进程、进程总数、我们的残留。"""
import os, time

def read(fn):
    try:
        with open(fn) as f:
            return f.read().strip()
    except Exception:
        return ""

load = read("/proc/loadavg")
print("loadavg:", load)
total = dstate = 0
procs = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    total += 1
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
        st = parts[2]
        if st == "D":
            dstate += 1
        utime = int(parts[13]); stime = int(parts[14])
        procs.append((pid, st, utime + stime, read(f"/proc/{pid}/comm")))
    except Exception:
        pass

print(f"总进程/线程数: {total}  D状态(IO卡死): {dstate}")
procs.sort(key=lambda x: -x[2])
print("CPU 时间 TOP12:")
for pid, st, t, comm in procs[:12]:
    print(f"  pid={pid:>8s} state={st} cpu_ticks={t:>10d} {comm[:60]}")

# 我们的残留
ours = [p for p in procs if any(k in p[3] for k in ("run_qmix", "run_dual", "premise_disc", "qcc"))]
print(f"我们的进程残留: {len(ours)}")
for pid, st, t, comm in ours:
    print(f"  pid={pid} state={st} ticks={t} {comm}")
