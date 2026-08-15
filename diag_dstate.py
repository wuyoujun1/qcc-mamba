#!/usr/bin/env python3
"""查 D 状态进程卡在哪个文件系统/设备。"""
import os, collections

def read(fn):
    try:
        with open(fn) as f:
            return f.read().strip()
    except Exception:
        return ""

d_by_comm = collections.Counter()
d_by_cwd = collections.Counter()
d_by_io = collections.Counter()
n = 0
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
        if parts[2] != "D":
            continue
        n += 1
        comm = read(f"/proc/{pid}/comm")
        d_by_comm[comm] += 1
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except Exception:
            cwd = "?"
        d_by_cwd[cwd] += 1
        io = read(f"/proc/{pid}/io").replace("\n", " | ")[:100]
        d_by_io[io[:60]] += 1
    except Exception:
        pass

print(f"D 状态总数: {n}\n")
print("按进程名:", dict(d_by_comm.most_common(10)))
print("\n按 cwd(挂载点):")
for k, v in d_by_cwd.most_common(10):
    print(f"  {v:5d}  {k}")
print("\n挂载点:")
print(read("/proc/mounts"))
