#!/usr/bin/env python3
"""杀掉卡死的 ps/pkill/pgrep/atop 诊断进程（D 状态风暴的根源）。"""
import os, signal

killed = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
        st = parts[2]
        with open(f"/proc/{pid}/comm") as f:
            comm = f.read().strip()
        if comm in ("ps", "pkill", "pgrep", "atop") and st == "D":
            os.kill(int(pid), signal.SIGKILL)
            killed.append((pid, comm, st))
    except Exception:
        pass

print(f"已对 {len(killed)} 个卡死进程发 SIGKILL:")
for pid, comm, st in killed[:20]:
    print(f"  pid={pid} {comm} ({st})")
if len(killed) > 20:
    print(f"  ... 共 {len(killed)} 个")
print("loadavg:", open("/proc/loadavg").read().strip())
