#!/usr/bin/env python3
"""并行批量实验运行器 — 最大化 24GB VRAM 利用。

原理：
    run_benchmark.py 默认在一个进程内顺序跑多个 (L, H) 设置。
    显存浪费严重（4 个设置串行，实际只用了 4-5GB 中的一个）。

    本脚本：
    1. 从 base config 为每个 (L, H) 生成 per-L 子配置文件
    2. 同时启动多个子进程，各跑一个 (L, H)，直到撑满 VRAM
    3. 所有进程完成后汇总结果

用法示例：
    # 单 config（4 个 L 并行，vs 原来串行）
    python batch_runner.py --config configs/e2_traffic_baseline.yaml --gpu 0

    # 队列文件（每行一个 config）
    python batch_runner.py --queue batch_queue.txt --gpu 0

    # 手动指定并发数
    python batch_runner.py --config configs/e2_traffic_qcc.yaml --gpu 0 --parallel 4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import signal
from pathlib import Path
from typing import Optional

import yaml

# 每变量每进程的显存估算 (GB) —— batch_size=32, d_token=128 下的经验值
# 实际受 batch_size 和 QCC 影响，这里是保守估计
EST_VRAM_PER_VAR = 0.006  # GB / variable

# 各 config 的已知 batch_size 和 QCC 系数
QCC_VRAM_MULTIPLIER = 1.8  # QCC 比 baseline 多 ~80% 显存


def estimate_vram(cfg_path: str) -> float:
    """估算一个进程的显存需求 (GB)。"""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    num_var = cfg.get("model", {}).get("num_var", 321)
    batch_size = cfg.get("batch_size", 32)
    method = cfg.get("method", {})
    is_qcc = method.get("use_qcc", True) and method.get("kernel", "quantum") in ("quantum",)

    # 基准：batch_size=32, d_token=128 下每变量 ~0.006GB
    base = num_var * EST_VRAM_PER_VAR
    # batch_size 线性缩放
    scaled = base * (batch_size / 32)
    # QCC 修正
    if is_qcc:
        scaled *= QCC_VRAM_MULTIPLIER
    # 保底下限
    return max(scaled, 1.5)


class ParallelRunner:
    """并行实验调度器。"""

    def __init__(
        self,
        gpu: int = 0,
        max_vram: float = 22,  # 留 2GB 给系统
        parallel: Optional[int] = None,
        out_dir: str = "results",
        config_dir: str = "_batch_configs",
        log_dir: str = "_batch_logs",
    ):
        self.gpu = gpu
        self.max_vram = max_vram
        self.parallel = parallel  # None = 自动
        self.out_dir = out_dir
        self.config_dir = Path(config_dir)
        self.log_dir = Path(log_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.processes: list[dict] = []
        self.running = True
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, sig, frame):
        print("\n⏹ 收到 SIGINT，正在终止所有子进程...")
        self.running = False
        for p in self.processes:
            if p["proc"].poll() is None:
                p["proc"].terminate()
        sys.exit(1)

    def generate_per_l_config(self, base_cfg: str, lookback: int, horizon: int) -> Path:
        """为单个 (L, H) 生成子配置文件。"""
        with open(base_cfg) as f:
            cfg = yaml.safe_load(f)

        # 只保留一个 setting
        cfg["settings"] = [[lookback, horizon]]

        # 输出文件名
        base_name = Path(base_cfg).stem
        out_name = f"{base_name}_L{lookback}H{horizon}.yaml"
        out_path = self.config_dir / out_name

        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        return out_path

    def _pick_batch(self, jobs: list[dict]) -> list[dict]:
        """从待办队列中选出能塞满 VRAM 的一个批次。"""
        if self.parallel is not None:
            return jobs[:self.parallel]

        batch = []
        total = 0.0
        for job in jobs:
            vram = job.get("vram", estimate_vram(job["cfg"]))
            if total + vram <= self.max_vram:
                batch.append(job)
                total += vram
            else:
                break
        # 至少返回 1 个
        if not batch and jobs:
            batch = [jobs[0]]
        return batch

    def run_experiment(self, job: dict, workdir: str) -> Optional[str]:
        """启动一个实验子进程。"""
        cfg_path = job["cfg"]
        log_path = job["log_path"]
        label = job["label"]

        log_file = open(log_path, "w")
        env = os.environ.copy()
        env["PYTHONPATH"] = workdir + ":" + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            [sys.executable, "-B", "run_benchmark.py",
             "--config", str(cfg_path),
             "--gpu", str(self.gpu),
             "--out", self.out_dir],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=workdir,
        )

        entry = {
            "label": label,
            "proc": proc,
            "cfg": cfg_path,
            "log": log_path,
            "vram": job.get("vram", estimate_vram(cfg_path)),
            "log_file": log_file,
        }
        self.processes.append(entry)
        print(f"  ▶ [{label}] PID={proc.pid}  (估计 {entry['vram']:.1f}GB)")
        return label

    def wait_any(self) -> None:
        """等待任意一个进程完成。"""
        while self.running and self.processes:
            for entry in list(self.processes):
                proc = entry["proc"]
                ret = proc.poll()
                if ret is not None:
                    entry["log_file"].close()
                    status = "✅" if ret == 0 else "❌"
                    print(f"  {status} [{entry['label']}] 退出码={ret}")
                    self.processes.remove(entry)
                    return
            time.sleep(2)

    def wait_all(self) -> None:
        """等待所有进程完成。"""
        while self.running and self.processes:
            self.wait_any()

    def run_config(self, base_cfg: str, workdir: str) -> None:
        """并行跑一个 config 的所有 setting。"""
        with open(base_cfg) as f:
            cfg = yaml.safe_load(f)

        settings = cfg.get("settings", [[96, 96]])
        label_prefix = Path(base_cfg).stem

        # 生成所有 per-L 任务
        jobs = []
        for L, H in settings:
            per_l_cfg = self.generate_per_l_config(base_cfg, L, H)
            label = f"{label_prefix}_L{L}"
            log_path = self.log_dir / f"{label}.log"
            jobs.append({
                "cfg": str(per_l_cfg),
                "label": label,
                "log_path": str(log_path),
                "vram": estimate_vram(str(per_l_cfg)),
            })

        print(f"\n{'='*60}")
        print(f"Config: {base_cfg}")
        print(f"Settings: {settings}")
        vram_strs = [f"{j['vram']:.1f}GB" for j in jobs]
        print(f"VRAM 需求: {vram_strs}")
        print(f"{'='*60}")

        completed = set()
        while len(completed) < len(jobs):
            remaining = [j for j in jobs if j["label"] not in completed]
            batch = self._pick_batch(remaining)

            if not batch:
                print("⚠️  batch 为空，等待中...")
                self.wait_any()
                continue

            print(f"\n  启动批次 ({len(batch)} 个进程):")
            for job in batch:
                self.run_experiment(job, workdir)

            # 等待这一批完成
            for _ in batch:
                self.wait_any()

            for job in batch:
                completed.add(job["label"])

            # 检查 GPU 是否还有空闲
            used_vram = sum(j["vram"] for j in self.processes)
            if used_vram < self.max_vram * 0.5:
                print(f"  当前使用 {used_vram:.1f}GB/{self.max_vram}GB, 继续下一批")

        self.wait_all()
        print(f"\n✅ Config {base_cfg} 全部完成\n")


def create_queue_file():
    """创建默认实验队列文件（到所有待跑数据集）。"""
    datasets = {
        "weather": {"num_var": 21, "batch_size": 64},
        "etth1": {"num_var": 7, "batch_size": 128},
        "etth2": {"num_var": 7, "batch_size": 128},
        "ettm1": {"num_var": 7, "batch_size": 128},
        "ettm2": {"num_var": 7, "batch_size": 128},
    }

    template_baseline = """# E2 {dataset} 基线（S-Mamba 无旁路）
experiment: e2_{dataset}_baseline
dataset: {dataset}
batch_size: {batch_size}
num_workers: 0

split:
  train_ratio: 0.7
  test_ratio: 0.2

model:
  num_var: {num_var}
  d_token: 128
  use_periodic_feat: true
  revin_affine: true

training:
  epochs: 100
  patience: 10
  lr: 0.0001
  weight_decay: 0.0001
  beta: 0.1
  alpha0: 0.1
  seed_base: 2026
  n_seeds: 1

method:
  use_qcc: true
  use_fmap: false
  kernel: none

settings:
  - [96, 96]
  - [192, 192]
  - [336, 336]
  - [720, 720]
"""

    template_qcc = """# E2 {dataset} QCC-Quantum
experiment: e2_{dataset}_qcc
dataset: {dataset}
batch_size: {batch_size}
num_workers: 0

split:
  train_ratio: 0.7
  test_ratio: 0.2

model:
  num_var: {num_var}
  d_token: 128
  use_periodic_feat: true
  revin_affine: true

training:
  epochs: 100
  patience: 10
  lr: 0.0001
  weight_decay: 0.0001
  beta: 0.1
  alpha0: 0.1
  seed_base: 2026
  n_seeds: 1

method:
  use_qcc: true
  use_fmap: true
  n_qubits: 8
  n_layers: 2
  entangle_topo: linear
  encode_gate: R_Y
  kernel: quantum

settings:
  - [96, 96]
  - [192, 192]
  - [336, 336]
  - [720, 720]
"""

    os.makedirs("configs", exist_ok=True)
    created = []

    for ds, info in datasets.items():
        # 基线
        bl_path = f"configs/e2_{ds}_baseline.yaml"
        if not os.path.exists(bl_path):
            with open(bl_path, "w") as f:
                f.write(template_baseline.format(
                    dataset=ds,
                    num_var=info["num_var"],
                    batch_size=info["batch_size"],
                ))
            created.append(bl_path)
            print(f"  ✅ 创建: {bl_path}")

        # QCC
        qcc_path = f"configs/e2_{ds}_qcc.yaml"
        if not os.path.exists(qcc_path):
            with open(qcc_path, "w") as f:
                f.write(template_qcc.format(
                    dataset=ds,
                    num_var=info["num_var"],
                    batch_size=info["batch_size"],
                ))
            created.append(qcc_path)
            print(f"  ✅ 创建: {qcc_path}")

    # 创建 traffic QCC per-L configs
    traffic_qcc_template = """# E2 traffic QCC L={lookback}
experiment: e2_traffic_qcc
dataset: traffic
batch_size: 16
num_workers: 0

split:
  train_ratio: 0.7
  test_ratio: 0.2

model:
  num_var: 862
  d_token: 128
  use_periodic_feat: true
  revin_affine: true

training:
  epochs: 100
  patience: 10
  lr: 0.0001
  weight_decay: 0.0001
  beta: 0.1
  alpha0: 0.1
  seed_base: 2026
  n_seeds: 1

method:
  use_qcc: true
  use_fmap: true
  n_qubits: 8
  n_layers: 2
  entangle_topo: linear
  encode_gate: R_Y
  kernel: quantum

settings:
  - [{lookback}, {lookback}]
"""
    for L in [96, 192, 336, 720]:
        path = f"configs/e2_traffic_qcc_L{L}.yaml"
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(traffic_qcc_template.format(lookback=L))
            created.append(path)
            print(f"  ✅ 创建: {path}")

    print(f"\n共创建 {len(created)} 个配置文件" if created else "\n所有配置文件已存在")
    return created


def main():
    parser = argparse.ArgumentParser(description="并行批量实验运行器")
    parser.add_argument("--config", help="单 config 文件路径")
    parser.add_argument("--queue", help="队列文件（每行一个 config 路径）")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 编号")
    parser.add_argument("--max-vram", type=float, default=22, help="最大 VRAM 用量 (GB)")
    parser.add_argument("--parallel", type=int, default=None, help="强制并发进程数")
    parser.add_argument("--generate-configs", action="store_true",
                        help="仅为所有数据集创建配置文件，不运行")
    parser.add_argument("--list-configs", action="store_true",
                        help="列出当前 configs 目录中的 E2 configs")
    args = parser.parse_args()

    workdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(workdir)

    if args.list_configs:
        for f in sorted(Path("configs").glob("e2_*.yaml")):
            size = os.path.getsize(f)
            print(f"  {f.name}  ({size} bytes)")
        return

    if args.generate_configs:
        create_queue_file()
        return

    runner = ParallelRunner(
        gpu=args.gpu,
        max_vram=args.max_vram,
        parallel=args.parallel,
    )

    if args.config:
        runner.run_config(args.config, workdir)
    elif args.queue:
        with open(args.queue) as f:
            configs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        for cfg_path in configs:
            if not os.path.exists(cfg_path):
                print(f"⚠️  跳过: {cfg_path} 不存在")
                continue
            runner.run_config(cfg_path, workdir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
