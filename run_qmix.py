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
  offdiag: {offdiag}
  angle_norm: {angle_norm}
  gate: {gate}
  gate_init: {gate_init}
  hp_scale: {hp_scale}
  delay_in_s: {delay_in_s}
  aux_loss: {aux_loss}
  aux_beta: {aux_beta}
  kernel_sup: {kernel_sup}
  init_ckpt: {init_ckpt}
  n_qubits: {n_qubits}
  # P2-1 双路径（2026-08-15）：时间 SSM 单向 + 量子核独占跨变量
  dual_path: {dual_path}
  dp_time_layers: {dp_time_layers}
  dp_time_dim: {dp_time_dim}
  dp_time_pool: {dp_time_pool}
  dp_var_embed: {dp_var_embed}
  dp_msg: {dp_msg}
  dp_fusion: {dp_fusion}
  # QK-Path（量子核独立预测通道）
  qk_path: {qk_path}
  qk_gate_init: {qk_gate_init}
  qk_use_H: {qk_use_H}
  qk_norm: {qk_norm}
  d_token: 512
  entangle_topo: linear
  kernel_fn: {kernel_fn}
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
  gate_lr: {gate_lr}
  stride: 1
  use_amp: false
  weight_decay: 1.0e-05
"""

VARIANTS = {
    # 2026-08-13 轮 C：有向线性内积核 + 零门控（量子特有优势的第一次使用）
    # 保真度 |⟨ψ|ψ⟩|² 扔掉相位(有向时滞信息)且撞浓度定理；线性内积虚部反对称=有向、
    # 均值 0 无浓度问题；γ 门控 init=0 → 结构保证不更差
    "qdir":        dict(qmix_layers=2, qmix_norm="l1", kernel_fn="linear_imag",
                        gate=True, angle_norm="clamp", n_qubits=8),
    "qdir_real":   dict(qmix_layers=2, qmix_norm="l1", kernel_fn="linear_real",
                        gate=True, angle_norm="clamp", n_qubits=8),
    "qdir_n4":     dict(qmix_layers=2, qmix_norm="l1", kernel_fn="linear_imag",
                        gate=True, angle_norm="clamp", n_qubits=4),
    "qdir_sin":    dict(qmix_layers=2, qmix_norm="l1", kernel_fn="linear_imag",
                        gate=True, angle_norm="clamp", n_qubits=8, spectrum_inject=True),
    # 轮 D：给量子混合分支直接学习目标（辅助残差损失）——解决 γ 不开的"无信号"问题
    # L = MSE(y,y_true) + β·MSE(aux_head(LN(Hp)), residual.detach())；推理仍 y=y_main
    "qdir_aux":    dict(qmix_layers=2, qmix_norm="l1", kernel_fn="linear_imag",
                        gate=True, aux_loss=True, aux_beta=0.1, angle_norm="clamp", n_qubits=8),
    "qdir_aux_g":  dict(qmix_layers=2, qmix_norm="l1", kernel_fn="linear_imag",
                        gate=True, gate_init=0.05, aux_loss=True, aux_beta=0.1,
                        angle_norm="clamp", n_qubits=8),
    # 轮 F：缩小态空间（浓度定理正解）——n2/n3 保真度幅度 1/4~1/8，对齐结构可读
    # 轮 E 注释保留：核监督 + warm-start（量子核第一次被告知数据里的跨变量结构）
    # K 直接学 |corr(x_norm)|（相关矩阵比水平可迁移）；从 plain checkpoint 微调，
    # 主干不重学、混合分支只学增量；γ init 0.05 给引导性开口
    "qkern_sup":   dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        gate=True, gate_init=0.05, kernel_sup=0.1, warm_start=True,
                        angle_norm="clamp", n_qubits=8),
    "qkern_warm":  dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        gate=True, gate_init=0.05, kernel_sup=0.0, warm_start=True,
                        angle_norm="clamp", n_qubits=8),
    # 轮 F（2026-08-13）：小态空间 = 浓度定理正解
    # n=2 (4维)/n=3 (8维)：保真度幅度 1/4~1/8 相对波动 O(1)——对齐结构第一次可读，
    # 有向核 Im 幅度 0.3~0.5（不再是 1/256 噪声级）
    "qdir_n2":     dict(qmix_layers=2, qmix_norm="l1", kernel_fn="linear_imag",
                        gate=True, gate_init=0.05, angle_norm="clamp", n_qubits=2),
    "qdir_n3":     dict(qmix_layers=2, qmix_norm="l1", kernel_fn="linear_imag",
                        gate=True, gate_init=0.05, angle_norm="clamp", n_qubits=3),
    "qoff_n2":     dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        gate=True, gate_init=0.05, angle_norm="clamp", n_qubits=2),
    "qoff_n2_f":   dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        gate=False, angle_norm="clamp", n_qubits=2),
    # 轮 G：混合输出固定缩放（防 V=21 大变量集强信号发散）+ 全矩阵扩档
    "qoff_n2_s":   dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        gate=False, hp_scale=0.2, angle_norm="clamp", n_qubits=2),
    "qoff_n2_g":   dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        gate=True, gate_init=0.05, hp_scale=0.5, angle_norm="clamp", n_qubits=2),
    "qoff_n2_tk":  dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        topk=2, hp_scale=0.5, angle_norm="clamp", n_qubits=2),
    "qoff_n2_fs":  dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        gate=False, hp_scale=0.5, angle_norm="clamp", n_qubits=2),
    "qoff_n2_f1":  dict(qmix_layers=1, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        gate=False, angle_norm="clamp", n_qubits=2),
    # 轮 I：自适应满强度门控——γ init 开着头学（etth1 能全开，electricity/weather 自动关小）
    "qoff_n2_g2":  dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        gate=True, gate_init=0.5, angle_norm="clamp", n_qubits=2),
    "qoff_n2_g3":  dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        gate=True, gate_init=1.0, angle_norm="clamp", n_qubits=2),
    # 轮 J：γ 加速学习（实测 γ 50 epoch 只动 7%——梯度弱；gate_lr=0.01 让它按 cell 快速自适应）
    "qoff_n2_g4":  dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        gate=True, gate_init=1.0, gate_lr=0.01, angle_norm="clamp", n_qubits=2),
    # 轮 K：hp_scale = 7/V 按变量数归一化（第一性原理）——混合注入随 V 稀释：
    # etth1(V=7)→1.0 全强度(f 的赢面)、weather(V=21)→0.33(fs 区间)、electricity(V=321)→0.022(微调)
    # 一个配置自动适配所有数据集，无需学习
    "qoff_n2_v":   dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        gate=False, hp_scale_v=True, angle_norm="clamp", n_qubits=2),
    # 轮 L（electricity 对策）：v 归一化缩放 + topk=2 选择性混合——321 变量全对全核噪音大，
    # 只混合 K 最相关的 2 个邻居（决策树 D 步；tk 固定 0.5 缩放对 321 变量过大，改用 v 缩放）
    "qoff_n2_vtk": dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        topk=2, gate=False, hp_scale_v=True, angle_norm="clamp", n_qubits=2),
    # 轮 L（P0-1，ins.md 头号候选）：δ̂ 时滞入 S → S=[Ã; φ̃; δ̂]（65 维），δ̂ detach，
    # 让量子核同时度量"形状相似 + 时滞关系"（DeMa 赢的核心是时滞建模；时滞是主干零响应的独有信息）
    "qoff_n2_vd":  dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        gate=False, hp_scale_v=True, angle_norm="clamp", n_qubits=2,
                        delay_in_s=True),
    # P2-1 双路径（2026-08-15）：时间 SSM 单向 + 量子核独占跨变量
    # 主干不再扫变量轴（去掉隐式跨变量），量子核是唯一跨变量通道；
    # K 读 S（对齐频谱，主干拿不到）而非 H；softmax 行归一化消息 O(1)，无 1/V 稀释。
    # hp_scale_v 不需要：γ 门控是唯一强度调节器（init=0.05 引导开口 + gate_lr 快速自适应）
    "dp_time":  dict(qmix_layers=0, dual_path=True, dp_fusion="time_only",
                     dp_time_layers=2, dp_time_dim=256),  # 新 plain 基线（无跨变量通道）
    "dp":       dict(qmix_layers=0, dual_path=True, dp_fusion="add",
                     gate=True, gate_init=0.05, gate_lr=0.01,
                     kernel_T=0.1, offdiag=True, n_qubits=2),  # 主线：S 消息 + 门控融合
    "dp_ng":    dict(qmix_layers=0, dual_path=True, dp_fusion="add",
                     gate=False, kernel_T=0.1, offdiag=True, n_qubits=2),  # 无门控满强度
    "dp_con":   dict(qmix_layers=0, dual_path=True, dp_fusion="concat",
                     kernel_T=0.1, offdiag=True, n_qubits=2),  # concat 融合消融
    "dp_hmsg":  dict(qmix_layers=0, dual_path=True, dp_fusion="add", dp_msg="H",
                     gate=True, gate_init=0.05, gate_lr=0.01,
                     kernel_T=0.1, offdiag=True, n_qubits=2),  # 消息=H_time（信息重合对照）
    "dp_d":     dict(qmix_layers=0, dual_path=True, dp_fusion="add",
                     gate=True, gate_init=0.05, gate_lr=0.01,
                     kernel_T=0.1, offdiag=True, n_qubits=2, delay_in_s=True),  # +δ̂ 时滞
    # V2（2026-08-15，etth1:96 诊断后）：恢复 H 驱动 K（use_H 默认 true，
    # K 读 H_time + S 双阶段，随表征演化）+ 消息 = H_time（dp_msg="H"）。
    # V1 发现：K 纯 S 驱动消息=S 有害（dp 0.7084）、消息=H_time 有益（dp_hmsg 0.6572）
    "dp2":      dict(qmix_layers=0, dual_path=True, dp_fusion="add", dp_msg="H",
                     gate=True, gate_init=0.05, gate_lr=0.01,
                     kernel_T=0.1, offdiag=True, n_qubits=2),  # 主线：H 驱动 K + H 消息
    "dp2_ng":   dict(qmix_layers=0, dual_path=True, dp_fusion="add", dp_msg="H",
                     gate=False, kernel_T=0.1, offdiag=True, n_qubits=2),  # 无门控
    "dp2_con":  dict(qmix_layers=0, dual_path=True, dp_fusion="concat", dp_msg="H",
                     kernel_T=0.1, offdiag=True, n_qubits=2),  # concat 消融
    "dp2_both": dict(qmix_layers=0, dual_path=True, dp_fusion="add", dp_msg="both",
                     gate=True, gate_init=0.05, gate_lr=0.01,
                     kernel_T=0.1, offdiag=True, n_qubits=2),  # 消息 = S_emb + H_time
    "dp2_d":    dict(qmix_layers=0, dual_path=True, dp_fusion="add", dp_msg="H",
                     gate=True, gate_init=0.05, gate_lr=0.01,
                     kernel_T=0.1, offdiag=True, n_qubits=2, delay_in_s=True),  # +δ̂
    # el 根因诊断（2026-08-15）：K 均匀（offdiag_std=0.009≈噪声级）——321 变量在
    # n2 4维态空间互相淹没（浓度定理另一面）。三个对照区分"态空间容量 vs 数据无差异"：
    "qoff_n3_v":  dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                       gate=False, hp_scale_v=True, angle_norm="clamp", n_qubits=3),  # 8维态空间
    "qoff_n4_v":  dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                       gate=False, hp_scale_v=True, angle_norm="clamp", n_qubits=4),  # 16维
    "qoff_n2_rbf": dict(qmix_layers=2, qmix_norm="softmax", kernel_T=0.1, offdiag=True,
                        gate=False, hp_scale_v=True, angle_norm="clamp", n_qubits=2,
                        kernel_fn="rbf"),  # 同机制换经典核（数据无差异则 rbf 也均匀）
    "qoff_n2_dir": dict(qmix_layers=2, qmix_norm="l1", kernel_fn="linear_imag",
                        gate=False, hp_scale_v=True, angle_norm="clamp", n_qubits=2),  # 有向核无浓度问题
    # QK-Path（2026-08-15）：量子核独立预测通道 —— 主干 plain 最强形态 + 量子核独立
    # 出预测 y_qk，γ 门控融合（γ=0 → 精确 plain）。核形态消融（保真度/rbf/有向）：
    "qk":      dict(qmix_layers=0, qk_path=True, kernel_T=0.1, offdiag=True,
                    n_qubits=2, gate_lr=0.01),  # 保真度核（保留元件）
    "qk_rbf":  dict(qmix_layers=0, qk_path=True, kernel_fn="rbf", kernel_T=0.1,
                    offdiag=True, n_qubits=2, gate_lr=0.01),  # rbf（el 诊断有选择性）
    "qk_dir":  dict(qmix_layers=0, qk_path=True, kernel_fn="linear_imag", qk_norm="l1",
                    n_qubits=2, gate_lr=0.01),  # 有向核（无浓度问题）
    # 基准（plain 永不变）
    "plain":       dict(qmix_layers=0, qmix_norm="avg", head_agg=False, spectrum_inject=False,
                        kernel_T=1.0, topk=0, offdiag=False, angle_norm="clamp",
                        n_qubits=8, kernel_fn="quantum", gate=False),
}
# 快速验证集（用户限定）：3 数据集 × 4 档位 × 1 seed
CELLS = [("etth1", 96), ("etth1", 192), ("etth1", 336), ("etth1", 720),
         ("weather", 96), ("weather", 192), ("weather", 336), ("weather", 720),
         ("electricity", 96), ("electricity", 192), ("electricity", 336), ("electricity", 720)]
# 各数据集变量数（hp_scale_v 归一化用）
DATASET_VARS = {"etth1": 7, "weather": 21, "electricity": 321, "chinaaqi": 342}
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
        offdiag="true" if flags.get("offdiag", False) else "false",
        angle_norm=flags.get("angle_norm", "clamp"),
        gate="true" if flags.get("gate", False) else "false",
        gate_init=flags.get("gate_init", 0.0),
        gate_lr=flags.get("gate_lr", ""),
        hp_scale=(7.0 / DATASET_VARS.get(ds, 21) if flags.get("hp_scale_v", False)
                  else flags.get("hp_scale", 1.0)),
        aux_loss="true" if flags.get("aux_loss", False) else "false",
        aux_beta=flags.get("aux_beta", 0.1),
        kernel_sup=flags.get("kernel_sup", 0.0),
        init_ckpt=(os.path.join(ROOT, SAVE_DIR, f"qm_plain_{ds}_{L}_{seed}_best.pt")
                   if flags.get("warm_start", False) else ""),
        kernel_fn=flags.get("kernel_fn", "quantum"),
        n_qubits=flags.get("n_qubits", 8),
        delay_in_s="true" if flags.get("delay_in_s", False) else "false",
        # P2-1 双路径
        dual_path="true" if flags.get("dual_path", False) else "false",
        dp_time_layers=flags.get("dp_time_layers", 2),
        dp_time_dim=flags.get("dp_time_dim", 256),
        dp_time_pool=flags.get("dp_time_pool", "mean"),
        dp_var_embed="true" if flags.get("dp_var_embed", True) else "false",
        dp_msg=flags.get("dp_msg", "S"),
        dp_fusion=flags.get("dp_fusion", "add"),
        # QK-Path
        qk_path="true" if flags.get("qk_path", False) else "false",
        qk_gate_init=flags.get("qk_gate_init", 0.05),
        qk_use_H="true" if flags.get("qk_use_H", False) else "false",
        qk_norm=flags.get("qk_norm", "softmax"),
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
    ap.add_argument("--seeds", default=None, help="种子列表，如 '42+2024'（默认 SEEDS 常量）")
    ap.add_argument("--jobs", type=int, default=2, help="并发数（大变量 cell 用 1 更稳）")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split("+")] if args.seeds else SEEDS

    variants = {k: v for k, v in VARIANTS.items() if k in args.variants.split("+")} if args.variants else VARIANTS

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
    print(f"变体: {list(variants)}；共 {total} 个实验（快 cell 优先），{args.jobs} 并发", flush=True)

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
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
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
