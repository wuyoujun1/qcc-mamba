#!/usr/bin/env python3
"""DualAE-QCC 批跑结果汇总：summary.csv → EXPERIMENT_SUMMARY.md + hetero_gain.png。

- 随时可跑（报告反映当前进度）。
- P2 生死线散点：x=异质强度 δ̂std%（缺的显示为占位），y=ΔMSE = MSE(no_align) − MSE(dual)。
- 数据集中异质度未回填的（ChinaAQI/METR_LA/PEMS_BAY）会在报告里标"待回填"。

用法：
    python summarize_results.py
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SUMMARY_CSV = os.path.join(ROOT, "results", "summary.csv")
REPORT = os.path.join(ROOT, "EXPERIMENT_SUMMARY.md")
PLOT = os.path.join(ROOT, "hetero_gain.png")
GEN_DIR = os.path.join(ROOT, "configs", "generated")
DONE_DIR = os.path.join(ROOT, "results", ".done")

# 异质强度 δ̂ std %（主导周期）。None = 待服务器回填（回填后散点图自动补全）
# 异质强度 = δ̂ std % 主导周期，来自 premise_verify.py / premise_results/premise_report.md §4
HETEROGENEITY = {
    "electricity": 20.5, "etth1": 28.6, "etth2": 20.9, "weather": 17.0,
    "chinaaqi": 25.5, "metr_la": 6.5, "pems_bay": 4.7,
    "ettm1": 6.9, "ettm2": 7.6, "traffic": 2.4, "exchange": 0.1,
}

# 主流技能分口径（社区标准做法，处理 test/train regime shift）：
#   skill = 1 − mse_norm / 常数基准  其中 mse_norm = MSE(test) / VAR(train)
#   常数基准 = "恒用 train 均值预测 test" 的 mse_norm（只依赖数据集+划分，与 L/配置无关）
#   每个数据集一个标量，由 ../ts_quantum/datasets/<file>.csv 按 7:1:2 划分算得。
#   意义：>0 = 模型比"无脑预测训练均值"好；0 附近 = 没学到规律；<0 = 比常数还差。
CONST_BASELINE = {
    "electricity": 1.0136, "chinaaqi": 0.9614, "metr_la": 1.4844,
    "etth1": 1.2082, "weather": 0.6363, "pems_bay": 1.1980,
}

# P1F 全矩阵配置名映射（summary.csv config 列 → 显示名）
P1F_CFG_LABEL = {"phase1": "dual", "ablation_h_only": "h_only",
                 "ablation_s_only": "s_only", "baseline_smamba": "baseline"}


def load_rows():
    if not os.path.exists(SUMMARY_CSV):
        return []
    with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(v, nd=4):
    if v in (None, ""):
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def p1_section(rows):
    lines = ["## P1：ECL×96 编码消融（2 seed 均值）", ""]
    # config 列 = 基 yaml 名去掉 dual_/.yaml：dual_baseline_smamba → baseline_smamba 等
    cfgs = [("baseline_smamba", "纯 S-Mamba（无旁路）"),
            ("ablation_h_only", "仅 H（语义路）"),
            ("ablation_s_only", "仅 S（频谱路）"),
            ("phase1", "双阶段（完整）")]
    table = ["| 配置 | 说明 | MSE_norm | MAE_norm | 完成 |", "|---|---|---|---|---|"]
    for tag, desc in cfgs:
        m = [r for r in rows if r["config"] == tag and r["phase"] == "phase1" and r["status"] == "ok"]
        if not m:
            table.append(f"| {tag} | {desc} | 待跑 | 待跑 | 0/{2} |")
            continue
        ok = sum(1 for r in m if r["test_mse"] not in ("", None))
        ms = np.mean([float(r["mse_norm"]) for r in m if r["mse_norm"] not in ("", None)])
        ma = np.mean([float(r["mae_norm"]) for r in m if r["mae_norm"] not in ("", None)])
        table.append(f"| {tag} | {desc} | {ms:.4f} | {ma:.4f} | {ok}/2 |")
    lines += table
    lines.append("")
    lines.append("判据：双阶段 ≤ 仅H ≤ 原版 → 旁路有效；仅S 明显差为正常（频谱无语义）。")
    lines.append("")
    return lines


def p2_section(rows):
    lines = ["## P2：生死线（6 数据集 × {dual, no_align} × L=96 × seed42）", ""]
    table = ["| 数据集 | 异质度 δ̂std% | MSE_norm(dual) | MSE_norm(no_align) | ΔMSE | 增益 |",
             "|---|---|---|---|---|---|"]
    gains = []
    for ds in ("electricity", "chinaaqi", "metr_la", "pems_bay", "etth1", "weather"):
        dual = [r for r in rows if r["phase"] == "phase2" and r["dataset"] == ds
                and r["config"] == "phase1" and r["status"] == "ok"]
        noa = [r for r in rows if r["phase"] == "phase2" and r["dataset"] == ds
               and r["config"] == "ablation_no_align" and r["status"] == "ok"]
        h = HETEROGENEITY.get(ds)
        hstr = fmt(h, 1) if h else "待回填"
        if not dual or not noa or dual[0]["mse_norm"] in ("", None) or noa[0]["mse_norm"] in ("", None):
            table.append(f"| {ds} | {hstr} | 待跑 | 待跑 | — | — |")
            continue
        d = float(dual[0]["mse_norm"])
        n = float(noa[0]["mse_norm"])
        d_mse = float(dual[0]["test_mse"]) if dual[0]["test_mse"] not in ("", None) else d
        n_mse = float(noa[0]["test_mse"]) if noa[0]["test_mse"] not in ("", None) else n
        # 增益优先用 mse_norm（跨数据集可比）；图里用 mse_norm
        delta = n - d
        table.append(f"| {ds} | {hstr} | {d:.4f} | {n:.4f} | {delta:+.4f} | {'✅' if delta > 0 else '❌'} |")
        if h is not None:
            gains.append((h, delta))
    lines += table
    lines.append("")
    lines.append(f"判断：ΔMSE 随异质度单调上升 → idea 成立；全部 ≈0 → 查频谱实现；强异质 ΔMSE<0 → 严重。")
    lines.append("")
    return lines, gains


def p3_section(rows):
    lines = ["## P3：对齐 + 参数消融（ECL×96 × seed42）", ""]
    table = ["| 消融 | 来源 | MSE_norm | MAE_norm |", "|---|---|---|---|"]
    # 对齐消融：time_only/freq_only 在 phase3；no_align 引用 phase2 ECL；双轴引用 phase1 ECL(seed42)
    refs = [
        ("仅时间轴", "phase3", "ablation_time_only", "p3"),
        ("仅频率轴", "phase3", "ablation_freq_only", "p3"),
        ("无对齐", "phase2", "ablation_no_align", "p2"),
        ("双轴对齐", "phase1", "phase1", "p1"),
    ]
    for desc, phase, config, tag in refs:
        m = [r for r in rows if r["phase"] == phase and r["config"] == config
             and r["dataset"] == "electricity" and r["status"] == "ok"]
        if tag == "p1":
            m = [r for r in m if r["seed"] == "42"]  # 双轴用 seed42，与 P3 同档
        if not m:
            table.append(f"| {desc} | {tag} | 待跑 | 待跑 |")
        else:
            ms = fmt(m[0]["mse_norm"]); ma = fmt(m[0]["mae_norm"])
            table.append(f"| {desc} | {tag} | {ms} | {ma} |")
    # N 消融
    for nq in (8, 12):
        m = [r for r in rows if r["phase"] == "phase3" and r["id"] == f"p3_ecl_N{nq}_42" and r["status"] == "ok"]
        table.append(f"| n_qubits={nq} | 量子空间 | {fmt(m[0]['mse_norm']) if m else '待跑'} | "
                     f"{fmt(m[0]['mae_norm']) if m else '待跑'} |")
    # M 消融
    for mm in (16, 48):
        m = [r for r in rows if r["phase"] == "phase3" and r["id"] == f"p3_ecl_M{mm}_42" and r["status"] == "ok"]
        table.append(f"| spectrum_M={mm} | 采样密度 | {fmt(m[0]['mse_norm']) if m else '待跑'} | "
                     f"{fmt(m[0]['mae_norm']) if m else '待跑'} |")
    lines += table
    lines.append("")
    return lines


def p4_section(rows):
    lines = ["## P4：E2 主档位（长 L 双阶段 × seed42）", ""]
    table = ["| id | 数据集 | L=H | MSE_norm | MAE_norm |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: x["id"]):
        if r["phase"] != "phase4":
            continue
        status = "⏳ 待跑" if r["status"] != "ok" else "✅"
        table.append(f"| {r['id']} | {r['dataset']} | {r['L']} | {fmt(r['mse_norm'])} | {fmt(r['mae_norm'])} | {status}")
    lines += table
    lines.append("")
    return lines


def p1f_section(rows):
    """P1 全矩阵（6 数据集 × 4 配置 × lookback × 2 seed）双栏：mse_norm / skill。

    skill = 1 − mse_norm / CONST_BASELINE[dataset]，主流技能分口径，
    直接可比地回答"模型是否比预测训练均值更有预测力"。
    """
    lines = ["## P1F：全矩阵技能分（mse_norm / skill，2 seed 均值）", ""]
    # 只取当前 P1F 作业（id 前缀 p1f_，ok）
    m = [r for r in rows if r["id"].startswith("p1f_") and r["status"] == "ok"]
    if not m:
        lines.append("（尚无完成的 P1F 作业）")
        lines.append("")
        return lines
    # 聚合：dataset × L × config → (mse_norm均值, skill均值)
    agg = {}
    for r in m:
        ds, L, cfg = r["dataset"], int(r["L"]), P1F_CFG_LABEL.get(r["config"], r["config"])
        agg.setdefault((ds, L, cfg), []).append(float(r["mse_norm"]))
    for ds in CONST_BASELINE:
        base = CONST_BASELINE[ds]
        cells = [(L, cfg, np.mean(vs)) for (d, L, cfg), vs in agg.items() if d == ds]
        if not cells:
            continue
        lines.append(f"### {ds}　常数基准 mse_norm={base:.4f}")
        hdr = "| L | " + " | ".join(c for c in P1F_CFG_LABEL.values()) + " |"
        sep = "|---|" + "---|" * len(P1F_CFG_LABEL)
        lines.append(hdr)
        lines.append(sep)
        for L in sorted({c[0] for c in cells}):
            row = [str(L)]
            for cfg in P1F_CFG_LABEL.values():
                vals = [v for (l, c, v) in cells if l == L and c == cfg]
                if not vals:
                    row.append("—")
                else:
                    mn = float(np.mean(vals))
                    sk = 1.0 - mn / base
                    row.append(f"{mn:.4f} / {sk:+.3f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines.append("判据：skill>0 = 优于常数（train-均值）基准；全配置同 L 共享同一分母，配置间相对排序与 mse_norm 完全一致。")
    lines.append("")
    return lines


def write_report(rows):
    lines = ["# DualAE-QCC 批量实验结果汇总", "",
             f"> 自动生成于 {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}，"
             f"数据来自 `results/summary.csv`（{len(rows)} 条记录）。", ""]

    p1 = p1_section(rows)
    lines += p1

    lines += p1f_section(rows)

    p2, gains = p2_section(rows)
    lines += p2

    lines += p3_section(rows)
    lines += p4_section(rows)

    # 状态分布
    from collections import Counter
    cnt = Counter(r["status"] for r in rows)
    lines += ["## 运行状态", "",
              f"- ok: {cnt.get('ok', 0)} / timeout: {cnt.get('timeout', 0)} / failed: {cnt.get('failed', 0)}",
              f"- 已生成配置 {len(os.listdir(GEN_DIR))} 个，done 标记 {len(os.listdir(DONE_DIR))} 个",
              f"- 待跑（生成了配置但无 done）: {len(set(f[:-5] for f in os.listdir(GEN_DIR) if f.endswith('.yaml')) - set(f[:-5] for f in os.listdir(DONE_DIR) if f.endswith('.done')))} 个", ""]

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return gains


def make_plot(gains):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[summarize] matplotlib 未安装，跳过 hetero_gain.png")
        return
    if not gains:
        print("[summarize] P2 尚无成对结果，跳过散点图")
        return
    xs = np.array([g[0] for g in gains])
    ys = np.array([g[1] for g in gains])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.scatter(xs, ys, s=80, c="steelblue", zorder=3)
    if len(gains) >= 2:
        k, b = np.polyfit(xs, ys, 1)
        xs_line = np.linspace(xs.min() - 1, xs.max() + 1, 100)
        ax.plot(xs_line, k * xs_line + b, color="tomato", lw=1.2,
                label=f"线性拟合 slope={k:.3f}")
        ax.legend()
    ax.set_xlabel("Heterogeneity: delta_hat std % of dominant period")
    ax.set_ylabel("delta_MSE = MSE(no_align) - MSE(dual)")
    ax.set_title("P2: alignment gain vs heterogeneity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    print(f"[summarize] 散点图已保存 → {PLOT} (n={len(gains)})")


def main():
    rows = load_rows()
    if not rows:
        print("[summarize] results/summary.csv 不存在或为空，先跑 run_experiment_batch.py")
        return
    gains = write_report(rows)
    make_plot(gains)
    print(f"[summarize] 报告已生成 → {REPORT}")


if __name__ == "__main__":
    main()
