# 前提验证实验：多变量时序数据的时移/周期异质统计

> 实验代号：PREMISE-VERIFY（前提验证，生死实验第一关）
> 目的：证明"频域跨变量对齐"方法成立的前提——多变量时序数据中确实存在跨变量的时移异质与周期（尺度）异质。
> 交付：本文件 + 输出结果（PNG 图 + 结构化 md 报告）

---

## 0. 一句话目标

对每个数据集、每个变量，估计两个量：
- **δ̂_v**：该变量的整体时移量（相位谱对频率的线性回归斜率 ÷ 2π）；
- **f̂_peak_v**：该变量的主导周期（幅度谱主峰位置）。

然后统计 V 个变量的 δ̂、f̂_peak 分布。**若 ECL/Traffic 的 δ̂ 分布宽、f̂_peak 分散 → 前提成立（数据里真有时移/周期异质，对齐有东西可消除）；若 ETT 分布集中 → 前提弱（对照组，解释为什么 ETT 增益小）。**

---

## 1. 背景（为什么这个实验是"生死验证"）

方法假设：不同变量的波形存在三类差异——①时移（采样起点/相位错位）、②周期尺度（主导周期不同）、③幅度大小。这些差异会让"模式相同"的变量被误判为不相关，所以我们做频域对齐（时间轴相位去趋势 + 频率轴主频归一化）消除它们。

**若数据里根本没有这些差异（δ̂ 全为 0、f̂_peak 全相同）→ 对齐无事可做 → 整个方法不成立。**
**若差异显著存在 → 对齐有价值，且这个统计图本身就是论文的"动机图"（对标 HAQJSK 的图 1 反例）。**

---

## 2. 数据集

| 数据集 | 变量数 V | 采样间隔 | 预期 | 角色 |
|---|---|---|---|---|
| ECL（Electricity） | 321 | 15 min | δ̂ 宽、f̂_peak 分散 | 主战场（前提成立） |
| Traffic | 862 | 1 h | δ̂ 宽、f̂_peak 分散 | 主战场（前提成立） |
| Weather | 21 | 1 h | 中等 | 参考 |
| ETTh1 / ETTh2 | 7 | 1 h | δ̂ 集中、f̂_peak 单一 | 对照组（前提弱） |
| ETTm1 / ETTm2 | 7 | 15 min | δ̂ 集中、f̂_peak 单一 | 对照组（前提弱） |

**数据格式假设**：`./data/<NAME>.csv`，形状 `(T, V)`（行=时间步，列=变量），每列一个变量的完整序列（无需切分 train/val/test，用全量或前 80% 即可）。若实际格式不同（npy/pt 等），按实际调整 load 函数。

---

## 3. 算法步骤（对每个数据集执行）

```
对每个变量 v（V 个变量独立处理）：
  ① 取 N_windows=10 个不重叠窗口，每个窗口长度 L=720（若序列不足 720×10，用可用的最大窗口数）
  ② 对每个窗口：
     a. rfft → 复谱 X（L/2+1 个复数）
     b. 幅度谱 A = |X|；相位谱 φ = np.unwrap(np.angle(X))   ← 必须解卷绕！
     c. 主频：k_peak = argmax A[1:]（跳过直流 k=0）+ 1；f̂ = k_peak / L
     d. 相位线性回归（幅度加权，只对能量显著区段）：
        权重 w = A²（或 A 超过阈值，如 A > max(A)*0.05 的频率段）
        斜率 slope = Σw(f−f̄)(φ−φ̄) / Σw(f−f̄)²
        δ̂ = −slope / (2π)（符号取绝对值报告亦可，我们关心分布宽度）
  ③ 汇总：δ̂_v = median(各窗口 δ̂)；f̂_peak_v = median(各窗口 f̂)
统计 V 个 (δ̂_v, f̂_peak_v)：
  · δ̂ 分布：mean / std / IQR / min / max / CV（变异系数 = std/|mean|）
  · f̂_peak 分布：直方图（观察簇数）、换算主导周期 T = 1/f̂（采样点单位）
画图 + 生成报告
```

---

## 4. 技术细节与常见坑（必须遵守，否则结果无效）

1. **相位必须解卷绕（unwrap）**：`np.angle` 返回 [-π, π] 的卷绕值，直接回归会得到错误斜率。必须 `np.unwrap(np.angle(X))`；
2. **跳过直流分量 k=0**：DC 是均值，幅度几乎总是最大，不跳过则 argmax 恒为 0，f̂_peak 无意义；
3. **只对能量显著区段回归**：高频噪声区相位是随机的（幅度小），会污染回归。用幅度加权（w=A²）或截断（A > max(A)×0.05 的频段）后再回归；
4. **多窗口取中位数**：单窗口噪声大，取多个不重叠窗口的估计值中位数，减少随机波动；
5. **f̂_peak 的周期换算**：f̂ 是归一化频率（[0, 0.5]），主导周期（采样点）= 1/f̂。报告时同时给"归一化频率"和"周期（采样点数）"；
6. **δ̂ 的物理单位**：δ̂ 是"时移量"（采样点为单位）——乘以采样间隔（15min/1h）可得真实时间。报告时给采样点单位即可；
7. **回归的符号**：时移定理下斜率 ≈ −2πδ（取决于 FFT 符号约定），报告时取绝对值或统一符号，不影响"分布宽度"的结论。

---

## 5. 参考实现（Python 脚本 premise_verify.py，服务器直接运行）

依赖：`numpy`, `matplotlib`, `pandas`（读取 CSV）。无 GPU 需求，几分钟跑完。

```python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("./data")
OUT_DIR = Path("./premise_results")
OUT_DIR.mkdir(exist_ok=True)

DATASETS = {
    "ECL": "ECL.csv", "Traffic": "Traffic.csv", "Weather": "Weather.csv",
    "ETTh1": "ETTh1.csv", "ETTh2": "ETTh2.csv", "ETTm1": "ETTm1.csv", "ETTm2": "ETTm2.csv",
}
L_WINDOW = 720          # 窗口长度
N_WINDOWS = 10          # 每变量窗口数
FREQ_THRESH = 0.05      # 幅度阈值（相对最大幅度）

def load_series(name):
    """返回 (T, V) 的 float 数组。按实际数据格式调整。"""
    df = pd.read_csv(DATA_DIR / DATASETS[name])
    return df.to_numpy(dtype=np.float64)

def window_estimates(x):
    """对单变量序列 x (T,)，返回 (δ̂, f̂_peak)（多窗口取中位数）。"""
    T = len(x)
    n_win = min(N_WINDOWS, T // L_WINDOW)
    if n_win == 0:
        L = T
        n_win = 1
    else:
        L = L_WINDOW
    shifts, peaks = [], []
    for i in range(n_win):
        seg = x[i * L : (i + 1) * L]
        if len(seg) < 64:      # 太短无法 FFT
            continue
        X = np.fft.rfft(seg)
        A = np.abs(X)
        phi = np.unwrap(np.angle(X))
        nf = len(A)
        f = np.arange(nf) / L                       # 归一化频率 [0, 0.5]
        # 主频（跳过直流）
        k_peak = int(np.argmax(A[1:])) + 1
        peaks.append(k_peak / L)
        # 幅度加权线性回归（截断到能量显著区段）
        thr = FREQ_THRESH * A.max()
        mask = (A > thr) & (f > 0)
        if mask.sum() < 3:
            mask = f > 0                            # 兜底：全频段
        fm, phim, Am = f[mask], phi[mask], A[mask]
        w = Am ** 2
        f_bar = np.sum(w * fm) / np.sum(w)
        phi_bar = np.sum(w * phim) / np.sum(w)
        num = np.sum(w * (fm - f_bar) * (phim - phi_bar))
        den = np.sum(w * (fm - f_bar) ** 2)
        slope = num / den if den > 1e-12 else 0.0
        shifts.append(abs(slope) / (2 * np.pi))     # |δ̂|，采样点单位
    if not shifts:
        return np.nan, np.nan
    return np.median(shifts), np.median(peaks)

def run_dataset(name):
    data = load_series(name)
    V = data.shape[1]
    shifts, peaks = [], []
    for v in range(V):
        s, p = window_estimates(data[:, v])
        shifts.append(s); peaks.append(p)
    shifts = np.array(shifts); peaks = np.array(peaks)
    # 有效（非 NaN）统计
    valid = ~np.isnan(shifts)
    s, p = shifts[valid], peaks[valid]
    return dict(
        name=name, V=V, n_valid=int(valid.sum()),
        shift_mean=float(np.mean(s)), shift_std=float(np.std(s)),
        shift_iqr=float(np.percentile(s, 75) - np.percentile(s, 25)),
        shift_min=float(np.min(s)), shift_max=float(np.max(s)),
        shift_cv=float(np.std(s) / max(abs(np.mean(s)), 1e-12)),
        peak_mode=float(np.argmax(np.bincount((p * 4).astype(int))) / 4),  # 粗分箱众数
        peak_std=float(np.std(p)),
        n_clusters=int(_estimate_clusters(p)),     # 见下
        all_shifts=s.tolist(), all_peaks=p.tolist(),
    )

def _estimate_clusters(peaks):
    """简单簇数估计：按 log 间隔分箱数非零的 bin 数。"""
    if len(peaks) < 3:
        return 1
    bins = np.logspace(np.log10(max(peaks.min(), 1e-6)), np.log10(max(peaks.max(), 1e-6)), 6)
    h, _ = np.histogram(peaks, bins=bins)
    return int((h > 0).sum())

def plot_distributions(results):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)
    for j, r in enumerate(results):
        ax = axes[0, j]
        ax.hist(r["all_shifts"], bins=30, color="#4A90D9", alpha=0.8)
        ax.set_title(f"{r['name']}  δ̂ (time shift)\nstd={r['shift_std']:.3f}, IQR={r['shift_iqr']:.3f}")
        ax.set_xlabel("time shift (samples)")
        ax = axes[1, j]
        ax.hist(r["all_peaks"], bins=30, color="#E8833A", alpha=0.8)
        ax.set_title(f"{r['name']}  f̂_peak (dominant freq)\nstd={r['peak_std']:.4f}, clusters={r['n_clusters']}")
        ax.set_xlabel("normalized frequency (1/T in samples)")
    plt.tight_layout()
    path = OUT_DIR / "premise_distributions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def make_report(results):
    lines = []
    lines.append("# 前提验证实验报告（PREMISE-VERIFY）\n")
    lines.append("## 1. 数据统计表\n")
    lines.append("| 数据集 | V | δ̂ mean | δ̂ std | δ̂ IQR | δ̂ CV | f̂_peak std | 周期簇数 | 前提判定 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        verdict = _verdict(r)
        lines.append(
            f"| {r['name']} | {r['n_valid']} | {r['shift_mean']:.3f} | {r['shift_std']:.3f} | "
            f"{r['shift_iqr']:.3f} | {r['shift_cv']:.2f} | {r['peak_std']:.4f} | {r['n_clusters']} | {verdict} |"
        )
    lines.append("\n## 2. 判据说明\n")
    lines.append(
        "- **时移异质**：δ̂ 的 std/IQR 越大，说明变量间时移差异越显著（>0 即有异质）；\n"
        "- **周期异质**：f̂_peak 的 std 越大、周期簇数越多，说明主导周期越分散；\n"
        "- **前提成立** = ECL/Traffic 的 δ̂ std 明显 > 0（相对其 IQR 范围）且 f̂_peak 出现多个簇；\n"
        "- **前提弱** = ETT 系列 δ̂ 集中（std 很小）且 f̂_peak 单一簇 → 对齐在 ETT 上无增益可解释。\n"
    )
    lines.append("\n## 3. 结论（按实际数据填写）\n")
    lines.append("- ECL：前提 [成立/不成立]——时移异质 [显著/微弱]，周期异质 [显著/微弱]；\n"
                 "- Traffic：……\n- ETTx：对照组，预期前提弱。\n")
    lines.append("\n## 4. 图\n\n![distributions](./premise_distributions.png)\n")
    path = OUT_DIR / "premise_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

def _verdict(r):
    strong = r["shift_std"] > 0.02 and r["n_clusters"] >= 2
    return "成立（强异质）" if strong else "弱/待观察"

if __name__ == "__main__":
    results = [run_dataset(name) for name in DATASETS]
    plot_distributions(results)
    rep = make_report(results)
    print(f"Done. Report: {rep}")
```

---

## 6. 输出物定义（服务器返回什么）

运行后输出到 `./premise_results/`：

| 文件 | 内容 |
|---|---|
| `premise_distributions.png` | 每个数据集 2 行子图：上行 δ̂ 直方图、下行 f̂_peak 直方图（一张大图，含全部 7 个数据集） |
| `premise_report.md` | **结构化结果报告**：统计表 + 判据说明 + 结论（按实际数据填写）+ 图引用 |

**返回给我们的就是 `premise_report.md`（+ 图）**——我们在它基础上补结论叙述。

---

## 7. 判据（怎么读结果）

1. **ECL**：期望 δ̂ std 明显 > 0（如 std ≥ 采样间隔的 1~2 倍以上，即 std ≥ 0.01~0.02 归一化单位）、f̂_peak 出现 2 个以上周期簇（如 24h 主导 + 少数 12h/168h）→ **前提成立**；
2. **Traffic**：同上预期；
3. **ETTh1/ETTh2/ETTm1/ETTm2**：期望 δ̂ std 接近 0、f̂_peak 单一簇（7 变量同步采样、同周期）→ **前提弱（对照组）**，正好解释"为什么 ETT 增益小"；
4. 若 ECL/Traffic 也集中 → **前提不成立，方法需重新审视**（这正是提前验证的价值）。

---

## 8. 执行方式（服务器端）

```bash
# 在数据集所在目录执行（数据文件放 ./data/）
python premise_verify.py
# 输出：./premise_results/premise_distributions.png + premise_report.md
```

若数据不是 CSV `(T,V)` 格式，调整 `load_series` 即可（npy：`np.load`；pt：`torch.load`；多文件：拼接）。**其他部分无需改动。**
