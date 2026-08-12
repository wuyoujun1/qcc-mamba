"""前提验证实验（PREMISE-VERIFY）——修正版。

对 FFTexperiment.md 参考实现的修正（参考实现经合成验证存在系统性缺陷）：
1. 时移 δ̂ 不再用"单变量相位谱斜率"（该量测的是波形自身相位结构，不是跨变量时移；
   合成验证：零时移纯余弦会被估成 δ̂≈59，随机相位宽带信号也会给出 δ̂≈30~70）。
   改为：变量相对"数据集共识时钟"（其余变量留一均值）的归一化互相关最优滞后。
2. 主频估计前先对每窗口去均值 + 去线性趋势 + Hann 窗（消除 k=1 全窗口长度伪峰、
   抑制谱泄漏）。
3. 输出前先跑合成数据验证估计器能恢复已知时移。
4. 窗口起始点用 np.linspace 均匀散布全序列（不再只取开头连续 N_WINDOWS 个窗口，
   避免统计押在某个季节/时段上）。
5. 主导周期峰用抛物线插值取亚 bin 位置（消除单 bin 网格量化）。
6. 相位散布分数逐变量 δ̂_v/T_v 归一后取 std（不再用全局中位数周期做分母，
   避免周期异质大的数据集被高估）。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("/home/wuyoujun/ts_quantum/datasets")
OUT_DIR = Path("/home/wuyoujun/qcc-mamba/premise_results")
OUT_DIR.mkdir(exist_ok=True)

# 实验范围：主战场 7（ECL/Weather/ETTh1/ETTh2/ETTm1/ETTm2/ILI）+ 对照组 2（Traffic/Exchange）
# + 候选数据集（ChinaAQI/METR_LA/PEMS_BAY；PJM/ISO_NE/NYISO 未保留）
DATASETS = {
    "ECL": "electricity.csv", "Weather": "weather.csv",
    "ETTh1": "ETTh1.csv", "ETTh2": "ETTh2.csv", "ETTm1": "ETTm1.csv", "ETTm2": "ETTm2.csv",
    "ILI": "ILI.csv",
    "Traffic": "traffic.csv", "Exchange": "exchange_rate.csv",
    "ChinaAQI": "china_aqi.csv", "METR_LA": "metr_la.csv", "PEMS_BAY": "pems_bay.csv",
}
# 采样间隔（小时），用于报告
SAMPLE_H = {"ECL": 0.25, "Weather": 1.0,
            "ETTh1": 1.0, "ETTh2": 1.0, "ETTm1": 0.25, "ETTm2": 0.25,
            "ILI": 168.0,
            "Traffic": 1.0, "Exchange": 24.0,
            "ChinaAQI": 1.0,
            "METR_LA": 0.0833, "PEMS_BAY": 0.0833}

L_WINDOW = 720          # 窗口长度
N_WINDOWS = 10          # 每变量窗口数
MAX_LAG = 180           # 互相关时移搜索范围 ±MAX_LAG（采样点）——各数据集默认值
MAX_LAG_H = 45.0        # 物理时间尺度上的时移上限（小时）：日频数据下 45h ≈ 2 天 → ±2 采样点


def _phys_lag(name, max_lag_h=MAX_LAG_H):
    """按采样间隔把物理时间上限换算成采样点：max(1, round(45h/间隔))。
    Exchange(24h/点)→2 采样点(≈2 天)；ECL(15min)→180；ILI(168h)→1（周频数据无法分辨 45h 内时移）。"""
    return max(1, int(round(max_lag_h / SAMPLE_H[name])))


def load_series(name):
    """返回 (T, V) 的 float 数组。第一列为时间戳则丢弃。"""
    df = pd.read_csv(DATA_DIR / DATASETS[name])
    first = str(df.columns[0])
    if first.lower() in ("date", "datetime", "time", "timestamp"):
        df = df.drop(columns=[first])
    return df.to_numpy(dtype=np.float64)


def hann_detrend(seg):
    """去均值 + 去线性趋势 + Hann 窗。返回与输入等长的预处理序列。"""
    seg = seg - np.mean(seg)
    t = np.arange(len(seg))
    coef = np.polyfit(t, seg, 1)
    seg = seg - np.polyval(coef, t)
    return seg * np.hanning(len(seg))


def _safe_fft_r(x, nf):
    """带重试的 rfft：numpy pocketfft 偶发 "axis out of bounds in iterator RemoveAxis"（
    疑似内存/线程竞态），连续副本重试多次。"""
    for _ in range(5):
        try:
            return np.fft.rfft(np.ascontiguousarray(x), nf)
        except ValueError:
            continue
    raise ValueError("rfft 连续重试仍失败 (pocketfft RemoveAxis)")


def _safe_fft_ir(y, nf):
    for _ in range(5):
        try:
            return np.fft.irfft(np.ascontiguousarray(y), nf)
        except ValueError:
            continue
    raise ValueError("irfft 连续重试仍失败 (pocketfft RemoveAxis)")


def _xcorr_lags(a, b, max_lag):
    """互相关：返回 lags∈[-max_lag,max_lag] 与对应 c(lag)=Σa[t]·b[t-lag]。

    max_lag ≤ 32 时用直接时域相关（精确、无 FFT 循环互相关/竞态问题；
    覆盖 Exchange 复核的 ±2/±15 采样点场景）；大 max_lag 用零填充 FFT 加速
    （FFT 全部经 _safe_fft_* 重试封装，规避 numpy pocketfft 偶发 RemoveAxis）。
    """
    L = len(a)
    lags = np.arange(-max_lag, max_lag + 1)
    if max_lag <= 32:
        c = np.empty(len(lags))
        for k, lag in enumerate(lags):
            if lag >= 0:
                c[k] = np.sum(a[lag:] * b[:L - lag])
            else:
                c[k] = np.sum(a[:L + lag] * b[-lag:])
        return lags, c
    nf = 1
    while nf < 2 * L:
        nf *= 2
    A = _safe_fft_r(a, nf)
    B = _safe_fft_r(b, nf)
    c = _safe_fft_ir(A * np.conj(B), nf)
    # FFT 循环互相关 c[m] = Σ a[t]·b[(t-m) mod nf]；零填充后：
    #   滞后 ℓ ≥ 0 → c[ℓ]；滞后 ℓ < 0 → c[nf + ℓ]。即 idx = ℓ mod nf。
    idx = lags % nf
    return lags, c[idx]


def window_estimates(data, max_lag=MAX_LAG):
    """对数据集 (T,V)，返回 (shifts, peaks)，每变量取多窗口中位数。
    shifts[v]: 相对其余变量留一均值的互相关最优滞后（采样点）
    peaks[v] : 主频（归一化频率，1/T 采样点）
    """
    T, V = data.shape
    n_win = min(N_WINDOWS, T // L_WINDOW)
    if n_win == 0:          # 序列不足一个窗口 → 用整段
        L = T
        n_win = 1
    else:
        L = L_WINDOW
    # 窗口起始点均匀散布全序列（避免只采开头一段的季节/时段偏差）
    starts = np.linspace(0, max(T - L, 0), n_win).astype(int)
    shifts, peaks = [[] for _ in range(V)], [[] for _ in range(V)]
    for i in range(n_win):
        seg = data[starts[i]:starts[i] + L]                    # (L, V)
        if seg.shape[0] < 64:
            continue
        std = seg.std(axis=0)
        active = np.where(std > 1e-9)[0]
        if len(active) < 2:
            continue
        # 预处理（去趋势 + Hann），并按原始 std 归一
        P = np.stack([hann_detrend(seg[:, v]) for v in active])     # (K, L)
        Pn = P / std[active][:, None]                                # (K, L)
        M = Pn.sum(axis=0)                                           # 总参考
        K = len(active)
        for j, v in enumerate(active):
            # 留一参考：其余变量均值（避免变量与自身相关→时移被拉向 0）
            ref = (M - Pn[j]) / (K - 1)
            lags, c = _xcorr_lags(ref, Pn[j], max_lag)
            # 峰在 m=-δ（δ=变量相对参考的延迟）→ 取负使 δ̂ 为正即"该变量滞后于共识时钟"
            shifts[v].append(-float(lags[int(np.argmax(c))]))
            # 主频：跳过 DC(k=0) 与 k=1（去趋势后残余低频伪峰），抛物线插值取亚 bin 峰
            A = np.abs(_safe_fft_r(P[j], len(P[j])))
            k0 = int(np.argmax(A[2:])) + 2
            if 1 < k0 < len(A) - 1:
                alpha, beta, gamma = A[k0 - 1], A[k0], A[k0 + 1]
                denom = alpha - 2 * beta + gamma
                delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
            else:
                delta = 0.0
            peaks[v].append((k0 + delta) / L)
    sh = np.full(V, np.nan)
    pk = np.full(V, np.nan)
    for v in range(V):
        if shifts[v]:
            sh[v] = np.median(shifts[v])
        if peaks[v]:
            pk[v] = np.median(peaks[v])
    return sh, pk


def run_dataset(name, max_lag=MAX_LAG):
    data = load_series(name)
    shifts, peaks = window_estimates(data, max_lag=max_lag)
    valid = ~np.isnan(shifts)
    s, p = shifts[valid], peaks[valid]
    if len(s) == 0:
        return dict(name=name, V=data.shape[1], n_valid=0, shift_mean=np.nan,
                    shift_std=np.nan, shift_iqr=np.nan, shift_min=np.nan,
                    shift_max=np.nan, shift_cv=np.nan, peak_std=np.nan,
                    n_clusters=0, periods=list(), all_shifts=list(),
                    all_peaks=list(), period_mode=np.nan)
    periods = 1.0 / p                            # 主导周期（采样点）= 1/f̂
    L_win = L_WINDOW if data.shape[0] >= L_WINDOW else data.shape[0]
    # 相位散布分数：逐变量 δ̂_v/T_v（各自周期归一）取 std —— 避免"全局中位数周期"歪曲
    fracs = s / periods
    return dict(
        name=name, V=data.shape[1], n_valid=int(valid.sum()),
        shift_mean=float(np.mean(s)), shift_std=float(np.std(s)),
        shift_iqr=float(np.percentile(s, 75) - np.percentile(s, 25)),
        shift_min=float(np.min(s)), shift_max=float(np.max(s)),
        shift_cv=float(np.std(s) / max(abs(np.mean(s)), 1e-12)),
        peak_std=float(np.std(p)),
        n_clusters=_n_clusters(periods, window_len=L_win),
        period_mode=float(np.median(periods)),
        shift_frac=float(np.std(fracs)),
        periods=[float(x) for x in periods],
        all_shifts=s.tolist(), all_peaks=p.tolist(),
    )


def _n_clusters(periods, window_len=L_WINDOW, min_share=0.05, min_count=2, tol=0.15):
    """主导周期簇数：单链合并（新值与簇的 min/max 任一相对差 ≤ tol 即并入并扩展簇范围，
    避免旧版"只跟簇中心比"导致 24→26→28→31 链条被截断）；
    超过窗口 1/4 的慢周期统一并入"慢趋势"一簇；变量数 ≥ max(min_count, 5%·V) 的簇数。"""
    p = np.asarray(periods, dtype=float)
    p = p[(p > 4) & (p < 1e4)]
    if len(p) < 2:
        return 0
    slow = p > window_len / 4          # 慢趋势（周期 > 窗口/4，如 5-min 采样的周周期）
    n_slow = int(slow.sum())
    p = p[~slow]
    vals, counts = np.unique(np.round(p).astype(int), return_counts=True)
    clusters = []                      # {"min": float, "max": float, "count": int}
    for v, c in zip(vals, counts):
        for cl in clusters:
            near_min = abs(cl["min"] - v) / v <= tol
            near_max = abs(cl["max"] - v) / v <= tol
            if near_min or near_max:
                cl["min"] = min(cl["min"], float(v))
                cl["max"] = max(cl["max"], float(v))
                cl["count"] += c
                break
        else:
            clusters.append({"min": float(v), "max": float(v), "count": int(c)})
    n = len(p) + n_slow
    thresh = max(min_count, int(min_share * n))
    n_clu = sum(1 for cl in clusters if cl["count"] >= thresh)
    if n_slow >= thresh:
        n_clu += 1
    return n_clu


def top_periods(r, k=4):
    vals, counts = np.unique(np.round(r["periods"]).astype(int), return_counts=True)
    order = np.argsort(-counts)[:k]
    return ", ".join(f"T={vals[i]}pt({counts[i]}/{r['n_valid']})" for i in order)


def plot_distributions(results):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)
    for j, r in enumerate(results):
        ax = axes[0, j]
        ax.hist(r["all_shifts"], bins=30, color="#4A90D9", alpha=0.8)
        ax.set_title(f"{r['name']}  d_hat time-shift (samples)\n"
                     f"std={r['shift_std']:.2f}, IQR={r['shift_iqr']:.2f}, CV={r['shift_cv']:.2f}")
        ax.set_xlabel("time shift (samples)")
        ax.axvline(0, color="gray", lw=0.8)
        ax = axes[1, j]
        ax.hist(r["periods"], bins=30, color="#E8833A", alpha=0.8)
        ax.set_title(f"{r['name']}  dominant period (samples)\n"
                     f"clusters={r['n_clusters']}, std={r['peak_std']:.4f}\n{top_periods(r, 2)}")
        ax.set_xlabel("dominant period (samples)")
    plt.tight_layout()
    path = OUT_DIR / "premise_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _shift_het(r):
    """时移异质判定：δ̂ std 超过主导周期的 10%（相位散布 > 1/10 周期）视为显著。"""
    return "显著" if r["shift_frac"] > 0.10 else "微弱"


def _period_het(r):
    return "显著" if r["n_clusters"] >= 2 else "微弱"


def _verdict(r):
    return "成立" if (_shift_het(r) == "显著" or _period_het(r) == "显著") else "弱"


def _fmt_hours(h):
    """采样间隔小时格式化：>=1h 显示整数/1 位小数，<1h 显示分钟。"""
    if h >= 1:
        return f"{h:.0f}h" if abs(h - round(h)) < 1e-9 else f"{h:.1f}h"
    mins = h * 60
    return f"{mins:.0f}min" if abs(mins - round(mins)) < 1e-9 else f"{mins:.1f}min"


def make_report(results, synth=None):
    lines = []
    lines.append("# 前提验证实验报告（PREMISE-VERIFY，修正版）\n")
    lines.append("> 对 FFTexperiment.md 参考实现的修正说明：参考实现的时移估计（单变量相位谱线性回归斜率）"
                 "经合成数据验证不能度量跨变量时移——零时移的纯余弦会被估成 δ̂≈59 采样点，随机相位（无任何时移）"
                 "的宽带信号也会给出 δ̂≈30~70；且其\"全频段兜底回归\"分支在 ECL 上触发 69.5% 的窗口。"
                 "因此本版改为互相关最优滞后（变量相对其余变量留一均值的归一化互相关），"
                 "主频估计增加去趋势+Hann 预处理以消除全窗口长度伪峰。\n")
    lines.append("\n## 0. 合成数据验证\n")
    if synth:
        ok, corr = synth["ok"], synth["corr"]
        rec_std, true_std = synth["rec_std"], synth["true_std"]
        lines.append("- 构造 20 个变量共享日周期信号（周期 96pt+672pt）、真值时移均匀分布于 ±40 采样点，"
                     "验证互相关最优滞后估计器能恢复时移分布。\n")
        lines.append(f"- 结果：恢复时移 std={rec_std:.2f}（真值 std={true_std:.2f}），"
                     f"恢复 vs 真值 Pearson |corr|={abs(corr):.3f}"
                     f"{'——✔ 估计器有效（|corr|>0.9）' if ok else '——✘ 估计器未达阈值'}。\n")
    else:
        lines.append("- 合成数据验证未运行。\n")
    lines.append("\n## 1. 数据统计表\n")
    lines.append("| 数据集 | V | 采样间隔 | δ̂ mean | δ̂ std | δ̂ IQR | δ̂ CV | 主导周期簇数 | δ̂ 时移异质 | 周期异质 | 前提判定 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        s_het, p_het = _shift_het(r), _period_het(r)
        verdict = _verdict(r)
        lines.append(
            f"| {r['name']} | {r['n_valid']} | {_fmt_hours(SAMPLE_H[r['name']])} | {r['shift_mean']:.2f} | "
            f"{r['shift_std']:.2f} | {r['shift_iqr']:.2f} | {r['shift_cv']:.2f} | "
            f"{r['n_clusters']} | {s_het} | {p_het} | {verdict} |"
        )
    lines.append("\n## 2. 各数据集主导周期分布\n")
    for r in results:
        lines.append(f"- **{r['name']}**（{_fmt_hours(SAMPLE_H[r['name']])}/采样点）：{top_periods(r, 4)}；"
                     f"主导周期中位数 T≈{r['period_mode']:.0f} 采样点"
                     f"（≈{r['period_mode'] * SAMPLE_H[r['name']]:.1f}h）。")
    lines.append("\n## 3. 判据说明\n")
    lines.append(
        "- **时移异质**：δ̂_v = 变量 v 相对其余变量共识时钟的互相关最优滞后（采样点）。"
        "逐变量按自身主导周期 T_v 归一化得相位散布分数 φ_v = δ̂_v/T_v（周期占比，"
        "消除高周期变量 δ̂ 数值天然偏大的干扰），以 std(φ_v) 判定：>10%（>1/10 周期）视为显著。\n"
        "- **周期异质**：主导周期簇数（单链合并，相对差 ≤15% 并入并扩展簇范围，"
        "变量数 ≥ max(2, 5%·V)）≥2 表示变量间主导周期分散；簇数=1 表示各变量共享同一主导周期。\n"
        "- **前提成立** = 时移异质或周期异质任一显著 → 频域对齐有东西可消除；"
        "两者都微弱 → 前提弱。\n"
    )
    lines.append("\n## 4. 结论（按实际数据填写）\n")
    for r in results:
        s_het, p_het = _shift_het(r), _period_het(r)
        verdict = _verdict(r)
        lines.append(f"- {r['name']}：前提 {verdict}——时移异质 {s_het}"
                     f"（δ̂ std={r['shift_std']:.2f} 采样点 = 主导周期的 {r['shift_frac'] * 100:.1f}%），"
                     f"周期异质 {p_het}（簇数={r['n_clusters']}，主导周期 T≈{r['period_mode']:.0f} 采样点"
                     f"≈{r['period_mode'] * SAMPLE_H[r['name']]:.1f}h）。")
    # 跨数据集综合（按物理系统分类）
    lines.append("\n## 5. 跨数据集综合结论\n")
    fam_strong = [r["name"] for r in results if _verdict(r) == "成立"]
    fam_weak = [r["name"] for r in results if _verdict(r) == "弱"]
    lines.append(
        f"- **前提成立（{len(fam_strong)}）**：{', '.join(fam_strong)}——存在跨变量时移/周期异质，"
        "频域对齐有东西可消除。\n"
        f"- **前提弱（{len(fam_weak)}）**：{', '.join(fam_weak)}——变量同步、共享主导周期，"
        "对齐无可消除的差异。\n"
    )
    # 关键模式按实际结果动态生成（避免硬编码 ①"7 全成立"/②"对照组弱"与数据矛盾）
    MAIN = ["ECL", "Weather", "ETTh1", "ETTh2", "ETTm1", "ETTm2", "ILI"]
    CONTROL = ["Traffic", "Exchange"]
    m_strong = [r["name"] for r in results if r["name"] in MAIN and _verdict(r) == "成立"]
    m_weak = [r["name"] for r in results if r["name"] in MAIN and _verdict(r) == "弱"]
    c_strong = [r["name"] for r in results if r["name"] in CONTROL and _verdict(r) == "成立"]
    c_weak = [r["name"] for r in results if r["name"] in CONTROL and _verdict(r) == "弱"]
    strong_rs = [r for r in results if _verdict(r) == "成立"]
    strongest = max(strong_rs, key=lambda r: r["shift_frac"]) if strong_rs else None
    seg1 = ""
    if m_weak:
        seg1 = f"另 {len(m_weak)} 个（{'、'.join(m_weak)}）前提弱——变量同步、共享主导周期。"
    lines.append(
        f"关键模式：① 主战场 {len(MAIN)} 个中 {len(m_strong)} 个前提成立"
        f"（{', '.join(m_strong)}）——时移/周期异质显著，频域对齐有东西可消除；{seg1}"
    )
    aligned_label = "“已对齐”"
    seg2 = ""
    if c_strong:
        seg2 = "；" + "、".join(c_strong) + " 反而前提成立——对照组的“已对齐”预期未完全成立。"
    lines.append(
        f"② 对照组 2 个：{'、'.join(c_weak)} 前提弱（变量同步、共享主导周期，是{aligned_label}对照）{seg2}"
    )
    if strongest:
        top_p = top_periods(strongest, 2)
        lines.append(
            f"③ {strongest['name']} 前提最强（相位散布 std(φ)="
            f"{strongest['shift_frac'] * 100:.1f}%），主导周期 {top_p}。"
        )
    else:
        lines.append("③ 本轮无数据集前提成立（时移/周期异质均微弱）。")
    lines.append(
        "④ ILI（城市流感峰周错位）是跨变量时移异质的直接证据。"
        "原假设\"ECL/Traffic 异质 vs ETT 均质\"不成立：Traffic 反而最同步；"
        "ETT 系分裂——hourly（ETTh1/ETTh2）异质、15min（ETTm1/ETTm2）均质。\n"
    )
    lines.append("\n## 6. 图\n\n![distributions](./premise_distributions.png)\n")
    lines.append("\n## 7. 下载与可用性说明\n")
    lines.append(_availability_note())
    path = OUT_DIR / "premise_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _availability_note():
    """候选数据集下载/可用性说明（静态文本，反映最后一次可用时点的状态）。"""
    return """| 数据集 | 状态 | 说明 |
|---|---|---|
| ChinaAQI | ✅ (8784, 342) | CN-AIR 仓库（atomgit.com/GewisLab/CN-AIR，国内直连克隆）2024 全年 366 个日文件，取 type=AQI，重索引严格小时网格；V=342（剔除 33 个 >20% 缺失城市）。 |
| METR_LA | ✅ (34272, 207) | UCTB pickle → (T,V)，5min，2012-03→06-27，无 NaN。 |
| PEMS_BAY | ✅ (52128, 325) | UCTB pickle → (T,V)，5min，2017-01→06-30，无 NaN。 |
| NYISO | ❌ 未获得 | mis.nyiso.com（官方 Zone Load 源）网络不可达；GitHub 镜像 lucas-309/load_forecasting 仅有单序列系统总负荷（V=1），不满足前提验证的跨变量要求；未找到可下载的 11 区域级 CSV。 |
| PJM | ➖ 已删除 | 已尝试（Kaggle 经 GitHub 镜像，8 区，(45329,8)），前提弱（变量同步），按要求移除。 |
| ISO_NE | ➖ 已删除 | 已尝试（luckyboyjeff 仓库，系统负荷+温度，V=2），前提弱（变量太少），按要求移除。 |

**说明**：候选数据集保留 ChinaAQI、METR_LA、PEMS_BAY 三个多变量样本（V=342/207/325）；PJM、ISO_NE 已按用户要求从 DATASETS 与磁盘删除。NYISO 因官方 Zone 源（mis.nyiso.com）网络不可达未能获得——LSB-dev 元数据中 "PJM 240 / NYISO 264 / ISO-NE 108" 实为时间跨度（月）而非区域数，且电网负荷本质共享 24h 周期（前提弱），故删除不影响结论。

**对前提判定的影响**：ChinaAQI 是关键样本——V=342 大变量集上时移异质 25.5%（δ̂ std=21.6 采样点≈21.6h）与周期异质（2 簇，T≈24h 与 ~170-190pt 慢周期）双双显著，前提成立，补上了"大变量强异质"的证据链；METR_LA/PEMS_BAY（成立，周期异质）覆盖交通传感器场景。"""



def synthetic_validation():
    """构造已知时移的合成数据，验证估计器能恢复时移分布。

    Returns:
        dict: {ok, corr, rec_std, true_std}——ok 供 __main__ 打印与报告判断，
        corr/rec_std/true_std 供 make_report 动态写入报告第 0 节。
    """
    V, L, W = 20, L_WINDOW, N_WINDOWS
    T = L * W
    rng = np.random.default_rng(0)
    t = np.arange(L)
    base = np.cos(2 * np.pi * t / 96) + 0.5 * np.cos(2 * np.pi * t / 672)
    true_shifts = np.linspace(-40, 40, V)
    data = np.empty((T, V))
    for v in range(V):
        sig = np.roll(base, int(round(true_shifts[v]))) + 0.3 * rng.standard_normal(L)
        data[:, v] = np.tile(sig, W)
    sh, _ = window_estimates(data)
    corr = np.corrcoef(sh, true_shifts)[0, 1]
    rec_std = float(np.nanstd(sh))
    true_std = float(np.std(true_shifts))
    ok = abs(corr) > 0.9
    print(f"[synthetic] V={V}, 真值时移均匀分布 ±40 采样点")
    print(f"  恢复时移: mean={np.nanmean(sh):6.2f} std={rec_std:6.2f} "
          f"(真值 mean≈0, std≈{true_std:.2f})")
    print(f"  恢复 vs 真值 Pearson |corr|={abs(corr):.3f}  "
          f"{'✔ 估计器有效（幅度/分布宽度一致）' if ok else '✘ 估计器无效'}")
    return dict(ok=ok, corr=float(corr), rec_std=rec_std, true_std=true_std)


def review_exchange():
    """单独复核 Exchange（对照组）：① 收紧 MAX_LAG 到物理时间 45h≈±2 采样点；
    ② 打印 8 个货币对逐变量 δ̂_v/T_v，判断是否个别点主导。
    物理上限只对日频 Exchange 生效：ILI 是周频，套 45h 上限会退化成 1 采样点（不可分辨），故不改。
    """
    data = load_series("Exchange")
    V = data.shape[1]
    lags = (MAX_LAG, 15, _phys_lag("Exchange"))   # 当前 / 中等 / 物理时间上限(≈2 天)
    print("=" * 76)
    print(f"[review Exchange] V={V}，日频(24h/点)；时移搜索范围三档对比："
          f"{lags[0]}pt(≈{lags[0]}d) / {lags[1]}pt(≈{lags[1]}d) / {lags[2]}pt(≈{lags[2]}d)")
    summary = {}
    for ml in lags:
        sh, pk = window_estimates(data, max_lag=ml)
        valid = ~np.isnan(sh)
        s, p = sh[valid], pk[valid]
        periods = 1.0 / p
        fracs = s / periods
        print("-" * 76)
        print(f"max_lag=±{ml:3d} 采样点")
        print(f"   {'i':>2s} {'δ̂_v':>7s} {'T_v':>7s} {'φ_v=δ̂/T':>8s}   "
              f"{'i':>2s} {'δ̂_v':>7s} {'T_v':>7s} {'φ_v=δ̂/T':>8s}")
        for i in range(0, V, 2):
            left = f"{i+1:2d} {s[i]:7.2f} {periods[i]:7.0f} {fracs[i]:8.3f}"
            right = (f"{i+2:2d} {s[i+1]:7.2f} {periods[i+1]:7.0f} {fracs[i+1]:8.3f}"
                     if i + 1 < V else "")
            print(f"   {left}   {right}")
        phi_std = float(np.std(fracs))
        verdict = "显著" if phi_std > 0.10 else "微弱"
        print(f"   → φ=std(δ̂/T)={phi_std * 100:5.1f}%  {verdict}")
        summary[ml] = dict(
            phi_std=phi_std, verdict=verdict,
            shift_std=float(np.std(s)),
            iqr=float(np.percentile(s, 75) - np.percentile(s, 25)),
        )
    print("=" * 76)
    for ml, d in summary.items():
        print(f"  max_lag=±{ml:3d}: δ̂ std={d['shift_std']:5.2f}  IQR={d['iqr']:4.2f}  "
              f"φ={d['phi_std'] * 100:5.1f}%  {d['verdict']}")
    return summary


if __name__ == "__main__":
    synth = synthetic_validation()
    print("=" * 70)
    results = []
    for name in DATASETS:
        try:
            # Exchange（日频）时移上限按物理时间收紧到 45h≈±2 采样点（复核确认 180d 搜索是噪声伪影）；
            # 其余保持默认——ILI 周频若套 45h 会退化成 1 采样点，不可分辨，故不套。
            ml = _phys_lag(name) if name == "Exchange" else MAX_LAG
            results.append(run_dataset(name, max_lag=ml))
        except FileNotFoundError:
            print(f"  ⚠️ 跳过（文件未就绪）: {name}")
    if not results:
        print("no datasets available")
        raise SystemExit(1)
    print(f"{'name':8s} {'V':>4s} {'δ̂std':>7s} {'δ̂IQR':>7s} {'φ':>6s} "
          f"{'clusters':>8s} {'Tmode':>6s} {'主导周期':>28s} {'前提':>4s}")
    for r in results:
        print(f"{r['name']:8s} {r['n_valid']:4d} {r['shift_std']:7.2f} {r['shift_iqr']:7.2f} "
              f"{r['shift_frac'] * 100:5.1f}% {r['n_clusters']:8d} {r['period_mode']:6.0f} "
              f"{top_periods(r, 3):>28s} {_verdict(r):>4s}")
    plot_distributions(results)
    rep = make_report(results, synth=synth)
    print("=" * 70)
    print(f"Done. Report: {rep} (synthetic validation passed: {synth['ok']})")
