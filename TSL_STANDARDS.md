# TSL（Time-Series-Library）社区标准协议

> 本文档梳理 THUML 维护的 Time-Series-Library（TSL，https://github.com/thuml/Time-Series-Library ）约定的标准协议，用于：
> 1. 与 DualAE-QCC 当前实现做差距分析
> 2. 实验跑完后对齐论文表格（避免被审稿人"为什么不和 SOTA 对比"打回）
> 3. 评判"什么算合规的 LSF 实验"
>
> 适用范围：**长序列预测（Long-Term Forecasting, LSF）** 任务
> 适用版本：TSL ≥ 2024-01（含 iTransformer / TimeMixer / PatchTST）

---

## 1. 标准数据集（8 个 LSF 数据集）

| 数据集 | 变量 V | 频率 | 时间跨度 | T（总步数）| 适用 split |
|--------|-------:|------|----------|----------:|-----------|
| **ECL**（Electricity） | 321 | 1h | 2011-2014（≈3 年）| 26304 | 7:1:2 |
| **Weather** | 21 | 10min | 2020（≈1 年）| 52696 | 7:1:2 |
| **Traffic** | 862 | 1h | 2015-2016（≈2 年）| 17544 | 7:1:2 |
| **Exchange** | 8 | 1day | 1990-2016 | 7588 | 7:1:2 |
| **ETTh1 / ETTh2** | 7 | 1h | 2016-2018（2 年）| 17420 | 12mo:4mo:4mo |
| **ETTm1 / ETTm2** | 7 | 15min | 2016-2018（2 年）| 69680 | 12mo:4mo:4mo |
| M4（仅短期预测）| 1~1k | 各种 | 各种 | 100k+ | 滚动窗口 |
| PEMS（交通流） | 883/307 | 5min | 2012 | 17856 | 6:2:2（**不属 LSF 圈**）|

> **M4、PEMS 不在 LSF 协议内**——M4 是短期（PMSA 协议），PEMS 是交通流预测圈（LibCity 协议），不要混。

---

## 2. Split 协议

### 2.1 按比例切（ECL / Weather / Traffic / Exchange）

```
train : val : test = 7 : 1 : 2
```

精确到步数：

| 数据集 | train | val | test | 总和 |
|--------|------:|----:|-----:|-----:|
| ECL | 18412 | 2631 | 5261 | 26304 |
| Weather | 36887 | 5271 | 10540 | 52698 |
| Traffic | 12281 | 1755 | 3508 | 17544 |
| Exchange | 5312 | 759 | 1517 | 7588 |

### 2.2 按月切（ETT 系，**不是按比例**）

```
ETTh: train=12mo, val=4mo, test=4mo  →  8640 : 2880 : 2880
ETTm: train=12mo, val=4mo, test=4mo  →  34560 : 11520 : 11520
```

> ⚠️ **ETT 圈是按月切，不是按比例**。即使你的比例算下来是 6:2:2，TSL 也只认 12mo:4mo:4mo。
>
> ✅ **DualAE-QCC 已修正**：`data/dataset.py` 中 ETT 系列默认使用 `use_months=True`，按 12mo:4mo:4mo 划分。
>
> 实际切分在 TSL 里是写死的边界 index（ETTh: 0..8640 / 8640..11520 / 11520..14400），前 14400 步是按月切出来的，最后 3020 步通常被丢弃或纳入 train。

---

## 3. L（lookback）和 H（horizon）约定

| 参数 | TSL 标准 | 含义 |
|------|----------|------|
| **L** | **96**（固定）| 输入窗口长度 |
| **H** | **{96, 192, 336, 720}** | 预测步长 |
| batch_size | 32 | 训练批大小 |

每个数据集 × 4 个 H = **每模型 32 个数**（ECL 4H + Weather 4H + Traffic 4H + ETT 4×4H + Exch 4H），这就是论文表格里"密密麻麻 32 行数"的来源。

> ⚠️ **TSL 表格里不会有 L=720, H=96 这种"超长输入短输出"配置**。"超长输入"实验是 iTransformer 之后才出现的变种，不属于 TSL 标准动作。

---

## 4. 训练超参

| 超参 | ETT 系 | 其他 LSF |
|------|--------|----------|
| Optimizer | Adam | AdamW |
| LR | 1e-3 | 1e-4 |
| Batch size | 32 | 32 |
| Epochs | 10 | 100 |
| Patience | 3 | 10 |
| LR scheduler | CosineAnnealing (T_max=epochs) | 同 |
| Dropout | 0.1 | 0.1 |
| d_model | 512 | 512 |
| n_heads | 8 | 8 |
| e_layers | 2 | 2 |
| d_layers | 1 | 1 |
| d_ff | 2048 | 2048 |
| seed | 2024 / 0（默认）| 同 |
| n_seeds | 1（多数）/ 3（严谨）| 同 |

> ✅ **DualAE-QCC 已修正**：d_model=512（TSL 合规），lr=1e-4 ✅、epochs=100 ✅、patience=10 ✅。支持 AMP 混合精度训练和梯度累积。

---

## 5. 评估指标

- **必报**：MSE、MAE（反归一化空间）
- **不报**：CRPS / MASE / Quantile Loss
- 每个 (dataset, H) 报一组 (MSE, MAE)
- 通常 mean over 1 seed（**严谨论文会做 3 seeds ± std**，TSL 默认 1 seed）

---

## 6. 标准化

**StandardScaler per-channel**：
- 用 train 集每个变量的 mean / std
- z-score 标准化
- test 集用 train 的 mean / std（不重算）
- 模型在标准化空间训练
- 评估时反标准化回原空间

```python
scaler = StandardScaler()
scaler.fit(train_data)  # (T_train, V) → mean (V,), std (V,)
train_x = (train_data - scaler.mean) / scaler.std
test_x = (test_data - scaler.mean) / scaler.std  # 同一个 mean/std
y_pred_real = y_pred * scaler.std + scaler.mean
```

> ⚠️ **DualAE-QCC 当前状态**：仍使用 per-window RevIN，但已支持 TSL 风格的 per-channel StandardScaler（通过配置切换）。建议在正式实验中使用 TSL 标准。

---

## 7. 模型 forward 接口（TSL 强约束）

```python
def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
    """
    x_enc:        (B, L, V)    输入序列
    x_mark_enc:   (B, L, F_t)  输入段时间特征（hour/dow/月/...）
    x_dec:        (B, H, V)    decoder 占位（全 0 / 最后一段 / token）
    x_mark_dec:   (B, H, F_t)  预测段时间特征
    mask:         None / (B, L) padding mask
    """
    # ...
    return dec_out  # (B, H, V)
```

> 即使不用 decoder（如 DLinear / iTransformer / NLinear），**接口也必须带 `x_dec` / `x_mark_dec`**。TSL 框架对所有模型统一这一个签名，方便 swap-in/swap-out。
>
> ⚠️ **DualAE-QCC 当前状态**：forward 接口为 `forward(x, x_mark=None, return_norm=False)`，**不兼容 TSL**。需要在跑 TSL 框架实验时写 adapter 包装。

---

## 8. DualAE-QCC 与 TSL 的差距分析

| 维度 | TSL 标准 | DualAE-QCC 当前 | 差距 |
|------|----------|-----------------|:----:|
| 数据集 | 8 个 LSF | 支持 12 个（含 ECL/Weather/Traffic/ETT 等） | ✅ |
| Split（ECL） | 7:1:2 | 7:1:2 | ✅ |
| Split（ETT） | 12mo:4mo:4mo | 12mo:4mo:4mo（已修正） | ✅ |
| d_model | 512 | 512（已修正） | ✅ |
| L | 96 | 支持 96/192/336/720 | ✅ |
| H | {96, 192, 336, 720} | 支持 96/192/336/720 | ✅ |
| MSE / MAE | 必报 | 必报 | ✅ |
| 标准化 | Train per-var StandardScaler | Per-window RevIN（待统一） | ⚠️ |
| n_seeds | 1 ~ 3 | 支持多 seed | ✅ |
| Forward 接口 | `(x_enc, x_mark_enc, x_dec, x_mark_dec)` | `(x, x_mark, return_norm)` | ⚠️ 不兼容 |
| 同表 SOTA | 必要 | 待跑实验 | ❌ |
| 完整数据表 | 8 数据集 × 4 H = 32 数 | 待跑实验 | ⚠️ 待完成 |
| 创新点 | - | 时频双阶段对齐 + 量子核 | ✅+ |

---

## 9. TSL 对齐 Checklist

### ✅ 已完成
- [x] **加数据集**：支持 12 个数据集（ECL/Weather/Traffic/Exchange/ETTh1/h2/ETTm1/m2/ChinaAQI/METR_LA/PEMS_BAY/ILI）
- [x] **对齐 ETT split**：ETT 系列默认使用 12mo:4mo:4mo 月份划分
- [x] **d_model 修正**：从 128 修正为 512（TSL 合规）
- [x] **支持标准 L/H 配置**：支持 L=96, H∈{96, 192, 336, 720}

### ⏳ 待完成
- [ ] **改 forward 接口**：加 `x_dec` / `x_mark_dec`（即使不用 decoder），写一层 TSL adapter
- [ ] **统一标准化**：删除 per-window RevIN，改用 TSL 风格 StandardScaler（保留 RevIN 作 ablation 即可）
- [ ] **跑 TSL 里的 SOTA baseline**：iTransformer / PatchTST / DLinear 至少 3 个
- [ ] **统一报数表**：每个 (dataset, L, H) 一行，DualAE-QCC vs SOTA 同表
- [ ] **n_seeds ≥ 3**：关键 dataset（H=720）必须 3 seeds ± std
- [ ] **完整实验**：8 数据集 × 4 H = 32 组实验

---

## 10. 时序预测圈的派系对比

| 圈子 | 数据集 | Split | 工具 | 主流会议 |
|------|--------|-------|------|----------|
| **LSF（长序列）** | ECL / WTH / Traffic / Exch / ETT | 7:1:2 + ETT 月切 | **TSL** | AAAI / ICLR / NeurIPS / ICML |
| 交通流预测 | METR-LA / PEMS-BAY / PEMS03-08 | 6:2:2 | LibCity | KDD / ITSC |
| 概率预测 | M5 / Walmart / Electricity | 滚动窗口 | GluonTS | IJF / NeurIPS |
| 时序分类 | UCR / UEA | 原始 train/test | aeon-toolkit | NeurIPS / DMKD |
| 时间序列基础模型 | 各数据集混合 | 各种 | GIFT-Eval | NeurIPS 2024+ |

> DualAE-QCC 当前属 **LSF 圈**。论文投稿前应确认目标 venue（AAAI/ICLR/NeurIPS），LSF 圈审稿人认 TSL 协议。

---

## 11. TSL 必引论文

| 模型 | 论文 | 年份 | TSL 内地位 |
|------|------|------|-----------|
| Informer | *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting* | AAAI 2021 | LSF 协议奠基 |
| Autoformer | *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting* | NeurIPS 2021 | baseline |
| FEDformer | *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Forecasting* | ICML 2022 | baseline |
| DLinear | *Are Transformers Effective for Time Series Forecasting?* | AAAI 2023 | **简单线性模型反超 transformer** |
| PatchTST | *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* | ICLR 2023 | 强 baseline |
| TimesNet | *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis* | ICLR 2023 | 多任务 SOTA |
| iTransformer | *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting* | ICLR 2024 | **当前 SOTA** |
| TimeMixer | *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting* | ICLR 2024 | 长序列 SOTA |

> DualAE-QCC 论文必须至少跟 **iTransformer / PatchTST / DLinear** 同表对比——这三个是"刷不动"的硬 baseline。

---

## 12. 数据下载

TSL 提供 `scripts/data_loader.py` 一键下数据：

```bash
# 在 TSL repo 里
bash scripts/data_preparation/ECL.sh
bash scripts/data_preparation/Weather.sh
bash scripts/data_preparation/Traffic.sh
bash scripts/data_preparation/Exchange.sh
bash scripts/data_preparation/ETT-small.sh
```

下载后文件结构：

```
Time-Series-Library/
  dataset/
    ECL/
      ECL.csv
    weather/
      weather.csv
    traffic/
      traffic.csv
    exchange_rate/
      exchange_rate.csv
    ETT-small/
      ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
```

---

*最后更新：2026-08-07*
*对应项目：DualAE-QCC（[IDEA_DualAE_QCC.md](file:///d:/download/qutest/qcc_mamba/IDEA_DualAE_QCC.md)）*
