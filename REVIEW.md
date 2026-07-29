# QCC-Mamba 项目代码审查与设计讨论

> 本文档记录对 `qcc_mamba/` 项目核心代码的逐项审查、关键设计讨论与待改进项。
> 适用于：项目成员快速了解架构、技术决策、潜在风险与改进方向。

---

## 1. 项目代码来源审查

### 1.1 明确标注的开源组件（有迹可循）

| 模块 | 来源 | 证据 |
|------|------|------|
| `backbone/smamba_backbone.py` 等 3 文件 | [sci-m-wang/S-D-Mamba](https://github.com/sci-m-wang/S-D-Mamba) | 每个文件 docstring 标注；README §8 说明 |
| `data/preprocess.py::RevIN` | Kim et al. ICLR 2022 (RevIN 论文) | docstring 注明 |
| `qcc/classical_kernels.py` 中 RBF/Periodic/RFF | 教科书/通用机器学习 | 标准核函数实现 |
| `data/dataset.py` 滑窗逻辑 | 通用时间序列预处理 | 模式与多数公开仓库一致 |

### 1.2 项目自有/创新组件

- `qcc/feature_map.py::EntanglingFeatureMap` — 量子线路经典模拟（QKCS）
- `qcc/kernel.py::quantum_kernel` — 保真度核
- `qcc/qcc_block.py::QCCBlock` — 量子核旁路融合架构
- `qcc/mps_kernel.py::MPSBypass` — MPS 张量网络基线
- `qcc/message_passing.py` — 核加权 GAT 风格消息传递
- `model/qcc_mamba.py::QCCMamba` — 端到端模型装配

### 1.3 主体逻辑核验

✅ 态矢量演化（旋转→纠缠顺序与 docstring 电路定义一致）
✅ 核矩阵对称/对角线=1（酉演化保范）
✅ 损失函数（主损失 + 残差辅助损失）
✅ 训练循环、评估、RevIN 归一化
✅ 测试覆盖了 feature map L2 范数=1、K 对角线=1、shape 正确性

### 1.4 需注意的细节

- **README §9 引用需自查**：`arXiv:2606.20402` 和 `arXiv:2607.20168`（2026 年 ID）需到 arxiv.org 确认是否真实存在
- **RevIN 训练时统计量**：用 `x` 的 mean/std 归一化 `y_true`，分布漂移显著时可能引入误差
- **MockBackbone 注释与 README 不一致**：注释说"用于 E1"，README 说 E1 用 SMambaBackbone
- **真机验证为占位**：`hardware/ibmq_verify.py` 全部 NotImplementedError

---

## 2. 核心代码阅读顺序

按"自顶向下、由总到分"原则：

| 顺序 | 文件 | 目的 | 预计时间 |
|:---:|------|------|:---:|
| 1 | `model/qcc_mamba.py` | 看总架构 | 15 min |
| 2 | `qcc/qcc_block.py` | 看旁路融合 | 10 min |
| 3 | `qcc/feature_map.py` | 看量子线路模拟 | 30 min |
| 4 | `qcc/kernel.py` | 看保真度核 | 5 min |
| 5 | `qcc/classical_kernels.py` | 看经典对照 | 15 min |
| 6 | `qcc/mps_kernel.py` | 看 MPS 对照 | 10 min |
| 7 | `qcc/message_passing.py` | 看消息传递 | 5 min |
| 8 | `backbone/smamba_backbone.py` | 看主干 | 15 min |

合计 ~2 小时可掌握全部核心逻辑。**最核心的 3 个文件**：`qcc_mamba.py` → `qcc_block.py` → `feature_map.py`。

---

## 3. model/qcc_mamba.py 详解

### 3.1 数据流（模块 docstring 自带）

```
x (B, L, V)
→ RevIN norm
→ [可选] 拼接周期时间特征 → (B, L, V+F)
→ Backbone → H (B, V, d), y_main (B, H, V)
→ QCC / MPS 旁路 → y (B, H, V), K (B, V, V)
→ RevIN denorm
```

### 3.2 关键约定

- `B` = batch，`L` = lookback 输入窗口，`H` = horizon 预测步长，`V` = 变量数，`d` = token 维度
- Backbone 输出 `H (B, V, d)` 是"每个变量一个 token"（S-Mamba DataEmbeddingInverted 思想）
- 输出除预测 `y` 外还返回**核矩阵 K**（E7 频谱分析需要）

### 3.3 `__init__` 装配四件套

1. **RevIN**（L97-L98）：每个变量独立 z-score 归一化
2. **周期时间特征**（L100-L107）：`[hour, dow]` → sin/cos → 4 维，**拼接到 V 维之后**
3. **Backbone**（L109-L120）：默认 S-Mamba，`use_norm=False` 避免双重归一化
4. **旁路**（L122-L143）：`use_qcc=True` → QCCBlock；`use_qcc=False` → MPSBypass

### 3.4 forward 主流程（5 步）

1. RevIN norm（缓存 mean/stdev）
2. 拼接周期特征
3. Backbone → H, y_main_norm
4. QCC/MPS 旁路 → y_norm, K, correction_norm
5. RevIN denorm

`return_norm` 参数：训练时 True（辅助损失需要归一化空间张量），推理时 False。

---

## 4. 损失函数深度解析

### 4.1 完整代码

```python
def compute_loss(self, y, y_main, y_true, y_norm, y_main_norm,
                 y_true_norm, correction_norm):
    loss_main = nn.functional.mse_loss(y, y_true)
    residual_norm = y_true_norm - y_main_norm
    loss_qcc = nn.functional.mse_loss(correction_norm, residual_norm)
    return loss_main + self.beta * loss_qcc
```

### 4.2 张量空间约定

| 符号 | 空间 | 形状 | 含义 |
|------|------|------|------|
| `y` | 反归一化 | (B,H,V) | 最终预测 |
| `y_main` | 反归一化 | (B,H,V) | backbone 主预测 |
| `y_true` | 反归一化 | (B,H,V) | 真实标签 |
| `y_norm` | 归一化 | (B,H,V) | QCC 融合预测 |
| `y_main_norm` | 归一化 | (B,H,V) | 主预测（归一化空间） |
| `correction_norm` | 归一化 | (B,H,V) | 旁路原始修正量（**未乘 α**） |

核心等式：`y_norm = y_main_norm + α · correction_norm`

### 4.3 两个损失的设计意图

**主损失** `MSE(y, y_true)`：
- 人话空间，与论文报告的 MSE/MAE 直接对齐
- 反归一化空间的物理意义明确（电力单位）

**辅助损失** `MSE(correction_norm, y_true_norm - y_main_norm)`：
- 监督"主预测应当补充的残差"
- **梯度解耦**：α 不在辅助损失的梯度里，只在主损失里被学
- 类似 ResNet 的残差学习思想，但跨"主预测→旁路修正"路径

### 4.4 为什么"未乘 α 的 correction"是正确设计

常见误解：因为 `y_norm = y_main_norm + α·correction_norm`，所以 correction 应该监督 `residual/α`。

**实际意图**：
- 让 correction_norm 学"自然单位"的残差（不受 α 干扰）
- α 只能从主损失学习，自由浮动
- 隐式归纳偏置：α 会被推向 1（旁路应充分发挥作用）

**类比**：α 是油门，correction 是"目标加速"。油门多大由主损失决定；目标加速是多少由辅助损失告诉旁路。

### 4.5 β 的取值

- β=0.1（项目默认值）：合理平衡
- 太小：残差监督形同虚设
- 太大：旁路喧宾夺主，训练不稳定

### 4.6 空间转换的细节

`train.py` 中 `y_true_norm` 用 **x 窗口的 mean/std** 计算（`engine/train.py` L37-L43）。隐含假设：x 和 y 在时间上同分布（强平稳性）。长序列预测时此假设会松，论文里需说明。

---

## 5. 关键设计问答

### 5.1 lookback vs horizon

- **lookback = 输入窗口长度**（如 720 = 30 天）
- **horizon = 预测步长**（如 96 = 4 天）

### 5.2 S-Mamba 是否使用周期时间特征

**S-Mamba 本身没有**。周期特征是 QCCMamba 在 backbone 前拼接进输入的：

```
x (B, L, V=321) → +4 维 → (B, L, 325) → Backbone
```

副作用：backbone 内部输出 token 数从 V 变成 V+4，smamba_backbone.py 里 `H = enc_out[:, :V, :]` 切回 V 个变量 token。

**Mamba 编码器通过隐状态把时间信号"挤进" V 个变量 token**，但编码完后 4 个时间 token 被丢弃。

⚠️ 输出侧（预测窗口的时间戳）目前**没有显式周期特征编码**。

### 5.3 RevIN 是不是给 321 个变量单独算

**是的**，但还有一个细节：每个 batch、每个变量独立算（沿 L 维求 mean/std），不是全训练集共享。

这与 BatchNorm / LayerNorm 的区别：
- BatchNorm：沿 batch 求
- LayerNorm：沿 feature 求
- **RevIN：沿时间 L 求**

⚠️ **当前实现的隐患**：
1. batch 间方差大（不同 batch 归一化尺度差异大，相当于输入噪声）
2. 训练-测试 gap（测试集 mean/std 分布可能与训练集不同）

**业界的解决方案**：
- per-instance（RevIN 原论文）：用全时间序列 mean/std
- per-channel：用全训练集统计
- moving average（DAIN）：EMA 累积
- 不归一化：靠模型适应

**本项目选择**：per-window + 仿射学习。保留现状，**不与 S-Mamba 引入差异**。

### 5.4 +4 维改进想法（待办）

在 QCC 旁路里加 `y_mark` 编码：

```python
# 未来在 QCCBlock 里加：
self.y_mark_proj = nn.Linear(y_mark_dim, d_token)

def forward(self, H, y_main, y_mark=None):
    if y_mark is not None:
        y_enc = self.y_mark_proj(y_mark).mean(dim=1, keepdim=True)
        H = H + y_enc  # 加到所有变量 token
```

**实施时机**：E1 跑完、知道 baseline 在哪之后。

---

## 6. 实验设计讨论

### 6.1 E1/E2/E3 概览

| 实验 | config | L | H | 角色 |
|------|--------|---|---|------|
| E1 决定性 | `e1_kernel_decisive.yaml` | 720 | 96 | 必要条件筛选 |
| E2 标准 benchmark | `e2_standard.yaml` | 96/192/336/720 | =L | 验证不退化 |
| E3 超长序列 | `e3_longterm.yaml` | 1440/8760/17520 | 720 | **主战场** |

**数据集**：都是 electricity（ECL，321 变量，3 年数据）。同一 7:1:2 切分。
**代码支持 9 个数据集**（electricity/etth1/etth2/ettm1/ettm2/traffic/weather/exchange/solar），E1/E2/E3 只用了 electricity。

### 6.2 E1 H=96 是否太短（关键讨论）

**判断：是的，H=96 对量子核不利。**

理由：
- H=96 时主导模式是日/周周期
- 跨变量相关性主要是局部的
- RBF/RFF 在这个尺度上已经够用
- 量子核的指数维特征空间优势需要更长 H 才能体现

### 6.3 三个实验的真实定位

| 实验 | 角色 | 期望 | 风险 |
|------|------|------|------|
| E1 | 必要条件 | 量子核 > RFF/RBF/periodic | H=96 可能太短 |
| E2 | 标准 benchmark | 不退化 | 低风险 |
| E3 | **主战场** | 量子核长序列优势 | 高风险高回报 |

E1 是**筛选实验**，不是**决战实验**。

### 6.4 建议的 E1 通过标准

E1 通过标准：**量子核 vs 最佳经典核 MSE 提升 ≥ 5%，p-value < 0.05**。

低于此标准视为"险胜/打平"，E1 不算通过，但**仍可继续 E3**（E3 才是决战）。

### 6.5 加强 E1 说服力的方案

1. **困难子集**：只取周末/异常时段数据
2. **多种 H 扫描**：在 E1 内做 H ∈ {96, 192, 336}，看量子核相对优势是否随 H 扩大
3. **N qubits 消融**：N ∈ {6, 8, 10, 12} 看 Hilbert 空间扩大的影响

### 6.6 E1 失败时的应对

如果 E1 量子核"险胜"或"打平"：
1. 不要慌——可能 H=96 太短
2. 直接看 E3 结果
3. 做消融：换 N、换拓扑（linear/ring/none）
4. E3 是决战，E1 失败不等于项目失败

---

## 7. 待办与改进项

### 7.1 立即处理（跑实验前）

- [ ] **README §9 引用自查**：确认 `arXiv:2606.20402` 和 `arXiv:2607.20168` 是否真实存在
- [ ] **MockBackbone 注释修正**：与 README §8 统一
- [ ] **E1 通过标准写入 README**：MSE 提升 ≥ 5%，p < 0.05

### 7.2 跑 E1 后处理

- [ ] 看 α 训练后收敛到多少（验证 "α → 1" 隐式偏置）
- [ ] 看 p-value 是否显著
- [ ] 看六组方法（quantum/rbf/periodic/rff/mps/none）的相对排序

### 7.3 长期改进（有余力时）

- [ ] **+4 维 y_mark 编码**：在 QCC 旁路加未来时间戳编码
- [ ] **RevIN 改 per-instance**：EMA 累积训练集 mean/std
- [ ] **真机验证**：补 `hardware/ibmq_verify.py` 的 PennyLane 实现
- [ ] **测试覆盖率**：增加端到端 mock 训练 1 step 测试、MPSBypass 对称性测试、辅助损失梯度测试

---

## 8. 关键设计原则总结

1. **接口一致性**：QCCBlock 和 MPSBypass 接口完全一致，E1 只换 `kernel_fn` 就能对比
2. **梯度解耦**：α 不在辅助损失梯度里，旁路和融合系数可独立优化
3. **Residual 学习**：辅助损失监督残差而非最终输出，让旁路学"补什么"
4. **per-window RevIN**：简化实现，与 S-Mamba 保持一致
5. **E1 筛选 + E3 决战**：H=96 太短不适合做决战，E3 才是主战场

---

*最后更新：2026-07-28*
