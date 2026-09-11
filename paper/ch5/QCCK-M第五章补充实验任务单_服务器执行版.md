# QCCK-M 第五章补充实验任务单（服务器执行版）

## 一、任务目标

当前论文理论已统一：

QCCK 输出原始耦合矩阵 K：

K[i,j] = |<ψ_i|ψ_j>|²

QCCK Layer 使用去除自耦合后的 K 直接进行跨变量消息传播：

H_q = H·W_q

H_p = K_off·H_q

H′ = LN(H + γ·H_p)

当前阶段目标：

不是重新覆盖主表和消融结果，而是补充第五章需要的：

- 耦合矩阵可视化；
- 可解释性分析；
- 定性预测案例。

---

# 二、需要执行的实验

## 实验 1：耦合矩阵 K 可视化（最高优先级）

### 1.1 ETTh1-96

数据集：
ETTh1

预测长度：
96


处理：

- 从测试集 checkpoint 提取原始 K；
- 对测试样本求平均；
- 绘图时去除对角显示；
- 不进行 softmax；
- 不进行 temperature；
- 不生成 K_n。


输出：

```
figs/K_ETTh1_96.png
results/K_ETTh1_96.npy
```


验收：

- 变量名称正确；
- 非对角耦合结构清晰；
- 低耦合区域不能全部显示为黑色；
- 强耦合变量对可观察。

---

### 1.2 ETTm2-96

数据集：

ETTm2

预测长度：

96


输出：

```
figs/K_ETTm2_96.png
results/K_ETTm2_96.npy
```


要求：

绘图风格与 ETTh1 保持一致。

---

### 1.3 四数据集 K 分布（图2c）

数据集：

- ETTh1
- ETTm2
- ECL
- Traffic


预测长度：

统一：

96


方法：

提取测试集平均 K。

去除：

K[i,i]


对所有非对角元素绘制 ECDF。


输出：

```
figs/K_distribution_ECDF.png
results/K_distribution_values.npy
```


验收：

图注需要说明：

ECDF 横轴为耦合强度 K，纵轴为不超过该值的变量对比例。


---

### 1.4 ETTh1-720（图2d）

数据集：

ETTh1

预测长度：

720


输出：

```
figs/K_ETTh1_720.png
results/K_ETTh1_720.npy
```


要求：

- 与 ETTh1-96 相同变量顺序；
- 相同绘图逻辑；
- 增强低值区域颜色区分；
- 不出现大片无法区分区域。


---

# 三、Spearman 可解释性分析

## 目标

验证：

QCCK 学习到的耦合排序是否与数据驱动变量关系一致。


数据集：

- ETTh1
- ETTh2
- ETTm1
- ETTm2


计算：

Spearman(K_off, statistical_relation)


输出：

```
results/spearman_results.csv
```


格式：

```
Dataset,Spearman_rho
ETTh1,xxx
ETTh2,xxx
...
```


验收：

需要同时保存：

- K 来源；
- 统计关系矩阵来源；
- 是否去除 diagonal。


---

# 四、Top-K 强耦合变量对

## 目标

展示：

K 中任意两个变量的耦合强度可以直接读取。


数据集：

- ETTh1
- ETTm2


方法：

- 测试集平均 K；
- 去除对角；
- 排序；
- 输出 Top-5。


输出：

```
results/topK_pairs.csv
```


格式：

```
Dataset
Variable_i
Variable_j
K_value
Rank
```


验收：

必须输出真实变量名称，不只输出 index。

---

# 五、预测可视化案例

## 目标

替换旧预测曲线。

寻找：

QCCK-M 相比 S-Mamba 明显改善的局部预测窗口。


优先数据集：

1. ETTh1
2. ETTm2
3. Beijing-AQI


优先预测长度：

96

如差异不明显，再尝试：

720


方法：

对测试窗口计算：

Δ =
MSE(S-Mamba)
-
MSE(QCCK-M)


选择：

Δ > 0 且具有代表性的窗口。


输出：

```
figs/prediction_case.png

results/case_info.txt
```


case_info 包含：

- Dataset
- Horizon
- Window index
- S-Mamba MSE
- QCCK-M MSE


验收：

图中包含：

- Ground Truth
- S-Mamba
- QCCK-M


说明：

该图用于定性展示，不替代主表结果。

---

# 六、不需要执行

以下实验当前不需要服务器补跑：

## 1. 主表

不重新跑。

原因：

保持当前实验结果。

---

## 2. 消融

不重新跑。

原因：

当前消融没有 gate 问题。


---

## 3. Temperature 实验

删除。

原因：

论文理论已经不包含 T。


---

## 4. Gate 实验

删除。

原因：

论文不存在 source gate / target gate。


---

## 5. K_n 相关实验

删除。

原因：

论文统一使用 K。


---

# 七、最终提交文件

服务器返回：

## 图片

```
figs/
├── K_ETTh1_96.png
├── K_ETTm2_96.png
├── K_distribution_ECDF.png
├── K_ETTh1_720.png
└── prediction_case.png
```


## 数据

```
results/
├── K_ETTh1_96.npy
├── K_ETTm2_96.npy
├── K_ETTh1_720.npy
├── spearman_results.csv
├── topK_pairs.csv
└── case_info.txt
```


# 八、最终验收标准

1. 所有可解释性结果基于原始 K，不使用 K_n。

2. 图2系列能够展示变量耦合结构，而不是只展示颜色。

3. Spearman 有明确统计意义。

4. Top-K 能对应真实变量关系。

5. 预测案例能够直观看到 QCCK-M 与 S-Mamba 的差异。

6. 不影响已有主表和消融实验。
