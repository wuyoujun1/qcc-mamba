# 状态与待办

> 2026-09-12 重排。**已定的不再重复，只留现行结论与仍待办的事。**

---

## ✅ 已定（不要再改回去）

**1. 理论：两层都纠缠（纠缠层数 n = 2）**
- `|ψ_v⟩ = U_v·U_v·|χ_v⟩`，其中 `U_v = CNOT_{4,5}·CNOT_{3,4}·CNOT_{2,3}·CNOT_{1,2} · (⊗_{q=1}^5 RZ_q(β_q)·RY_q(α_q))`，**两层各含一次 CNOT**。2026-09-12 用户拍板固定。
- 依据：`qcc/feature_map.py:7-9` 文件头的"数学定义"（`U_ent` 出现 D 次）与第三章定稿一致。
- ⚠️ **代码暂不改**：`qcc/feature_map.py:318-332` 的实际实现是首层乘积态（无纠缠）+ 重上传层（1 次 CNOT），即 D=2 时只有 **1 个纠缠层**、且 CNOT 在最后。论文全部实验都是用这份代码跑的 → **方法与实验在这点上有差异，已确认暂不处理**。若日后要改代码，主表/消融/机制验证的数字需全部重跑。

**2. 表 2 的 Exchange 720 / Traffic 336 按"持平"报**
- 按"与 S-Mamba 基本持平"报，定版 `ch5cn_pdf.py` 的数字为准，不改成"略低"。
- 备查：定版 `OFF_SM` 与 `data/第五章_主表消融_MSEMAE_20260904.md` 主表段有 17/30 格不同（这两格 md 填的是 QCC 自己的值）。**以定版为准**，md 若要对齐回 log 重导即可，低优先。

**3. 仓库代码 ≠ 跑出论文结果的代码**
- 跑结果的是工作副本 `/home/youjun/dataops_ws/`；仓库 `main` 只是参考快照，缺 `--qmix_norm raw_k` / `rawk_row` / `--qmix_msg` / `--qmix_ln_hp` / `intervention_mask`。
- 决定：**不同步代码**，只在 `README.md` 写明不一致。换机复现要把工作副本一起带走。

**4. 正文以现行稿为准**（`paper/ch3/第三章修改版1.md` 定稿、`paper/ch4/第四章修改2.md`、`paper/ch5/ch5cn_pdf.py`）；旧的矛盾稿件已全部从仓库删除。

---

## 🔧 仍待办

1. **造图脚本输出路径写死**：`paper/ch5/ch5_winheat.py`、`paper/ch5/ch5_supp_20260909/gen_pred_adv.py` 等仍输出到 `/home/youjun/paper/figs/`（不在仓库内），换机即失败。改成相对脚本的 `paper/ch5/figs/`（可加 `QCC_FIG_DIR` 覆盖）。
2. **`raw_runs/` 未入库**：`paper/ch5/ch5_supp_20260909/raw_runs/`（150 个逐 run json/npz）只在本地，仓库只有 `summary_coupling.csv`。日后要复核斯皮尔曼/屏蔽结果需要它。
