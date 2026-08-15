# 自主实验日志（2026-08-13 晚 ~ 11 小时，用户睡觉期间）

## 任务目标（用户定义）
- 找到比 plain 更好的模型：**4 种配置 × 3 数据集（etth1/weather/electricity），≥60% cell 赢 plain**
- 量子核必须保留（底线）；其它架构/超参可以任意改
- 每个变体 ~1h；11h 够 7-8 个变体
- 代码不能有 bug（每步改动必须冒烟测试后再批量）

## 当前状态（2026-08-13 晚）
- 轮 F 结论：**n2（4 维态空间）是突破点**——浓度定理正解
  - qoff_n2_f（fidelity+offdiag, n=2, 强迫）: etth1:96 = 0.5226 (plain 0.6177, −0.095), etth1:192 = 0.6154 (plain 0.6619, −0.047)
  - γ 在 n2 打开（0.04–0.09）——网络第一次主动接受混合
  - weather（V=21）发散（loss→1.8万, test=nan）→ hp_scale 缩放修复（已实现）
- 轮 G 运行中：qoff_n2_f / qoff_n2_s(scale 0.2) / qoff_n2_g(gate 0.05+scale 0.5) × etth1+weather 全 8 档
- 失败模式：weather n2 发散（非崩溃）；etth1 正常

## 关键机制知识（决策依据）
1. 浓度定理：n8 信号 std 0.04（噪声级）→ n2 std 0.34（全幅度）——核信息量放大 8 倍
2. 判别实验：跨变量结构时段专属，静态读出不迁移（0/23）——赢面只能来自"可迁移小部分"
3. γ 探针：网络在 n8 拒绝混合（γ→0），n2 打开——信号幅度是开关
4. topk 在 n8 弱信号下无意义；n2 强信号下才有选择性——electricity(321变量) 对策

## 决策树（每步跑完按此推进）
- A. qoff_n2_f 在 336/720 也赢 → 主推 qoff_n2_f 全矩阵（12 cell）+ 2-seed
- B. 336/720 输 → 试 qoff_n2_g（门控自适应强度）或 kernel_T 更小（0.05/0.03）
- C. weather 仍发散 → hp_scale 降到 0.1/0.05 或 qmix_layers=1
- D. electricity 全输 → qoff_n2_tk（n2+topk=2 选择性混合）→ 还不行就 topk=3 + 门控
- E. 2-seed 验证：赢面 cell 补 seed 2024（同变体同配置）

## 4 种配置（满足用户要求）
1. qoff_n2_f — fidelity+offdiag, n2, 无门控（当前赢家）
2. qoff_n2_s — 同上 + hp_scale=0.2（稳定版）
3. qoff_n2_g — 同上 + gate init 0.05 + scale 0.5（自适应强度版）
4. qoff_n2_tk — 同上 + topk=2（选择性版，electricity 对策）

## 运行纪律
- 批量运行期间**禁止编辑共享代码**（run_dual_ae.py/qcc_mamba.py/quantum_mix.py——子进程重新 import 会混版本）
- run_qmix.py 的编辑安全（runner 已 import，子进程不 import 它）
- 每次代码改动后必须冒烟测试
- 结果从各轮 runner log 读（summary.csv 会被每轮覆盖）
- 注意"ok"但 mse_norm=nan 的发散 run（runner 会标 ok）——分析时必须过滤 nan
- 2 并发；GPU 4090 24G

## ⚠️ 负载风暴事后分析（重要教训！）
- **风暴机制**：另一用户 wangshuolei 的 VS Code Server node 持续轮询 ps（每秒数个 `ps -ax -o ...` 监视命令）
  平时无害；某次 nvidia-smi 卡住 → 我的诊断命令超时后台化 → ps 叠加在 /proc 上 → D 状态反馈环 → load 500+
- **实锤**：D 状态 {ps:525}；kill_storm.py 杀 493 个后回落；残余 ps 是 VSCode 监视器持续生成的（非我所能杀，正常基线）
- **新运行纪律（防服务器关机）**：
  1. **禁止 ps/nvidia-smi**——用 python 快速扫 /proc（timeout 保护）
  2. Bash 调用加 timeout，卡住即弃，不连续补发诊断
  3. 每 10 分钟 cron 先读 /proc/loadavg：1-min > 300 暂停，回落 < 200 恢复
  4. 批量并发 2
- 已恢复：轮 G2（weather 336/720 剩余 8 runs，2026-08-14 凌晨）
- 之后：electricity 波次（4 配置 × 4 档）→ 2-seed 验证

## 每轮结果记录（追加）

### 轮 G/G2 完整结果（qoff_n2 家族 × etth1+weather 全 8 档，2026-08-14 完成）
- **etth1 全档 qoff_n2_f 大赢**：96 −0.095 / 192 −0.047 / 336 −0.106 / 720 −0.100（1-seed 锁定）
- weather（1-seed）: 96 f −0.0036 ✅ / s ✗ / g ✗；192 s −0.0023 ✅ / f ✗ / g 失败；
  336 s −0.0099 ✅ / g −0.0018 ✅ / f ✗；720 f +0.012 ✗ / s 失败 / g 待重跑
- **形态：变体按 horizon 互补**（f 赢 96、s 赢 192/336）——家族覆盖 weather 3/4
- 单配置视角：qoff_n2_f = etth1 4/4 + weather:96 = **5/8**；距 60%（12 cell 需 ≥8）差 electricity
- 稳定性：weather:96/192 的 n2 发散已由 hp_scale 解决（s 变体稳定）；g-192 仍崩溃
- 风暴联动：weather:96 f 首次失败是负载风暴期间，恢复正常后重跑即赢

### 轮 H（electricity 波次，2026-08-14 ~02:00 运行中）
- **electricity:96/192 全变体全输**（g 最接近 +0.003，f 最差 +0.049/+0.056）——321 变量混合结构性有害
- **判断：electricity = 损失列（~0/4），60% 目标 = 一个配置赢 etth1+weather 全 8 档**
- f 目前 5/8（etth1 4 + weather:96）；fs（scale 0.5 平衡版）etth1 结果待测——是关键候选
- 已加 qoff_n2_f1（qmix_layers=1）变体备用
- **electricity:192 g = 0.2112 < 0.2125（−0.0013，全程首个 electricity 赢面！）**——门控+缩放自适应在 321 变量上有用
- **关键思路（轮 I）**：门控 γ 要"开着头"学——g（init 0.05）在 etth1 开不动（全输），
  f（无门控）在 electricity/weather 关不掉（输）→ **g2（gate_init=0.5）/ g3（gate_init=1.0）满缩放**
  = etth1 全开（像 f）+ electricity/weather 自动关小（像 g）——"一个配置自适应"的正确形态
- 下一步：H 完 → 轮 I（g2/g3/fs × etth1+weather 8 档）→ 赢家 × electricity + 2-seed 验证

### 轮 H 完成（electricity + weather:720 补跑，2026-08-14 ~02:30）
- **weather:720: s = 0.3380（−0.0171 ✅ 最大赢面）fs = 0.3460（−0.0091 ✅）**
- electricity 总结：全变体几乎全输，唯一赢 g-192（−0.0013）；g 每档最接近（96 +0.003, 336 +0.009, 720 待定缺 plain 基准）
- **强度分工规律（关键）**：etth1 全强度 f 赢 / weather 缩放 0.2（s）赢 / electricity 门控微调（g）赢——
  **一个自适应强度配置（g2/g3 开着头学门）是统一解**

### 轮 I 完成（g2/g3 × etth1+weather + electricity:720）
- **g3 etth1 4/4 大赢**（96 −0.058 / 192 −0.057 / 336 −0.058 / 720 −0.065）；g2 也 4/4（−0.003~−0.035）
- **weather 全输**：γ 实测不学——etth1 γ=0.93~0.98（该开就开✓）但 weather γ=0.90~1.24（该关到 0.2 不动✗）
- **慢门诊断实锤** → 实现 gate_lr（γ 参数组单独 100 倍 lr，build_optimizer 扩展）
- **修复重大 bug**：qmix_last 在 if 分支内 → plain 模型 UnboundLocalError → plain_electricity:720 一直崩
  （其它 plain 都是旧 .done 跳过未暴露）；已修 + 双路径冒烟通过
- electricity:720: g2 0.3494 / g3 0.4360（plain 基准待修复后重跑）

### 轮 J 完成（g4 快速门控 + g2/g3 失败重试 + plain:720）
- **g4 失败**：etth1 变差（720 +0.006、336 只 −0.009）——快门在好 cell 上振荡；weather 全崩
- **g2 重试立功：weather 3/4**（96 −0.0007 / 336 −0.0069 / 720 −0.0156 ✅✅✅）+ etth1 4/4 = **7/8**
- **plain:720 是训练发散**（val 334 万 test=nan，非 bug——修复已生效）——electricity:720 对 plain 都极难；
  该 cell 基准不稳，需低负载重试或按 11-cell 口径
- **轮 K 设计（第一性原理）**：hp_scale = 7/V 按变量数归一化（etth1→1.0 / weather→0.33 / el→0.022）
  ——混合注入随 V 稀释的物理补偿，一个配置自动适配，无需学习
etth1 已出（1-seed, plain 对照）：
- etth1:96   qoff_n2_f 0.5226 (−0.095) ✅ | qoff_n2_s 0.5922* | qoff_n2_g 0.5922*（*轮 F 数据）
- etth1:192  qoff_n2_f 0.6154 (−0.047) ✅ | s 0.6573* | g 0.6575
- etth1:336  **qoff_n2_f 0.6076 (−0.106) ✅✅** | s 0.7050 | g 0.7193
- etth1:720  **qoff_n2_f 0.7785 (−0.100) ✅✅** | s 0.8892 ✗ | g 0.9043 ✗
- **结论：etth1 全档赢（qoff_n2_f），全强度强迫混合最佳，缩放削弱**
- weather:96  **三变体全失败**（s 0.7min / f 1.9min / g 4.5min 即崩，空日志=SIGKILL 类）——该 cell 放弃或换对策
- weather:192 **qoff_n2_s = 0.2415 vs 0.2438（−0.0023）✅**（缩放版稳定且赢！）| f 0.2501 ✗(轮F) | g 失败
- weather 336/720 未跑（暂停时剩余 8 runs）

### 关键决策记录
- 2026-08-13: hp_scale 实现（防 V=21 weather 发散）；qoff_n2_tk 变体就绪（electricity 对策）；--seeds CLI 就绪

### 轮 L（2026-08-14 白天，P0-1 时滞入 S + electricity 对策）
- 决策背景：轮 K 后 qoff_n2_v = etth1 4/4 大赢 + weather 3/4（weather:96 发散）= 7/8；
  electricity 全输/未跑——主战场必赢（ins.md 判据 3）是唯一堵点
- 轮 L 波次 A（用户已批准，跑批中）：qoff_n2_vtk（v 缩放 + topk=2 选择性混合）× electricity 4 档
  + qoff_n2_v weather:96 重试（1 并发）。weather:96 v 再次发散（epoch1 loss=25181→NaN→CUDA 断言），
  确认是该 cell 训练动力学问题非随机抖动；g2 已赢此 cell（0.1963 vs 0.1970），v 缺它不影响大局
- 波次 B（P0-1，代码已实现+语法过，待波次 A 后冒烟）：qoff_n2_vd = v + delay_in_s
  ——S = [Ã; φ̃; δ̂]（64→65 维），δ̂ 仍 detach（归一化到 ±0.25，L=96 时 ±0.24）
  改动面：spectrum.py（δ̂ 通道）/feature_map.py（proj_S 65）/quantum_mix.py（s_ln 65）
  /qcc_mamba.py（注入 65 + 两处 QuantumMixLayer 透传）/run_dual_ae.py（config 解析）
  /run_qmix.py（BASE_YAML + qoff_n2_vd 变体）/tests/test_smoke.py（Test 6 delay_in_s）
  全程默认关闭、纯增量——跑批子进程混版本零风险
- 波次 C（待 A/B 出赢家）：2-seed 确认（42/2024）。plain 2024 基线已齐
  （etth1 全档 0.6071/0.6816/0.7024/0.8886；weather 0.2017/0.2399/0.2887/0.3700；el 仅 96=0.2282）
- 服务器纪律：波次 A 用 --jobs 1 跑大变量 cell（前日 el:192/336 的 CUDA 错误/空日志=SIGKILL 疑并发显存压力）
