# QCC-Mamba 服务器部署与运行指南

> 本文档指导你在 Linux GPU 服务器上配置环境、下载数据、跑通 E1/E2/E3 实验。

---

## 1. 上传 / 克隆代码

### 方式 A：Git（推荐）
```bash
# 在服务器上（需要先建好远程仓库）
git clone <你的远程仓库URL> qcc_mamba
cd qcc_mamba
```

### 方式 B：SCP 直接上传
```bash
# 本地打包
tar czf qcc_mamba.tar.gz qcc_mamba/ --exclude=qcc_mamba/s_mamba_official --exclude=qcc_mamba/.git --exclude='*.pyc' --exclude='__pycache__'

# 上传到服务器（替换 user@server_ip）
scp qcc_mamba.tar.gz user@server_ip:/path/to/workspace/

# 服务器上解压
cd /path/to/workspace
tar xzf qcc_mamba.tar.gz
cd qcc_mamba
```

---

## 2. 环境配置

### 先确认 CUDA 版本
```bash
nvidia-smi | grep "CUDA Version"
```

### 安装 Miniconda（如果没有）
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 创建虚拟环境并安装依赖
```bash
# 创建环境
conda create -n qcc_mamba python=3.11 -y
conda activate qcc_mamba

# 安装 PyTorch（根据 CUDA 版本选择，以下以 CUDA 12.1 为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 mamba-ssm（需要 Linux + CUDA，耗时 2-5 分钟）
pip install mamba-ssm[causal-conv1d]

# 安装其余依赖
pip install -r requirements.txt
```

**验证安装**：
```bash
python -c "import torch; print('PyTorch', torch.__version__)"
python -c "from mamba_ssm import Mamba; print('mamba-ssm OK')"
python -c "from qcc.feature_map import EntanglingFeatureMap; print('QCC modules OK')"
```

---

## 3. 下载数据集

```bash
# 进入 deploy 目录
cd deploy

# 下载全部标准数据集
python download_datasets.py --dir ../datasets

# 或只下载需要的几个
python download_datasets.py --dir ../datasets --datasets electricity etth1 etth2

# 数据集目录结构：
# qcc_mamba/../ts_quantum/datasets/          ← 默认路径
# 或
# qcc_mamba/datasets/                       ← 如果传 --dir ./datasets
```

**⚠️ 需要手动下载的跨变量强电力数据**（需注册申请）：
- **Pecan Street**：https://dataport.pecanstreet.org/ （免费研究许可）
- **NREL PVDAQ** 光伏出力数据
- **ISO New England / PJM** 区域电网负荷

下载后统一放到 `datasets/` 目录，格式要求：
- 第一列是时间戳（`date` / `datetime` 列名均可）
- 其余列为变量数值
- CSV 格式

---

## 4. 跑单元测试（验证环境）

```bash
cd qcc_mamba
PYTHONPATH=. python tests/test_qcc_basic.py
```

预期输出 6 个 ✅，全部 `passed` ✅。

---

## 5. 跑 E1 决定性实验

### 5.1 快速摸底（1 seed, L=720, 单组约 20-40 分钟）
```bash
# 只用量子核 + RFF 快速看一下趋势
PYTHONPATH=. python run_e1.py \
    --config configs/e1_kernel_decisive.yaml \
    --gpu 0 \
    --methods quantum rff
```

### 5.2 完整 E1（6 组 × 3 seeds）
```bash
# 串行跑（约 6-12 小时）
PYTHONPATH=. python run_e1.py --config configs/e1_kernel_decisive.yaml --gpu 0
```

```bash
# 并行跑（6 个 GPU / 6 个 terminal）——分别开 6 个 terminal 各跑一组
# Terminal 1: 量子核
PYTHONPATH=. python run_e1.py --config configs/e1_kernel_decisive.yaml --gpu 0 --methods quantum
# Terminal 2: RBF
PYTHONPATH=. python run_e1.py --config configs/e1_kernel_decisive.yaml --gpu 0 --methods rbf
# ... 依次类推: periodic, rff, mps, none
```

### 5.3 输出
每运行完一条命令，控制台输出：
```
Method: quantum
  seed 2026: test_mse=0.xxxxxx
  seed 2027: test_mse=0.xxxxxx
  seed 2028: test_mse=0.xxxxxx

E1 决定性实验结果
| Method | MSE mean ± std | vs baseline (ΔMSE) | p-value |
|--------|----------------|--------------------|---------|
| quantum| 0.xxxx ± 0.xxx| -0.00xxx           | 0.00xx  |
| rbf    | 0.xxxx ± 0.xxx| +0.00xxx           | 0.00xx  |
...
```

**结果判据**：
- ✅ **量子核显著最优（MSE 低 >2%, p<0.05）** → 量子优势立住 → 进 Day 2（E2/E3）
- ⚠️ 量子核 ≈ RFF / RBF → 跑 E0a 诊断（是核设计问题还是数据问题）
- ❌ 经典核/ MPS 最优 → 联系我调整

---

## 6. 跑 E2 / E3

### E2：标准 benchmark（不退化验证）
```bash
PYTHONPATH=. python run_benchmark.py --config configs/e2_standard.yaml --gpu 0
```

### E3：超长序列主战场
```bash
PYTHONPATH=. python run_benchmark.py --config configs/e3_longterm.yaml --gpu 0
# 注意：L=8760 时显存需求大，建议 batch_size=8-16
```

---

## 7. 实验决策树速查

```
Day 1 下午 → E1（量子核 vs 经典核 vs MPS）
  ├─ 量子核赢 → Day 2 E2/E3
  ├─ 量子核平 → 跑 E0a 诊断
  └─ 量子核输 → 联系我

Day 2 上午 → E2 标准 benchmark
  ├─ 不退化 → 进 E3
  └─ 退化 → 调 α/β

Day 2 下午~晚 → E3 超长序列
  ├─ 提升 5-15% → 进 Day 3 消融
  └─ 持平 → 回检

Day 3 → E4 消融 + 结论
```

完整决策树见 `../ts_quantum/paperidea/experiment-design.md` §二。

---

## 8. 常见问题

| 问题 | 解决 |
|------|------|
| `ImportError: No module named mamba_ssm` | `pip install mamba-ssm[causal-conv1d]`，需要 Linux+CUDA |
| `CUDA out of memory` | E3 L=8760 时 `batch_size` 降到 8，或用梯度累积 |
| `FileNotFoundError: electricity.csv` | 先跑 `python deploy/download_datasets.py` |
| `test_qcc_basic 卡在 SMambaBackbone` | 本地无 mamba-ssm，这是正常的；服务器上跑 |
| 想知道各组 seed 结果是否显著 | `engine/evaluate.py` 的 `paired_t_test` 自动输出 p 值 |
