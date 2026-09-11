# 实验环境搭建（新服务器照此执行，~15 分钟）

> 目标：任意 GPU 机器上把本仓库跑起来（训练/测试/评估）。
> 适用于 **NVIDIA 驱动 CUDA 12.x** 的机器（本组机器为 8×RTX4090 + 驱动 570/CUDA12.8，无 sudo、无 conda，系统 python3.12）。

## 0. 三条铁律
1. **环境与数据都放 `/tmp`**（根盘通常很满）；结果输出也尽量指到 `/tmp` 或大盘。
2. **必须 `export OMP_NUM_THREADS=8`**，否则线程爆炸/卡死。
3. torch 必须用 **2.0.1+cu118**（NGC 自带 torch 常是 cu13 编的，驱动 12.x 下 `torch.cuda.is_available()=False`）。

```bash
export UV_CACHE_DIR=/tmp/uv-cache
export UV_PYTHON_INSTALL_DIR=/tmp/uv-python
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
```

## 1. 代码
```bash
git clone git@github.com:wuyoujun1/qcc-mamba.git && cd qcc-mamba
git checkout main
```

## 2. Python 3.11（torch 2.0.1 不支持 3.12）
```bash
mkdir -p /tmp/python-3.11
curl -sL -o /tmp/python311.tar.gz "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/20251217/cpython-3.11.14%2B20251217-x86_64-unknown-linux-gnu-install_only.tar.gz"
tar xzf /tmp/python311.tar.gz -C /tmp/python-3.11 --strip-components=1 && rm /tmp/python311.tar.gz
```

## 3. venv + torch 2.0.1+cu118（走 SJTU 镜像，~26MB/s）
```bash
uv venv /tmp/qcc-env --python /tmp/python-3.11/bin/python3.11
curl -sL -o "/tmp/torch-2.0.1+cu118-cp311-cp311-linux_x86_64.whl" \
  "https://mirrors.sjtug.sjtu.edu.cn/pytorch-wheels/cu118/torch-2.0.1%2Bcu118-cp311-cp311-linux_x86_64.whl"
uv pip install --python /tmp/qcc-env/bin/python "/tmp/torch-2.0.1+cu118-cp311-cp311-linux_x86_64.whl"
rm /tmp/torch-2.0.1+cu118-cp311-cp311-linux_x86_64.whl
/tmp/qcc-env/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available())"   # True
```
> wheel 文件名保持原样，改名会报 "Must have a platform tag"。

## 4. 常用依赖
```bash
uv pip install --python /tmp/qcc-env/bin/python "numpy==1.26.4" "scikit-learn==1.3.0" "matplotlib==3.7.0" \
  pyyaml pandas einops ninja packaging setproctitle "transformers==4.40.0"
```
> 任何依赖 torch 的包都要 `--no-deps`，否则 uv 会把 torch 升到 cu13 版本。

## 5. mamba-ssm 1.2.0 + causal-conv1d 1.4.0（**不要本地编译**）
```bash
curl -sL -o "/tmp/causal_conv1d-1.4.0+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
  "https://gh-proxy.com/https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0%2Bcu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
curl -sL -o "/tmp/mamba_ssm-1.2.0+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
  "https://gh-proxy.com/https://github.com/state-spaces/mamba/releases/download/v1.2.0/mamba_ssm-1.2.0%2Bcu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
uv pip install --python /tmp/qcc-env/bin/python --no-deps /tmp/causal_conv1d-*.whl /tmp/mamba_ssm-*.whl
rm /tmp/causal_conv1d-*.whl /tmp/mamba_ssm-*.whl
```

## 6. reformer-pytorch 依赖链（全 `--no-deps`）
```bash
for p in reformer-pytorch==1.4.4 local-attention rotary-embedding-torch hyper-connections beartype axial-positional-embedding product-key-memory colt5-attention; do
  uv pip install --python /tmp/qcc-env/bin/python --no-deps "$p"
done
```

## 7. 数据（仓库 `dataset/gz/*.csv.gz` 已含全部数据集，**不要重新下载**）
```bash
mkdir -p /tmp/qcc_data/{ETT-small,weather,electricity,traffic,exchange_rate,beijing}
cd qcc-mamba/dataset/gz
for f in ETTh1 ETTh2 ETTm1 ETTm2; do gunzip -c $f.csv.gz > /tmp/qcc_data/ETT-small/$f.csv; done
gunzip -c weather.csv.gz > /tmp/qcc_data/weather/weather.csv
gunzip -c electricity.csv.gz > /tmp/qcc_data/electricity/electricity.csv
gunzip -c traffic.csv.gz > /tmp/qcc_data/traffic/traffic.csv
gunzip -c exchange_rate.csv.gz > /tmp/qcc_data/exchange_rate/exchange_rate.csv
gunzip -c beijing.csv.gz > /tmp/qcc_data/beijing/beijing.csv
cd qcc-mamba/dataset && for d in ETT-small weather electricity traffic exchange_rate beijing; do ln -sfn /tmp/qcc_data/$d $d; done
```
（Beijing 用 `--data custom --root_path ./dataset/beijing/ --data_path beijing.csv --target pm2.5`；其余盘见 README/HANDOFF。）

## 8. 路径修正 + venv 软链
```bash
cd qcc-mamba
grep -rl "home/wuyoujun" --include="*.sh" --include="*.py" --include="*.yaml" . | grep -v .git | xargs -r sed -i "s#/home/wuyoujun#$HOME#g"
ln -sfn /tmp/qcc-env .venv
```

## 9. 校验
```bash
cd qcc-mamba && export PYTHONPATH=.
.venv/bin/python test_dp_smoke.py            # ALL SMOKE TESTS PASSED
.venv/bin/python -c "import torch;from mamba_ssm import Mamba;m=Mamba(d_model=64,d_state=16,d_conv=4,expand=2).to('cuda');print(m(torch.randn(2,32,64,device='cuda')).shape)"
```

## 10. 跑起来
```bash
export OMP_NUM_THREADS=8
python run.py --is_training 1 --model_id demo_ETTh1_96 --model Q_S_Mamba --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --features M --target OT --freq 15min \
  --seq_len 96 --label_len 48 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 2 --d_model 256 --d_ff 256 --d_state 2 --learning_rate 0.00007 \
  --qmix_layers 2 --n_qubits 5 --qmix_norm softmax --kernel_T 0.1 --offdiag \
  --qmix_gate --qmix_gate_init 0.1 --batch_size 32 --train_epochs 10 --patience 3
```
> `--offdiag` 与 `--qmix_gate` 是**无参数开关**，写成 `--offdiag True` 会被 argparse 当成多余参数直接报错。
> 主表口径的完整开关表见 README「复现」一节（`--qmix_norm` 只支持 `avg/softmax/l1`）。

## 坑位速查
| 症状 | 原因 | 正解 |
|---|---|---|
| `torch.cuda.is_available()=False` / "driver too old" | 用了 cu13 编的 torch | 装 torch 2.0.1+cu118 |
| `uv python install` 卡死 | GitHub release 太慢 | 用 NJU 镜像手动下 |
| mamba-ssm/causal-conv1d 构建卡死或 `csrc/*` missing | sdist 不含 CUDA 源码 | 下预编译 wheel（cu118+cxx11abiFALSE），别编译 |
| `cannot import GreedySearchDecoderOnlyOutput` | transformers 5.x 删了该类 | `transformers==4.40.0` |
| `No space left on device` | 缓存/输出写根盘 | 全部指到 /tmp；`UV_CACHE_DIR=/tmp/uv-cache` |
| `ModuleNotFoundError: pkg_resources` | 新版 setuptools 移除 | 只在源码编译时需要 `setuptools==75.6.0` |
| import reformer_pytorch 连环缺包 | 依赖链深 | 第 6 节全部 `--no-deps` |
| 线程爆炸/卡死 | 未限制线程 | `export OMP_NUM_THREADS=8` |

## 备注
- 环境在 `/tmp`，**重启即丢**；重建即照本文（~15 分钟）。
- 训练好的 **ckpt 不在仓库**（体积大、在旧机 `/tmp/dataops_checkpoints`）→ 复现要重训，或从旧机拷贝。
