#!/usr/bin/env python3
"""用 torch.profiler 定位 QCCMamba 前向/反向后向热点（ECL L=96 一批）。

跑 2 批预热 + 3 批 profiling，输出按自时间排序的 GPU 耗时 Top。
"""
import torch
import torch.profiler as prof

from data.dataloader import build_standard_loaders
from model.qcc_mamba import QCCMamba

device = torch.device("cuda")
loaders = build_standard_loaders(dataset_name="electricity", lookback=96, horizon=96,
                                 batch_size=32, stride=1, num_workers=0,
                                 data_dir="../ts_quantum/datasets")
inner = getattr(loaders['train'].dataset, 'dataset', loaders['train'].dataset)
V = inner.data.shape[1]
model = QCCMamba(num_var=V, lookback=96, horizon=96, d_token=512, n_qubits=10,
                 n_layers=2, entangle_topo="linear", kernel_fn="quantum",
                 use_fmap=True, use_spectrum=True, use_H=True, use_S=True,
                 use_bypass=True).to(device)

batch = next(iter(loaders["train"]))
x, y = batch[0].to(device), batch[1].to(device)
xm = batch[2].to(device)

# 预热（编译/显存分配）
for _ in range(2):
    out = model(x, x_mark=xm, return_norm=True)
    loss = out[0].sum() if isinstance(out, tuple) else out.sum()
    loss.backward()
    model.zero_grad()

with prof.profile(activities=[prof.ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as p:
    for _ in range(3):
        out = model(x, x_mark=xm, return_norm=True)
        loss = out[0].sum() if isinstance(out, tuple) else out.sum()
        loss.backward()
        model.zero_grad()

print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25))
