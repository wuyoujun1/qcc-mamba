#!/usr/bin/env python3
"""DualAE-QCC 统一运行入口。

用法：
    python run_dual_ae.py --config configs/dual_phase1.yaml
    python run_dual_ae.py --config configs/dual_ablation_h_only.yaml
"""
import argparse
import os
import sys
import yaml
import torch
import numpy as np
import random

from data.dataloader import build_standard_loaders
from model.qcc_mamba import QCCMamba
from engine.train import fit, build_optimizer, evaluate, compute_global_stats, set_global_stats


def set_seed(seed):
    """设置随机种子。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """加载 YAML 配置。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


@torch.no_grad()
def compute_k_stats(model, loader, device, n_batches=4):
    """在测试集上采样若干 batch，计算核矩阵 K 的统计量（可解释性素材）。

    返回 {'diag_mean', 'offdiag_mean', 'offdiag_std'}；无旁路时返回 None。
    """
    model.eval()
    diags, offdiags = [], []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        x, y_true = batch[0], batch[1]
        x_mark = batch[2].to(device) if len(batch) == 4 else None
        x = x.to(device)
        _, _, K = model(x, x_mark=x_mark, return_norm=False)
        if K is None:
            return None
        K = K.float()
        V = K.shape[-1]
        diag = torch.diagonal(K, dim1=-2, dim2=-1)  # (B, V)
        off_sum = K.sum(dim=(-1, -2)) - diag.sum(dim=-1)
        diags.append(diag.mean().item())
        offdiags.append((off_sum / (V * (V - 1))).mean().item())
    if not diags:
        return None
    return {
        "diag_mean": float(np.mean(diags)),
        "offdiag_mean": float(np.mean(offdiags)),
        "offdiag_std": float(np.std(offdiags)),
    }


def main():
    parser = argparse.ArgumentParser(description='DualAE-QCC 运行入口')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    
    # 设置种子
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据集
    dataset_name = config['dataset']
    data_dir = config.get('data_dir', '../ts_quantum/datasets')
    lookback = config.get('lookback', 96)
    horizon = config.get('horizon', 96)
    
    train_cfg = config['train']
    batch_size = train_cfg.get('batch_size', 32)
    stride = train_cfg.get('stride', 1)
    num_workers = train_cfg.get('num_workers', 4)
    
    print(f"Loading dataset: {dataset_name}")
    loaders = build_standard_loaders(
        dataset_name=dataset_name,
        data_dir=data_dir,
        lookback=lookback,
        horizon=horizon,
        batch_size=batch_size,
        stride=stride,
        num_workers=num_workers,
    )

    # 预热：在 CUDA 初始化前启动 DataLoader worker（persistent_workers 下只 fork 一次）。
    # 若等 fit() 里第一次迭代才 fork，此时 CUDA 已初始化 + pin_memory 后台线程存活，
    # 曾有 epoch 边界死锁（main=futex_wait, workers=do_poll, GPU 0%）。提前 fork 彻底规避。
    if num_workers > 0:
        print("Warming up DataLoader workers before CUDA init...")
        for _name, _dl in loaders.items():
            try:
                next(iter(_dl))
            except StopIteration:
                pass

    # P2-2：预计算全局每变量 mean/std（供 evaluate 输出归一化 MSE/MAE）
    # 访问路径：DataLoader.dataset = ECLDataset(.dataset=TimeSeriesDataset(.data))
    inner = getattr(loaders['train'].dataset, 'dataset', None)
    train_data = getattr(inner, 'data', None) if inner is not None else None
    if train_data is not None:
        arr = train_data.detach().cpu().numpy()  # (T, V)
        gmean, gstd = compute_global_stats(arr.reshape(-1, arr.shape[-1]))
        set_global_stats(gmean, gstd)
        print(f"Global stats computed: {arr.shape[1]} variables")
    else:
        print("Warning: 无法从 train dataset 获取数据，跳过全局标准化（mse_norm 不可用）")
    
    # 模型
    model_cfg = config['model']
    print("Building model...")
    # 数据形状：ECLDataset 包装 TimeSeriesDataset，实际数据在 .dataset.data
    _train_inner = getattr(loaders['train'].dataset, 'dataset', loaders['train'].dataset)
    _train_data = getattr(_train_inner, 'data', None)
    if _train_data is None:
        raise RuntimeError("无法获取训练数据形状，请检查 dataloader 结构")
    model = QCCMamba(
        num_var=_train_data.shape[1],
        lookback=lookback,
        horizon=horizon,
        d_token=model_cfg.get('d_token', 512),
        qmix_layers=model_cfg.get('qmix_layers', 0),
        qmix_norm=model_cfg.get('qmix_norm', 'avg'),
        head_agg=model_cfg.get('head_agg', False),
        spectrum_inject=model_cfg.get('spectrum_inject', False),
        kernel_T=model_cfg.get('kernel_T', 1.0),
        topk=model_cfg.get('topk', 0),
        offdiag=model_cfg.get('offdiag', False),
        gate=model_cfg.get('gate', False),
        gate_init=model_cfg.get('gate_init', 0.0),
        hp_scale=model_cfg.get('hp_scale', 1.0),
        aux_loss=model_cfg.get('aux_loss', False),
        aux_beta=model_cfg.get('aux_beta', 0.1),
        kernel_sup=model_cfg.get('kernel_sup', 0.0),
        n_qubits=model_cfg.get('n_qubits', 8),
        n_layers=model_cfg.get('n_layers', 2),
        entangle_topo=model_cfg.get('entangle_topo', 'linear'),
        kernel_fn=model_cfg.get('kernel_fn', 'quantum'),
        use_fmap=model_cfg.get('use_fmap', True),
        theta_S_scale0=model_cfg.get('theta_S_scale0', 0.5),
        use_periodic_feat=model_cfg.get('use_periodic_feat', True),
        revin_affine=model_cfg.get('revin_affine', True),
        use_spectrum=model_cfg.get('use_spectrum', True),
        spectrum_M=model_cfg.get('spectrum_M', 32),
        spectrum_range=model_cfg.get('spectrum_range', '0_2'),
        spectrum_amp_normalize=model_cfg.get('spectrum_amp_normalize', False),
        spectrum_time_align=model_cfg.get('spectrum_time_align', True),
        spectrum_freq_align=model_cfg.get('spectrum_freq_align', True),
        delay_in_s=model_cfg.get('delay_in_s', False),
        use_H=model_cfg.get('use_H', True),
        use_S=model_cfg.get('use_S', True),
        reupload_source=model_cfg.get('reupload_source', 'S'),
        angle_norm=model_cfg.get('angle_norm', 'clamp'),
        angle_radius=model_cfg.get('angle_radius', 1.0),
        # P2-1 双路径（2026-08-15）：时间 SSM 单向 + 量子核独占跨变量
        dual_path=model_cfg.get('dual_path', False),
        dp_time_layers=model_cfg.get('dp_time_layers', 2),
        dp_time_dim=model_cfg.get('dp_time_dim', 256),
        dp_time_pool=model_cfg.get('dp_time_pool', 'mean'),
        dp_var_embed=model_cfg.get('dp_var_embed', True),
        dp_msg=model_cfg.get('dp_msg', 'S'),
        dp_fusion=model_cfg.get('dp_fusion', 'add'),
        # QK-Path（2026-08-15）：量子核独立预测通道
        qk_path=model_cfg.get('qk_path', False),
        qk_gate_init=model_cfg.get('qk_gate_init', 0.05),
        qk_use_H=model_cfg.get('qk_use_H', False),
        qk_norm=model_cfg.get('qk_norm', 'softmax'),
    )
    model = model.to(device)

    # warm-start（qkern，2026-08-13）：从 plain checkpoint 初始化（strict=False，
    # 缺失的量子参数保持随机/门控 init），主干无需从头学，混合分支只学增量
    init_ckpt = model_cfg.get('init_ckpt', None)
    if init_ckpt and os.path.exists(init_ckpt):
        ck = torch.load(init_ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
        print(f"[warm-start] 从 {os.path.basename(init_ckpt)} 初始化: "
              f"missing={len(missing)} unexpected={len(unexpected)}")

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 优化器
    lr = train_cfg.get('lr', 1e-4)
    weight_decay = train_cfg.get('weight_decay', 1e-5)
    proj_weight_decay = train_cfg.get('proj_weight_decay', None)
    
    optimizer = build_optimizer(
        model,
        lr=lr,
        weight_decay=weight_decay,
        proj_weight_decay=proj_weight_decay,
        gate_lr=train_cfg.get('gate_lr', None),
    )
    
    # 训练参数
    epochs = train_cfg.get('epochs', 100)
    patience = train_cfg.get('patience', 10)
    use_amp = train_cfg.get('use_amp', False)
    accumulation_steps = train_cfg.get('accumulation_steps', 1)
    eval_test_every_epoch = train_cfg.get('eval_test_every_epoch', True)
    
    # 保存目录
    save_dir = config.get('save_dir', 'results/dual_default')
    run_name = config.get('run_name', 'dual_run')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练
    print(f"\nStarting training for {epochs} epochs...")
    history = fit(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        epochs=epochs,
        patience=patience,
        device=device,
        save_dir=save_dir,
        run_name=run_name,
        use_amp=use_amp,
        accumulation_steps=accumulation_steps,
        eval_test_every_epoch=eval_test_every_epoch,
    )
    
    # 最终评估
    print("\nFinal evaluation on test set...")
    test_metrics = evaluate(model, loaders['test'], device)
    print(f"Test MSE: {test_metrics['mse']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    if 'mse_norm' in test_metrics:
        print(f"Test MSE (normalized): {test_metrics['mse_norm']:.6f}")
    if 'mae_norm' in test_metrics:
        print(f"Test MAE (normalized): {test_metrics['mae_norm']:.6f}")
    
    # 保存训练历史
    history_path = os.path.join(save_dir, f"{run_name}_history.npy")
    np.save(history_path, history)
    print(f"\nTraining history saved to: {history_path}")

    # 可解释性：K 矩阵统计（对角 ≈ 1，非对角 = 跨变量保真度均值）
    _backbone = getattr(model, 'backbone', None)
    _has_k = (getattr(_backbone, 'quantum_mix_layers', None)
              or getattr(_backbone, 'var_path', None))  # P2-1 双路径变量路径也有 K
    if _backbone is not None and _has_k:
        try:
            kstats = compute_k_stats(model, loaders['test'], device)
            if kstats is not None:
                print(f"K stats: diag_mean={kstats['diag_mean']:.4f} "
                      f"offdiag_mean={kstats['offdiag_mean']:.4f} "
                      f"offdiag_std={kstats['offdiag_std']:.4f}")
        except Exception as e:
            print(f"K stats skipped: {e}")

    print("\nTraining completed!")


if __name__ == '__main__':
    main()
