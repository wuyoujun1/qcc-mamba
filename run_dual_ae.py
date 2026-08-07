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
    model = QCCMamba(
        num_var=loaders['train'].dataset.data.shape[1],
        lookback=lookback,
        horizon=horizon,
        d_token=model_cfg.get('d_token', 512),
        n_qubits=model_cfg.get('n_qubits', 10),
        n_layers=model_cfg.get('n_layers', 2),
        entangle_topo=model_cfg.get('entangle_topo', 'linear'),
        kernel_fn=model_cfg.get('kernel_fn', 'quantum'),
        use_fmap=model_cfg.get('use_fmap', True),
        alpha0=model_cfg.get('alpha0', 0.1),
        theta_S_scale0=model_cfg.get('theta_S_scale0', 0.5),
        beta=model_cfg.get('beta', 0.1),
        use_periodic_feat=model_cfg.get('use_periodic_feat', True),
        revin_affine=model_cfg.get('revin_affine', True),
        use_spectrum=model_cfg.get('use_spectrum', True),
        spectrum_M=model_cfg.get('spectrum_M', 32),
        spectrum_range=model_cfg.get('spectrum_range', '0_2'),
        spectrum_amp_normalize=model_cfg.get('spectrum_amp_normalize', False),
        spectrum_time_align=model_cfg.get('spectrum_time_align', True),
        spectrum_freq_align=model_cfg.get('spectrum_freq_align', True),
        use_H=model_cfg.get('use_H', True),
        use_S=model_cfg.get('use_S', True),
    )
    model = model.to(device)
    
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
    )
    
    # 训练参数
    epochs = train_cfg.get('epochs', 100)
    patience = train_cfg.get('patience', 10)
    use_amp = train_cfg.get('use_amp', False)
    accumulation_steps = train_cfg.get('accumulation_steps', 1)
    
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
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
