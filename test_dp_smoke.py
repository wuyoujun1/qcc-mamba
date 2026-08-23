"""P2-1 双路径模型 GPU 冒烟自测（前向/反向/数值稳定）。
注意：mamba_ssm 的 causal_conv1d 只有 CUDA 内核，无法在 CPU 跑。
"""
import torch
from types import SimpleNamespace
from model.Q_S_Mamba_dp import Model

torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)


def make_configs(**over):
    cfg = dict(
        seq_len=96, pred_len=96, enc_in=7, use_norm=1,
        d_model=64, d_state=16,
        spectrum_M=32, spectrum_range='0_2', spectrum_amp_normalize=False,
        spectrum_time_align=True, spectrum_freq_align=True, delay_in_s=True,
        n_qubits=2, qmix_n_layers=2, entangle_topo='linear', kernel_fn='quantum',
        kernel_T=0.1, topk=0, offdiag=True, theta_S_scale0=0.5, angle_norm='clamp',
        angle_groups=1, kernel_group_agg='product',  # 第五轮：多组角度桥接
        dp_time_layers=2, dp_time_dim=64, dp_time_pool='mean', dp_var_embed=1,
        dp_msg='S', dp_fusion='add', dp_gate_init=0.05, use_dp_feats=True,
    )
    cfg.update(over)
    return SimpleNamespace(**cfg)


def run_case(name, configs):
    print(f"\n=== {name} ===")
    model = Model(configs).to(DEVICE)
    B, L, V = 2, 96, 7
    x_enc = torch.randn(B, L, V, device=DEVICE)
    x_mark_enc = torch.randint(0, 24, (B, L, 4), device=DEVICE).float()
    x_dec = torch.zeros(B, 48, V, device=DEVICE)
    x_mark_dec = x_mark_enc[:, :48]
    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print("output shape:", tuple(out.shape), "expected:", (B, configs.pred_len, V))
    assert out.shape == (B, configs.pred_len, V), "shape mismatch"
    assert torch.isfinite(out).all(), "NaN in output"
    # 反向传播
    loss = out.pow(2).mean()
    loss.backward()
    n_nan = 0
    n_par = 0
    for p in model.parameters():
        if p.grad is not None:
            n_par += 1
            if not torch.isfinite(p.grad).all():
                n_nan += 1
    print(f"backward ok: {n_par} params, {n_nan} NaN grads")
    assert n_nan == 0, "NaN gradient"
    # 门控
    gv = model.gate_value.item()
    print("gate_value:", gv)
    assert 0.0 <= gv <= 2.0, "gate out of range"
    # 两步 loss 下降（用 SGD 走一步）
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    l1 = out.pow(2).mean().item()
    for _ in range(2):
        opt.zero_grad()
        out2 = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        out2.pow(2).mean().backward()
        opt.step()
    l2 = model(x_enc, x_mark_enc, x_dec, x_mark_dec).pow(2).mean().item()
    print(f"loss {l1:.4f} -> {l2:.4f}")
    torch.cuda.empty_cache()
    print("PASS")


if __name__ == "__main__":
    run_case("dp add + S msg", make_configs())
    run_case("dp add + H msg", make_configs(dp_msg='H'))
    run_case("dp add + both msg", make_configs(dp_msg='both'))
    run_case("dp time_only (new plain baseline)", make_configs(dp_fusion='time_only'))
    run_case("dp no feats", make_configs(use_dp_feats=False))
    run_case("dp gate_init=0", make_configs(dp_gate_init=0.0))
    # 第五轮：多组角度桥接
    run_case("dp G2 product", make_configs(angle_groups=2, kernel_group_agg='product'))
    run_case("dp G4 product", make_configs(angle_groups=4, kernel_group_agg='product'))
    run_case("dp G4 mean", make_configs(angle_groups=4, kernel_group_agg='mean'))
    run_case("dp G8 product", make_configs(angle_groups=8, kernel_group_agg='product'))
    print("\nALL SMOKE TESTS PASSED")
