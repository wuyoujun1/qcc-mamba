"""探针：加载已训练的 dual checkpoint，实测 H 结构、角度分布、K 结构与残差修正质量。"""
import os, math, sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataloader import build_standard_loaders
from model.qcc_mamba import QCCMamba

def main():
    ds, L, H_, seed = "electricity", 96, 96, 42
    ckpt = f"results/matrix/p1f_96/p1f_ecl_dual_96_{seed}_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaders = build_standard_loaders(dataset_name=ds, lookback=L, horizon=H_, batch_size=32, stride=1, num_workers=0, data_dir="../ts_quantum/datasets")
    inner = getattr(loaders['train'].dataset, 'dataset', None)
    train_data = getattr(inner, 'data', None)
    V = train_data.shape[1] if train_data is not None else 321

    model = QCCMamba(num_var=V, lookback=L, horizon=H_, d_token=512, n_qubits=8, n_layers=2,
                     entangle_topo="linear", use_fmap=True, alpha0=0.1, theta_S_scale0=0.5,
                     use_spectrum=True, spectrum_M=32, spectrum_range="0_2",
                     spectrum_amp_normalize=False, spectrum_time_align=True, spectrum_freq_align=True,
                     use_H=True, use_S=True, use_bypass=True, reupload_source="S")
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model_state_dict"])
    model = model.to(device).eval()
    print(f"loaded {ckpt}, V={V}, backbone params: {sum(p.numel() for p in model.backbone.parameters()):,}")

    # ---------- 数据 ----------
    dl = loaders['val']
    it = iter(dl)
    xb = next(it)
    if isinstance(xb, (list, tuple)):
        xb, yb = xb[0], xb[1]
    xb, yb = xb[:8].to(device), yb[:8].to(device)

    with torch.no_grad():
        y, y_main, K, y_norm, y_main_norm, corr_norm = model(xb, return_norm=True)
        # 取 backbone 中间 H：直接调用 qcc 的输入
        x_norm = model.revin(xb, mode="norm")
        x_in = model._prepare_input(x_norm, None)
        out = model.backbone(x_in)
        H = out.H.float()  # (B, V, d)
        S = model.spectrum(x_norm)  # (B, V, 2M)

    # ---------- 1. H 结构 ----------
    Hc = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    Hcos = torch.einsum("bvd,bwd->bvw", Hc, Hc)
    hcos_off = Hcos[0].fill_diagonal_(0)
    print(f"\n[H] H-cosine offdiag: mean={hcos_off.mean():.4f} std={hcos_off.std():.4f} min={hcos_off.min():.4f} max={hcos_off.max():.4f}")

    # ---------- 2. 角度分布 ----------
    fmap = model.qcc.fmap
    with torch.no_grad():
        h_proj = fmap.proj_H(H)  # (B,V,20)
        s_proj = fmap.proj_S(model.qcc.s_ln(S))
        print(f"[angles] proj_H out: std={h_proj.std():.4f} clamp命中率={((h_proj.abs()>math.pi-1e-6).float().mean()*100):.1f}%")
        print(f"[angles] proj_S out: std={s_proj.std():.4f} clamp命中率={((s_proj.abs()>math.pi-1e-6).float().mean()*100):.1f}%")

    # ---------- 3. K vs H-cosine：核是否反映 H 相似性 ----------
    K0 = K[0]
    Hc0 = Hcos[0]
    off_mask = ~torch.eye(V, dtype=bool)
    k_off, hc_off = K0[off_mask], Hc0[off_mask]
    # 与序列相关性（真实"相似变量"）对照：未来值相关
    yv = yb[0]  # (H, V)
    yc = (yv - yv.mean(dim=0)) / yv.std(dim=0).clamp_min(1e-8)
    ycorr = torch.corrcoef(yc.T)[off_mask]
    print(f"\n[K] K_offdiag mean={k_off.mean():.4f} std={k_off.std():.4f}")
    print(f"[K] corr(K_off, H_cos_off) = {torch.corrcoef(torch.stack([k_off, hc_off]))[0,1]:.4f}")
    print(f"[K] corr(K_off, y_corr_off) = {torch.corrcoef(torch.stack([k_off, ycorr.float()]))[0,1]:.4f}")
    print(f"[K] corr(H_cos_off, y_corr_off) = {torch.corrcoef(torch.stack([hc_off, ycorr.float()]))[0,1]:.4f}")

    # ---------- 4. correction 与残差的关系 ----------
    resid = (yb.float() - y_main).transpose(1, 2)  # (B, V, H)
    corr_norm_t = corr_norm.transpose(1, 2)  # (B, V, H)
    c = corr_norm_t[0].flatten()
    r = resid[0].flatten()
    if c.std() > 0 and r.std() > 0:
        print(f"\n[corr] corr(correction, residual) = {torch.corrcoef(torch.stack([c, r]))[0,1]:.4f}")
        print(f"[corr] correction scale std={c.std():.4f} vs residual std={r.std():.4f}")
        print(f"[corr] α={model.qcc.alpha.item():.4f} γ={model.qcc.gamma.item():.4f}")
        # 归一化空间的残差（loss 监督的目标）
        rn = (yb.float() - y_main).transpose(1, 2) / model.revin.denorm_std.abs().clamp_min(1e-8) if hasattr(model.revin,'denorm_std') else resid
        cn = corr_norm_t[0].flatten()
        rnn = rn[0].flatten()
        if cn.std() > 0 and rnn.std() > 0:
            print(f"[corr] corr(correction, norm_residual) = {torch.corrcoef(torch.stack([cn, rnn]))[0,1]:.4f}")

if __name__ == "__main__":
    main()
