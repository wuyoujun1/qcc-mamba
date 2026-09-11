# QCCK-M Ch.5 Supplementary Experiments — Deliverables (2026-09-09)

> Auto-run on server from the task sheet `QCCK-M第五章补充实验任务单_服务器执行版.md`.
> Everything below uses the **raw symmetric fidelity kernel f** (NOT the message weights K_n):
> **f[i,j] = |⟨ψ_i|ψ_j⟩|²**, symmetric, diagonal = 1, values in [0,1]. No softmax, no temperature, no K_n.

## Summary of results (already consistent with the finalized Ch.5 PDF)

| Item | Result |
|---|---|
| ETTh1 @96 strongest pair | HUFL–MUFL f = **0.596** (~0.60 in text) |
| ETTh2/ETTm1 @96 (newly extracted) | ETTh2 max 0.242; ETTm1 max 0.507 |
| ETTm2 @96 strongest pair | HULL–MULL f = 0.139 |
| ETTh1 @720 strongest pair | HUFL–MUFL f = **0.834** (~0.83 in text), same pair as @96 |
| Strong pairs f>0.1 share | ETTh1/ETTm2 ≈9.5%, ECL ≈0.8%, Traffic ≈1.0% |
| **Spearman(f_offdiag, data coupling)** | ETTh1 0.071, ETTm2 0.396, ETTh2 −0.049, ETTm1 0.138; merged 4 datasets (84 pairs) **ρ=0.249, p=0.0224** — matches PDF Table 5 exactly |
| Prediction case (ETTh1-96, OT) | window 310: QCCK-M 0.0337 vs S-Mamba 0.0621 (S-Mamba 45.7% higher MSE); selected by rule Δ>0 with QCC error within median±0.5IQR (anti cherry-pick) |

Top-5 pairs (f, off-diagonal): ETTh1 = HUFL-MUFL 0.596, HULL-MULL 0.309, HUFL-HULL 0.055, HULL-MUFL 0.052, MUFL-LULL 0.046; ETTm2 = HULL-MULL 0.139, MULL-LULL 0.121, HUFL-LUFL 0.079, HULL-LUFL 0.079, MULL-OT 0.076.

Provenance: f arrays = per-sample fidelity averaged over the **first 10 test batches** (identical method/scripts as the finalized Ch.5 Fig.2, stored at `paper/figs/f_*.npz` locally; ETTh2/ETTm1 added the same way). Data-driven coupling = per-variable z-normalized full series, lag-aligned (window ≤5% of length) max|cross-correlation|. Diagonals removed in every analysis/plot.

## Files

| Path | Content |
|---|---|
| `figs/K_ETTh1_96.png` | ETTh1@96 heatmap of mean f, diagonal hidden (light-grey band), real variable names, cell values |
| `figs/K_ETTm2_96.png` | ETTm2@96, same style |
| `figs/K_ETTh1_720.png` | ETTh1@720, same variable order as 96 |
| `figs/K_distribution_ECDF.png` | ECDF of off-diagonal f: ETTh1/ETTm2/ECL/Traffic @96 |
| `figs/prediction_case.png` | ETTh1-96 window 310, Ground truth / S-Mamba / QCCK-M (OT) |
| `results/K_ETTh1_96.npy` `results/K_ETTm2_96.npy` `results/K_ETTh1_720.npy` | raw mean f matrices (V×V) |
| `results/K_distribution_values.npy` `.npz` | concatenated / per-dataset off-diagonal f values |
| `results/spearman_results.csv` | per-dataset + merged, with source & diag-removed notes |
| `results/topK_pairs.csv` | top-5 off-diagonal pairs with real variable names |
| `results/case_info.txt` | prediction-case metadata + selection rule |
| `gen_figs_stats.py` `gen_pred_case.py` | reproducible CPU/GPU generators |
| `运行记录_20260909.md` | full Chinese run record (same content, Chinese filename) |

## ⚠️ Note for the paper (found during this run, pending author decision)
The finalized Ch.5 PDF embeds `figs/ch5_k_*.png` made by `/tmp/f_opt1.py` that **show the diagonal** (integer axes, no masking) — inconsistent with their own captions ("diagonal hidden, axis = variable names"). The `K_*.png` in this folder are the caption-consistent versions. Not yet swapped into the PDF.

## Reproduce
- CPU: `OMP_NUM_THREADS=8 /tmp/qcc-env/bin/python gen_figs_stats.py`
- GPU: `OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=<free> /tmp/qcc-env/bin/python gen_pred_case.py`
