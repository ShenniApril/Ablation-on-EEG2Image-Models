# The accuracy of inter and intra with seed9999

Things-EEG **original vs band-zero vs power-matched Gaussian noise** Top-1 / Top-5 accuracy for **10 subjects**, using **seed=9999** checkpoints from [junmoxiao-cloud/2026BMI-](https://github.com/junmoxiao-cloud/2026BMI-).

## Protocol

| Setting | Encoder | Channels | Checkpoint source |
|---------|---------|----------|-------------------|
| **Intra** | EEGProject | 17 (O+P) | `NeuroBridge/9999-intra-subjects/.../checkpoint_test_best.pth` |
| **Inter** | TSConv | all 63 | `NeuroBridge/9999-inter-subjects/.../checkpoint_test_best.pth` |

- Seed: **9999** (checkpoint training seed + Gaussian-noise RNG; zero-band EEG is seed-independent)
- Bands (Hz): delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30, low_gamma 30–45, gamma 45–70, high_gamma 70–100
- Zero: Fourier-mask band erasure (`code/zero_bands_subjects`)
- Noise: power-matched complex Gaussian from original amplitude & phase (`code/gaussian-noise`, `--seed 9999`)
- Eval: `code/acc_ori_zero_noise` (CPU, `image_test_aug=True`, RN50 fused aug features)

Sister repo (seed1234): [The-accuracy-of-inter-and-intra-with-seed1234](https://github.com/laura-gendon/The-accuracy-of-inter-and-intra-with-seed1234)

## Results

### Intra seed9999 — Original Top-1 / Top-5 (%)

| sub | Top-1 | Top-5 |
|-----|-------|-------|
| 01 | 59.5 | 84.5 |
| 02 | 65.0 | 92.0 |
| 03 | 62.0 | 93.0 |
| 04 | 62.5 | 88.0 |
| 05 | 56.5 | 87.5 |
| 06 | 70.0 | 95.0 |
| 07 | 62.0 | 90.5 |
| 08 | 72.5 | 95.0 |
| 09 | 62.5 | 93.0 |
| 10 | 73.5 | 97.0 |
| **mean** | **64.6** | **91.6** |

Group-mean Top-1 drop when zeroed: theta **−35.2**, beta −26.1, delta −24.5, alpha −22.8; gamma bands ≈ 0.

Files: `intra_seed9999/acc_results/`

### Inter seed9999 — Original Top-1 / Top-5 (%)

| sub | Top-1 | Top-5 |
|-----|-------|-------|
| 01 | 22.5 | 57.0 |
| 02 | 28.5 | 54.0 |
| 03 | 13.0 | 41.5 |
| 04 | 21.5 | 54.5 |
| 05 | 19.0 | 49.5 |
| 06 | 27.5 | 57.5 |
| 07 | 14.5 | 49.5 |
| 08 | 20.5 | 48.5 |
| 09 | 22.0 | 49.5 |
| 10 | 32.5 | 68.0 |
| **mean** | **22.2** | **53.0** |

Group-mean Top-1 drop when zeroed: delta **−14.8**; gamma bands ≈ 0.

Files: `inter_seed9999/acc_results/`

### vs seed1234 (same protocol)

| Setting | seed1234 mean Top-1 | seed9999 mean Top-1 |
|---------|---------------------|---------------------|
| Intra | 62.6 | **64.6** |
| Inter | 20.8 | **22.2** |

Each subject folder contains:
- `sub-XX_ori_zero_noise_acc.csv` — original / zero / noise per band
- `sub-XX_ori_zero_noise_acc.png` — bar figure
- `all_subs_ori_zero_noise_summary.csv` — all subjects stacked

## Code

| File | Role |
|------|------|
| `code/run_band_ablation_seed9999.py` | Batch: zero → amp/phase → noise → eval (intra+inter) |
| `code/download_ckpts_seed9999.py` | Download seed9999 checkpoints from 2026BMI- |
| `code/download_seed9999_sequential.ps1` | Robust sequential curl download (exact size) |
| `code/download_seed9999_intra_8910.ps1` | Intra-only download helper |
| `code/download_seed9999_inter.ps1` | Inter-only download helper |
| `code/zero_bands_subjects` | Fourier-mask zero 7 bands |
| `code/amplitude&phase(orignal)` | Extract band amplitude & phase |
| `code/gaussian-noise` | Power-matched noise replacement |
| `code/acc_ori_zero_noise` | Eval original vs zero vs noise |

Run from NeuroBridge repo root (needs `data/things_eeg/preprocessed_eeg` + image features):

```powershell
conda activate neurobridge
cd D:\git\NeuroBridge
$env:PYTHONPATH = "."
# checkpoints default under D:\NeuroBridge\checkpoints\seed9999\ (see scripts)
python scripts/download_ckpts_seed9999.py
# or: powershell -File scripts/download_seed9999_sequential.ps1
python scripts/run_band_ablation_seed9999.py --mode both
```

Large EEG `.npy` (zeroed / noise) and `.pth` checkpoints are **not** stored here; regenerate / download with the scripts above.

## Reference

- Checkpoints: [junmoxiao-cloud/2026BMI-](https://github.com/junmoxiao-cloud/2026BMI-) `NeuroBridge/9999-intra-subjects` & `9999-inter-subjects`
- Temporal / delivery notes: [Results_temporal全部产物交付](https://github.com/junmoxiao-cloud/2026BMI-/blob/main/archive/Results_temporal%E5%85%A8%E9%83%A8%E4%BA%A7%E7%89%A9%E4%BA%A4%E4%BB%98.md)
