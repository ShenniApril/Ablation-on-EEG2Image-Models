# The accuracy of inter and intra with seed2025

Things-EEG **original vs band-zero vs power-matched Gaussian noise** Top-1 / Top-5 accuracy for **10 subjects**, using **seed=2025** checkpoints from [junmoxiao-cloud/2026BMI-](https://github.com/junmoxiao-cloud/2026BMI-).

## Protocol

| Setting | Encoder | Channels | Checkpoint source |
|---------|---------|----------|-------------------|
| **Intra** | EEGProject | 17 (O+P) | `NeuroBridge/2025-intra-subjects/.../checkpoint_test_best.pth` |
| **Inter** | TSConv | all 63 | `NeuroBridge/2025-inter-subjects/.../checkpoint_test_best.pth` |

- Seed: **2025** (checkpoint training seed + Gaussian-noise RNG; zero-band EEG is seed-independent)
- Bands (Hz): delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30, low_gamma 30–45, gamma 45–70, high_gamma 70–100
- Zero: Fourier-mask band erasure (`code/zero_bands_subjects`)
- Noise: power-matched complex Gaussian from original amplitude & phase (`code/gaussian-noise`, `--seed 2025`)
- Eval: `code/acc_ori_zero_noise` (CPU, `image_test_aug=True`, RN50 fused aug features)

Sister repo (seed9999): [The-accuracy-of-inter-and-intra-with-seed9999](https://github.com/laura-gendon/The-accuracy-of-inter-and-intra-with-seed9999)

## Results

### Intra seed2025 — Original Top-1 / Top-5 (%)

| sub | Top-1 | Top-5 |
|-----|-------|-------|
| 01 | 50.0 | 80.5 |
| 02 | 64.5 | 92.5 |
| 03 | 60.5 | 88.5 |
| 04 | 64.0 | 88.0 |
| 05 | 55.0 | 84.5 |
| 06 | 71.0 | 94.5 |
| 07 | 64.0 | 89.5 |
| 08 | 71.0 | 94.0 |
| 09 | 60.0 | 92.5 |
| 10 | 73.0 | 98.0 |
| **mean** | **63.3** | **90.2** |

Group-mean Top-1 drop when zeroed: theta **-31.8**, beta -25.2, delta -22.8, alpha -21.1; gamma bands ≈ 0.

Files: `intra_seed2025/acc_results/`

### Inter seed2025 — Original Top-1 / Top-5 (%)

| sub | Top-1 | Top-5 |
|-----|-------|-------|
| 01 | 22.5 | 55.5 |
| 02 | 26.5 | 51.5 |
| 03 | 18.0 | 42.5 |
| 04 | 19.5 | 50.5 |
| 05 | 15.0 | 47.0 |
| 06 | 30.0 | 58.5 |
| 07 | 16.5 | 48.5 |
| 08 | 19.5 | 49.0 |
| 09 | 20.0 | 50.5 |
| 10 | 34.5 | 62.0 |
| **mean** | **22.2** | **51.5** |

Group-mean Top-1 drop when zeroed: delta **-14.2**; theta/alpha/beta ≈ −5 to −6; gamma bands ≈ 0.

Files: `inter_seed2025/acc_results/`

### Intra vs Inter (same seed2025 weights)

| Setting | mean Top-1 | mean Top-5 |
|---------|------------|------------|
| **Intra** | **63.3** | **90.2** |
| **Inter** | **22.2** | **51.5** |
| **Intra − Inter** | **+41.1** | **+38.7** |

Each subject folder contains:
- `sub-XX_ori_zero_noise_acc.csv` — original / zero / noise per band
- `sub-XX_ori_zero_noise_acc.png` — bar figure
- `all_subs_ori_zero_noise_summary.csv` — all subjects stacked

## Code

| File | Role |
|------|------|
| `code/run_band_ablation_seed2025.py` | Batch: zero → amp/phase → noise → eval (intra+inter) |
| `code/download_ckpts_seed2025.py` | Download seed2025 checkpoints from 2026BMI- |
| `code/download_seed2025_inter_curl.ps1` | Curl download all inter checkpoints |
| `code/zero_bands_subjects` | Fourier-mask zero 7 bands |
| `code/amplitude&phase(orignal)` | Extract band amplitude & phase |
| `code/gaussian-noise` | Power-matched noise replacement |
| `code/acc_ori_zero_noise` | Eval original vs zero vs noise |
| `code/print_seed2025_comparison.py` | Print intra vs inter summary from local CSVs |

Run from NeuroBridge repo root (needs `data/things_eeg/preprocessed_eeg` + image features). Copy `code/*` into the matching NeuroBridge paths or run with `PYTHONPATH=.` and paths as in scripts.

```powershell
conda activate neurobridge
cd D:\git\NeuroBridge
$env:PYTHONPATH = "."
python scripts/download_ckpts_seed2025.py
python scripts/run_band_ablation_seed2025.py --mode both
```

Large EEG `.npy` (zeroed / noise) and `.pth` checkpoints are **not** stored here; regenerate / download with the scripts above.

## Reference

- Checkpoints: [junmoxiao-cloud/2026BMI-](https://github.com/junmoxiao-cloud/2026BMI-) `NeuroBridge/2025-intra-subjects` & `2025-inter-subjects`
