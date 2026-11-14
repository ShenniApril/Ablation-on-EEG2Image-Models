# The accuracy of inter and intra with seed3407

Things-EEG **original vs band-zero vs power-matched Gaussian noise** Top-1 / Top-5 accuracy for **10 subjects**, using **seed=3407** checkpoints from [junmoxiao-cloud/2026BMI-](https://github.com/junmoxiao-cloud/2026BMI-).

## Protocol

| Setting | Encoder | Channels | Checkpoint source |
|---------|---------|----------|-------------------|
| **Intra** | EEGProject | 17 (O+P) | `results/intra-subjects-seed3407/.../checkpoint_test_best.pth` |
| **Inter** | TSConv | all 63 | `results/inter-subjects-seed3407/.../checkpoint_test_best.pth` |

- Seed: **3407** (checkpoint training seed + Gaussian-noise RNG; zero-band EEG is seed-independent)
- Bands (Hz): delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30, low_gamma 30–45, gamma 45–70, high_gamma 70–100
- Zero: Fourier-mask band erasure (`code/zero_bands_subjects`)
- Noise: power-matched complex Gaussian from original amplitude & phase (`code/gaussian-noise`, `--seed 3407`)
- Eval: `code/acc_ori_zero_noise` (CPU, `image_test_aug=True`, RN50 fused aug features)

Sister repos: [seed2025](https://github.com/laura-gendon/The-accuracy-of-inter-and-intra-with-seed2025), [seed9999](https://github.com/laura-gendon/The-accuracy-of-inter-and-intra-with-seed9999)

## Results

### Intra seed3407 — Original Top-1 / Top-5 (%)

| sub | Top-1 | Top-5 |
|-----|-------|-------|
| 01 | 53.5 | 82.5 |
| 02 | 63.5 | 93.5 |
| 03 | 57.5 | 92.0 |
| 04 | 61.0 | 90.0 |
| 05 | 56.5 | 86.0 |
| 06 | 67.0 | 93.5 |
| 07 | 61.5 | 90.5 |
| 08 | 69.0 | 93.5 |
| 09 | 60.5 | 89.5 |
| 10 | 76.0 | 97.0 |
| **mean** | **62.6** | **90.8** |

Group-mean Top-1 drop when zeroed: theta **-31.2**, beta -24.3, delta -23.1, alpha -20.2; gamma bands ≈ 0.

Files: `intra_seed3407/acc_results/`

### Inter seed3407 — Original Top-1 / Top-5 (%)

| sub | Top-1 | Top-5 |
|-----|-------|-------|
| 01 | 22.0 | 54.0 |
| 02 | 22.5 | 52.5 |
| 03 | 14.5 | 40.0 |
| 04 | 18.5 | 51.0 |
| 05 | 18.0 | 46.0 |
| 06 | 25.5 | 55.0 |
| 07 | 20.5 | 50.5 |
| 08 | 17.0 | 48.5 |
| 09 | 16.0 | 45.0 |
| 10 | 35.5 | 68.5 |
| **mean** | **21.0** | **51.1** |

Group-mean Top-1 drop when zeroed: delta **-14.2**; theta/alpha/beta ≈ −5 to −6; gamma bands ≈ 0.

Files: `inter_seed3407/acc_results/`

### Intra vs Inter (same seed3407 weights)

| Setting | mean Top-1 | mean Top-5 |
|---------|------------|------------|
| **Intra** | **62.6** | **90.8** |
| **Inter** | **21.0** | **51.1** |
| **Intra − Inter** | **+41.6** | **+39.7** |

Each subject folder contains:
- `sub-XX_ori_zero_noise_acc.csv` — original / zero / noise per band
- `sub-XX_ori_zero_noise_acc.png` — bar figure
- `all_subs_ori_zero_noise_summary.csv` — all subjects stacked

## Code

| File | Role |
|------|------|
| `code/run_band_ablation_seed3407.py` | Batch: zero → amp/phase → noise → eval (intra+inter) |
| `code/download_ckpts_seed3407.py` | Download seed3407 checkpoints from 2026BMI- |
| `code/download_seed3407_all_curl.ps1` | Curl download all intra+inter checkpoints |
| `code/zero_bands_subjects` | Fourier-mask zero 7 bands |
| `code/amplitude&phase(orignal)` | Extract band amplitude & phase |
| `code/gaussian-noise` | Power-matched noise replacement |
| `code/acc_ori_zero_noise` | Eval original vs zero vs noise |
| `code/print_seed3407_comparison.py` | Print intra vs inter summary from local CSVs |

Run from NeuroBridge repo root (needs `data/things_eeg/preprocessed_eeg` + image features). Copy `code/*` into the matching NeuroBridge paths or run with `PYTHONPATH=.` and paths as in scripts.

```powershell
conda activate neurobridge
cd D:\git\NeuroBridge
$env:PYTHONPATH = "."
python scripts/download_ckpts_seed3407.py
python scripts/run_band_ablation_seed3407.py --mode both
```

Large EEG `.npy` (zeroed / noise) and `.pth` checkpoints are **not** stored here; regenerate / download with the scripts above.

## Reference

- Checkpoints: [junmoxiao-cloud/2026BMI-](https://github.com/junmoxiao-cloud/2026BMI-) `results/intra-subjects-seed3407` & `results/inter-subjects-seed3407`
