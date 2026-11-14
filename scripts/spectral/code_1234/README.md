# The accuracy of inter and intra with seed1234

Things-EEG **original vs band-zero vs power-matched Gaussian noise** Top-1 / Top-5 accuracy for **10 subjects**, using **seed=1234** checkpoints from [junmoxiao-cloud/2026BMI-](https://github.com/junmoxiao-cloud/2026BMI-).

## Protocol

| Setting | Encoder | Channels | Checkpoint source |
|---------|---------|----------|-------------------|
| **Intra** | EEGProject | 17 (O+P) | `results/intra-subjects-seed1234/sub-XX-seed1234/checkpoint_test_best.pth` |
| **Inter** | TSConv | all 63 | `NeuroBridge/1234-inter-subjects/.../checkpoint_test_best.pth` |

- Seed: **1234** (training + noise RNG + eval)
- Bands (Hz): delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30, low_gamma 30–45, gamma 45–70, high_gamma 70–100
- Zero: Fourier-mask band erasure (`code/zero_bands_subjects`)
- Noise: power-matched complex Gaussian from original amplitude & phase (`code/gaussian-noise`)
- Eval: `code/acc_ori_zero_noise` (CPU, `image_test_aug=True`, RN50 fused aug features)

## Results

### Intra seed1234 — Original Top-1 / Top-5 (%)

| sub | Top-1 | Top-5 |
|-----|-------|-------|
| 01 | 53.0 | 78.5 |
| 02 | 61.0 | 92.0 |
| 03 | 58.0 | 90.5 |
| 04 | 61.5 | 91.0 |
| 05 | 54.5 | 88.5 |
| 06 | 70.5 | 91.5 |
| 07 | 64.0 | 91.0 |
| 08 | 67.5 | 92.0 |
| 09 | 62.0 | 90.5 |
| 10 | 74.0 | 98.0 |
| **mean** | **62.6** | **90.3** |

Group-mean Top-1 drop when zeroed: theta **−30.6**, beta −23.6, delta −21.4, alpha −20.2; gamma bands ≈ 0.

Files: `intra_seed1234/acc_results/`

### Inter seed1234 — Original Top-1 / Top-5 (%)

| sub | Top-1 | Top-5 |
|-----|-------|-------|
| 01 | 22.0 | 52.5 |
| 02 | 22.0 | 50.0 |
| 03 | 11.5 | 34.5 |
| 04 | 18.0 | 52.0 |
| 05 | 15.5 | 45.5 |
| 06 | 29.5 | 56.5 |
| 07 | 17.0 | 49.0 |
| 08 | 20.5 | 42.0 |
| 09 | 18.5 | 51.0 |
| 10 | 33.5 | 65.0 |
| **mean** | **20.8** | **49.8** |

Group-mean Top-1 drop when zeroed: delta **−13.0**; gamma bands ≈ 0.

Files: `inter_seed1234/acc_results/`

Each subject folder contains:
- `sub-XX_ori_zero_noise_acc.csv` — original / zero / noise per band
- `sub-XX_ori_zero_noise_acc.png` — bar figure
- `all_subs_ori_zero_noise_summary.csv` — all subjects stacked

## Code

| File | Role |
|------|------|
| `code/run_band_ablation_seed1234.py` | Batch: zero → amp/phase → noise → eval (intra+inter) |
| `code/download_ckpts_seed1234.py` | Download seed1234 checkpoints from 2026BMI- |
| `code/zero_bands_subjects` | Fourier-mask zero 7 bands |
| `code/amplitude&phase(orignal)` | Extract band amplitude & phase |
| `code/gaussian-noise` | Power-matched noise replacement |
| `code/acc_ori_zero_noise` | Eval original vs zero vs noise |

Run from NeuroBridge repo root (needs `data/things_eeg/preprocessed_eeg` + image features):

```powershell
conda activate neurobridge
cd D:\git\NeuroBridge
$env:PYTHONPATH = "."
python scripts/download_ckpts_seed1234.py
python scripts/run_band_ablation_seed1234.py --mode both
```

Large EEG `.npy` (zeroed / noise) are not stored here; regenerate with the scripts above.

## Reference

- Checkpoints & temporal generalization context: [Results_temporal全部产物交付](https://github.com/junmoxiao-cloud/2026BMI-/blob/main/archive/Results_temporal%E5%85%A8%E9%83%A8%E4%BA%A7%E7%89%A9%E4%BA%A4%E4%BB%98.md) (Fig. 8–13, intra/inter seed1234)
