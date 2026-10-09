# Ablation-on-EEG2Image-Models
Our ablation framework to investigate EEG2Image Model's dependency on spectral, temporal and spatial neural features.

## Temporal Ablation
The temporal ablation study components are organized as follows:
- **`scripts/temporal/`**: Core execution scripts for temporal ablation (e.g., STFT, amplitude, generalization), statistical testing, and visualization.
- **`results/temporal/`**: Raw results generated from different phases of temporal ablation.
- **`assets/temporal/`**: Final visualizations, heatmaps, and summary charts (e.g., phase 2 heatmaps, generalization charts) related to temporal ablation experiments.

## Temporal Experiments — Reproduction Guide

The temporal pipeline can be reproduced at two levels. All commands below are
run from the repository root, and all flags are written out explicitly — do not
rely on the scripts' built-in defaults.

### Tier 1 — read-only layer (committed results only)

After `pip install -r requirements.txt` (the Tier 1 packages are enough for
this section), everything below reproduces statistics and figures from the
CSVs committed under `results/temporal/`. No dataset or checkpoint is needed.

1. Group-level paired recalculation of the inter-subject generalization drops
   (pairs each subject's baseline vs. ablated Top-1 accuracy, one-sample
   t-tests on the ten subject-level differences, BH-FDR over the 35 conditions):

   ```bash
   python scripts/temporal/recompute_inter_paired_drops.py \
       --input-dir results/temporal/generalization_ablation \
       --output-dir results/temporal/generalization_inter_paired
   ```

   Outputs: `results/temporal/generalization_inter_paired/inter_paired_subject_drops.csv`
   and `inter_paired_condition_summary.csv`.

2. Group-level temporal generalization heatmaps:

   ```bash
   python scripts/temporal/plot_generalization_temporal.py --output-dir assets/temporal
   ```

   Reads the subject CSVs in `results/temporal/generalization_ablation/` and
   writes the intra-/inter-subject heatmaps into `assets/temporal/`.

3. Phase 1/2/3 ablation figures:

   ```bash
   python scripts/temporal/ablation_visualize.py --phase 123 \
       --phase1-csv results/temporal/temporal_sub08_phase1_canonical/stft_ablation_results.csv \
       --phase2-csv results/temporal/temporal_sub08_phase2_canonical/intra-subjects-averaged/phase2_canonical_averaged.csv \
       --phase3-csv results/temporal/phase3_full_freq_masking/phase3_window_masking_results.csv \
       --mcnemar-csv results/temporal/temporal_sub08_phase2_canonical/intra-subjects-averaged/mcnemar_fdr_results.csv \
       --output-dir assets/temporal_ablation
   ```

   The CSV flags matter: the script's built-in paths assume the historical
   `scripts/temporal/results/` layout, so point them at `results/temporal/...`
   as above. Figures land in `assets/temporal_ablation/`.

4. Composite Fig-3 panel figure. `plot_composite_fig3.py` takes no CLI
   arguments and also reads from the historical `scripts/temporal/results/`
   layout (in the original NeuroBridge-main checkout these CSVs lived next to
   the script), so stage the committed CSVs there first (copy or symlink):

   ```bash
   mkdir -p scripts/temporal/results/temporal_stft_ablation \
            scripts/temporal/results/temporal_amplitude_ablation \
            scripts/temporal/results/phase3_full_freq_masking
   cp results/temporal/temporal_stft_ablation/stft_ablation_results.csv \
      scripts/temporal/results/temporal_stft_ablation/
   cp results/temporal/temporal_amplitude_ablation/amplitude_ablation_results.csv \
      scripts/temporal/results/temporal_amplitude_ablation/
   cp results/temporal/phase3_full_freq_masking/phase3_window_masking_results.csv \
      scripts/temporal/results/phase3_full_freq_masking/
   cp results/temporal/mcnemar_fdr_results.csv scripts/temporal/results/
   python scripts/temporal/plot_composite_fig3.py
   ```

   Output: `assets/temporal_ablation/Fig3_Temporal_Composite.png`.

5. Statistical testing over the raw per-condition predictions (McNemar exact
   test + BH-FDR). `--output` is required:

   ```bash
   python scripts/temporal/statistical_testing.py \
       --phase1 results/temporal/temporal_stft_ablation/stft_ablation_details.json \
       --phase2 results/temporal/temporal_amplitude_ablation/amplitude_ablation_details.json \
       --phase3 results/temporal/phase3_full_freq_masking/phase3_full_freq_masking_details.json \
       --output results/temporal/verification/statistical_testing_report.csv
   ```

6. Read-only data/pipeline audit. No CLI arguments; writes five artifacts to
   `results/temporal/verification/audit/` (`temporal_provenance_map.csv`,
   `temporal_recompute_checks.csv`, `temporal_statistics_conventions.csv`,
   `temporal_reproducibility_gaps.md`, `temporal_audit_report.md`):

   ```bash
   python scripts/temporal/audit_temporal_data_and_pipeline.py
   ```

   The audit also probes external paths from the original workstation (the
   local NeuroBridge data tree and the active results root). On any other
   machine those entries are simply reported as missing / reproducibility
   gaps; the audit artifacts are still produced.

### Tier 2 — inference layer (needs public data + your own checkpoints)

Everything under `scripts/temporal/` that loads a NeuroBridge checkpoint
belongs to this tier. You need (a) the preprocessed THINGS-EEG data and image
features, and (b) checkpoints you trained yourself — this repository hosts no
weights.

#### Data acquisition

The ablation scripts consume the same preprocessed THINGS-EEG layout as
NeuroBridge (full preparation instructions in
`scripts/original scripts of NeuroBridge/README.md`):

```
data/things_eeg/
├── preprocessed_eeg/          # info.json + sub-01..sub-10/{train.npy,test.npy}
└── image_feature/
    └── RN50/                  # image_train.npy, image_test.npy
        └── GaussianBlur-GaussianNoise-LowResolution-Mosaic/
                              # train.npy, test.npy (fused augmentation features)
```

- Raw THINGS-EEG comes from the OSF repositories referenced in
  `scripts/original scripts of NeuroBridge/README.md` (EEG: osf.io/crxs4,
  images: osf.io/y63gw); `scripts/things_eeg/download_eeg.sh` automates the
  EEG download.
- Preprocessing (`preprocess_eeg.py --mvnn`) and image-feature
  extraction/fusing (`extract_feature.py`, `fuse_feature.py`, wrapped by
  `scripts/things_eeg/image_feature_extract.sh`) live in
  `scripts/original scripts of NeuroBridge/`; their extra dependencies (mne,
  scikit-learn, open_clip_torch, torchvision, transformers, ...) are listed in
  the upstream requirements.txt inside that directory. That README also points
  to ready-made preprocessed data downloads from the upstream NeuroBridge
  repository.

#### Checkpoints

This repository does not host any model weights: one checkpoint is ~77 MB, and
the full sweep (10 subjects × 5 training seeds × the intra-/inter-subject
settings, ~100 checkpoints) would be ~7.7 GB. Train your own checkpoints with
the upstream NeuroBridge pipeline (`scripts/original scripts of
NeuroBridge/train.py`, or the shell wrappers
`scripts/things_eeg/intra-subjects.sh` / `inter-subjects.sh`). Each training
run saves `checkpoint_last.pth` and `checkpoint_test_best.pth`; per-subject
files follow the upstream naming, e.g. `intra-subjects_sub-08_checkpoint_last.pth`.

The ablation scripts expect a checkpoint root laid out as:

```
<checkpoint-root>/
├── intra-subjects-seed<seed>/sub-<NN>-seed<seed>/checkpoint_test_best.pth
└── inter-subjects-seed<seed>/sub-<NN>-seed<seed>/checkpoint_test_best.pth
```

- `<setting>` ∈ {`intra-subjects`, `inter-subjects`}; the inter-subject
  runners also accept the alternative directory name
  `inter-leave-sub-<NN>-seed<seed>`.
- Training seeds used throughout the study: 1234, 2025, 42, 9999, 3407.

#### Run commands

In the examples below, `<ckpt>` is your checkpoint root and `<data>` is your
`data/things_eeg` root. Point `--output-dir` / `--output-root` at fresh
directories if you want to keep the committed results intact.

Note: `ablation_temporal_stft.py` and `ablation_temporal_stft_v2b.py` import
the `module` package and self-inject the repository root into `sys.path`, so
either invocation works from the repository root:
`python -m scripts.temporal.<name>` or `python scripts/temporal/<name>.py`.

Phase 1 — single-subject STFT time-frequency masking:

```bash
python -m scripts.temporal.ablation_temporal_stft --subject 8 --device cuda --batch-size 200 \
    --mode priority \
    --checkpoint <ckpt>/intra-subjects-seed2025/sub-08-seed2025/checkpoint_test_best.pth \
    --eeg-data-dir <data>/preprocessed_eeg \
    --image-feature-dir <data>/image_feature/RN50 \
    --aug-feature-dir <data>/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic \
    --output-dir results/temporal/temporal_stft_ablation
```

The DC-preserving variant (`ablation_temporal_stft_v2b.py`, the condition
definitions used by the Phase 2/3 scripts) takes the same flags:

```bash
python -m scripts.temporal.ablation_temporal_stft_v2b --subject 8 --device cuda --batch-size 200 \
    --mode priority \
    --checkpoint <ckpt>/intra-subjects-seed2025/sub-08-seed2025/checkpoint_test_best.pth \
    --eeg-data-dir <data>/preprocessed_eeg \
    --image-feature-dir <data>/image_feature/RN50 \
    --aug-feature-dir <data>/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic \
    --output-dir results/temporal/temporal_stft_ablation_v2b
```

Phase 2 — amplitude perturbation (omit `--conditions` to auto-pick the top-N
conditions from the Phase 1 CSV):

```bash
python scripts/temporal/ablation_temporal_amplitude.py --subject 8 --device cuda --batch-size 200 \
    --top-n 3 --types scaling phase noise \
    --phase1-csv results/temporal/temporal_stft_ablation/stft_ablation_results.csv \
    --checkpoint <ckpt>/intra-subjects-seed2025/sub-08-seed2025/checkpoint_test_best.pth \
    --eeg-data-dir <data>/preprocessed_eeg \
    --image-feature-dir <data>/image_feature/RN50 \
    --aug-feature-dir <data>/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic \
    --output-dir results/temporal/temporal_amplitude_ablation
```

Phase 2 — STFT masking generalization sweep over the five seeds and both
settings. This script has no per-run `--checkpoint` flag: it locates
checkpoints itself under the `--base-results-dir` root (default: the
repository's `results/`), using the
`{setting}-seed{seed}/sub-<NN>-seed{seed}/checkpoint_test_best.pth` layout
above — point `--base-results-dir` at your own checkpoint root:

```bash
python scripts/temporal/ablation_temporal_generalization.py --subject 8 --device cuda --batch-size 200 \
    --mode priority \
    --base-results-dir <ckpt> \
    --eeg-data-dir <data>/preprocessed_eeg \
    --image-feature-dir <data>/image_feature/RN50 \
    --aug-feature-dir <data>/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic \
    --output-dir results/temporal/generalization_ablation
```

Group Phase 1 — DC-preserving window masking across subjects/seeds (resumable;
`--neurobridge-root` must point at a code tree containing `module/`, e.g. this
mirror's root; `--data-root` is the `data/things_eeg` root):

```bash
python scripts/temporal/run_phase1_window_generalization.py \
    --neurobridge-root . \
    --data-root <data> \
    --checkpoint-root <ckpt> \
    --output-dir results/temporal/phase1_window_group \
    --settings intra-subjects inter-subjects \
    --subjects 1 2 3 4 5 6 7 8 9 10 \
    --seeds 1234 2025 42 9999 3407 \
    --device cuda --batch-size 200
```

Canonical sub-08 Phase 1 (runs the five intra-subject seeds and averages):

```bash
python scripts/temporal/run_temporal_sub08_phase1_canonical.py --device cuda \
    --mode priority \
    --base-ckpt-dir <ckpt> \
    --eeg-data-dir <data>/preprocessed_eeg \
    --image-feature-dir <data>/image_feature/RN50 \
    --aug-feature-dir <data>/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic \
    --output-dir results/temporal/temporal_sub08_phase1_canonical
```

Canonical sub-08 Phase 2 (independent seed checkpoints, then averages the
result tables; `--results-root` is the checkpoint root):

```bash
python scripts/temporal/run_temporal_sub08_phase2_canonical.py --subject 8 \
    --settings intra-subjects-averaged inter-subjects-averaged \
    --seeds 42 3407 1234 2025 9999 \
    --device cuda --batch-size 200 --top-n 3 \
    --phase1-csv results/temporal/temporal_sub08_phase1_canonical/stft_ablation_results.csv \
    --results-root <ckpt> \
    --output-root results/temporal/temporal_sub08_phase2_canonical \
    --eeg-data-dir <data>/preprocessed_eeg \
    --image-feature-dir <data>/image_feature/RN50 \
    --aug-feature-dir <data>/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic
```

### Time-window naming caveat

The condition keys in the committed CSVs (e.g. `T0_0-50ms`, `T1_50-150ms`,
`T2_150-300ms`, `T3_300-500ms`, `T4_500-800ms`) are legacy nominal labels. The
actual sample slices (THINGS-EEG preprocessed at FS = 250 Hz, one sample =
4 ms) are:

| Key (nominal)  | Samples     | True window  |
| -------------- | ----------- | ------------ |
| `T0_0-50ms`    | `[0, 13)`   | 0–52 ms      |
| `T1_50-150ms`  | `[13, 38)`  | 52–152 ms    |
| `T2_150-300ms` | `[38, 75)`  | 152–300 ms   |
| `T3_300-500ms` | `[75, 125)` | 300–500 ms   |
| `T4_500-800ms` | `[125, 200)`| 500–800 ms   |

Figures produced by the plotting scripts label the windows with these true
boundaries (e.g. "0-52ms", "52-152ms", ...).

### Repository layout

- **`scripts/temporal/`** — temporal ablation scripts: Phase 1 STFT masking
  (`ablation_temporal_stft.py`, `ablation_temporal_stft_v2b.py`), Phase 2
  amplitude perturbation and generalization sweeps
  (`ablation_temporal_amplitude.py`, `ablation_temporal_generalization.py`),
  group/canonical runners (`run_phase1_window_generalization.py`,
  `run_temporal_sub08_phase{1,2}_canonical.py`), statistics
  (`statistical_testing.py`, `recompute_inter_paired_drops.py`), plotting
  (`plot_generalization_temporal.py`, `ablation_visualize.py`,
  `plot_composite_fig3.py`) and the read-only audit
  (`audit_temporal_data_and_pipeline.py`).
- **`module/`** — the NeuroBridge model code (dataset, EEG encoder,
  projectors) imported by the inference-layer scripts.
- **`results/temporal/`** — committed per-phase result CSVs/JSONs (Phase 1/2/3,
  generalization, inter-subject paired recomputation, verification outputs).
- **`assets/temporal/`** — final figures (heatmaps, generalization charts).
- Also present: `scripts/original scripts of NeuroBridge/` (upstream
  training / preprocessing / feature-extraction entry points), 
  `scripts/things_eeg/` (download and training shell scripts), and the spatial
  ablation counterpart (`scripts/spatial/`, `assets/spatial/`).
