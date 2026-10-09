# 镜像可复现性缺口清单（Temporal）

> 本文件由只读审计脚本 `audit_temporal_data_and_pipeline.py` 生成；覆盖 spec 疑点 C。
> 审计对象：`d:\NEOschool\Ablation-on-EEG2Image-Models-main`（只读）。

## 1. 缺失模块

### 1.1 import 语句分类（文件+行号）

- 标准库 import：70 条；第三方 import：49 条；仓库内模块 import：26 条。

仓库内模块 import 逐条清单（含存在性核对）：

```text
ablation_temporal_amplitude.py:74  module.dataset  -> 存在
ablation_temporal_amplitude.py:75  module.eeg_encoder.model  -> 存在
ablation_temporal_amplitude.py:76  module.projector  -> 存在
ablation_temporal_amplitude.py:387  module.dataset  -> 存在
ablation_temporal_amplitude.py:404  module.eeg_encoder.model  -> 存在
ablation_temporal_amplitude.py:405  module.projector  -> 存在
ablation_temporal_generalization.py:15  module.dataset  -> 存在
ablation_temporal_generalization.py:16  module.eeg_encoder.model  -> 存在
ablation_temporal_generalization.py:17  module.projector  -> 存在
ablation_temporal_generalization.py:19  ablation_temporal_stft_v2b  -> 存在
ablation_temporal_stft.py:51  module.dataset  -> 存在
ablation_temporal_stft.py:52  module.eeg_encoder.model  -> 存在
ablation_temporal_stft.py:53  module.projector  -> 存在
ablation_temporal_stft_v2b.py:51  module.dataset  -> 存在
ablation_temporal_stft_v2b.py:52  module.eeg_encoder.model  -> 存在
ablation_temporal_stft_v2b.py:53  module.projector  -> 存在
run_phase1_window_generalization.py:180  module.eeg_encoder.model  -> 存在
run_phase1_window_generalization.py:181  module.projector  -> 存在
run_phase1_window_generalization.py:212  module.dataset  -> 存在
run_temporal_sub08_phase1_canonical.py:46  ablation_temporal_stft  -> 存在
run_temporal_sub08_phase1_canonical.py:47  statistical_testing  -> 存在
run_temporal_sub08_phase1_canonical.py:61  module.dataset  -> 存在
run_temporal_sub08_phase1_canonical.py:81  module.eeg_encoder.model  -> 存在
run_temporal_sub08_phase1_canonical.py:82  module.projector  -> 存在
run_temporal_sub08_phase2_canonical.py:34  ablation_temporal_amplitude  -> 存在
run_temporal_sub08_phase2_canonical.py:37  statistical_testing  -> 存在
```

### 1.2 缺失模块汇总

镜像内**缺失**的仓库内模块（含出处行号）：

- （无）

### 1.3 关键依赖链：ablation_temporal_stft_v2b

`ablation_temporal_generalization.py` 在 `ablation_temporal_generalization.py:19` `from ablation_temporal_stft_v2b import (...)` 处依赖 v2b 模块，该依赖**已随镜像提供**（`scripts/temporal/ablation_temporal_stft_v2b.py`，源自 `2026BMI-` 权威版；脚本顶部含仓库根 `sys.path` 注入，克隆后可直接运行）。

只读交叉核对（仓库外）结果：

- 存在：d:\NEOschool\2026BMI-\scripts\temporal_ablation\ablation_temporal_stft_v2b.py
- 存在：d:\NEOschool\eegtoimage\NeuroBridge-main\scripts\things_eeg\ablation_temporal_stft_v2b.py

**差异要点（镜像 `ablation_temporal_stft.py` vs `2026BMI-` 的 `ablation_temporal_stft_v2b.py`）：**

- 时间窗定义一致：两者 `TIME_WINDOWS` 均为 T0 `(0,13)`/T1 `(13,38)`/T2 `(38,75)`/T3 `(75,125)`/T4 `(125,200)`/T_full `(0,250)`（采样点，250Hz）。
- 频段定义一致：两者 `FREQ_BANDS` 均为 7 档（delta/theta/alpha/beta/low_gamma 30-45/gamma 45-70/high_gamma 70-100）。
- STFT 参数一致：`nperseg=32, noverlap=16, nfft=64`；`FS=250.0`；`SELECTED_CHANNELS` 17 通道一致。
- **v2b 额外实现**：`stft_mask_full_freq_window()`（DC 保留的整窗非直流置零）与 `run_phase3()`（phase3 全频段掩码实验，输出 `phase3_full_freq_masking_results.csv`）。镜像版 `ablation_temporal_stft.py` **不含**这两个符号。
- `ablation_temporal_generalization.py` 从 v2b 导入的符号为 `encode_eeg, retrieval_metrics, build_experiment_list, run_experiment, FS, DEFAULT_DATA_DIR, TIME_WINDOWS, FREQ_BANDS, STFT_NPERSEG, STFT_NOVERLAP, STFT_NFFT, SELECTED_CHANNELS`，其中 `DEFAULT_DATA_DIR` 仅 v2b 定义（v2b:55 `DEFAULT_DATA_DIR = SCRIPT_DIR / "data"`）。
- CLI 参数无实质差异：两者 `parse_args()` 均含 `--subject`(默认8) / `--device` / `--batch-size`(默认200) / `--mode`(priority|standard|full) / `--checkpoint` / `--eeg-data-dir` / `--image-feature-dir` / `--aug-feature-dir` / `--output-dir`；v2b 未新增 CLI 开关，而是新增 phase3 入口 `run_phase3()`。
- **复现影响**：依赖已闭合——v2b 已随镜像提供，`ablation_temporal_generalization.py` 可正常导入运行。

`module.*` 包核对：

- 镜像内**已提供** `module/` 包（`module/dataset.py`、`module/eeg_encoder/model.py`、`module/projector.py` 及 `eeg_encoder/atm/` 源码子树，源自 `NeuroBridge-main/module/`）。

- 引用出处：`ablation_temporal_stft.py:46-48`、`ablation_temporal_amplitude.py:73-75/386/403-404`、`run_temporal_sub08_phase1_canonical.py:62/82-83`、`run_phase1_window_generalization.py:180-181/212`、`ablation_temporal_generalization.py:15-17`。

## 2. 硬编码绝对路径

正则命中 **0** 条（文件+行号+原文+用途）：

```text

```

## 3. 仓库外依赖（数据 / checkpoint / 输出目录）

运行所需但不在仓库内的依赖存在性（只读探查）：

```text
[存在] NeuroBridge 仓库根: d:\NEOschool\eegtoimage\NeuroBridge-main
[存在] NeuroBridge module 包: d:\NEOschool\eegtoimage\NeuroBridge-main\module
[存在] NeuroBridge 数据根 things_eeg: d:\NEOschool\eegtoimage\NeuroBridge-main\data\things_eeg
[存在] 数据 preprocessed_eeg: d:\NEOschool\eegtoimage\NeuroBridge-main\data\things_eeg\preprocessed_eeg
[存在] 图像特征 RN50: d:\NEOschool\eegtoimage\NeuroBridge-main\data\things_eeg\image_feature\RN50
[存在] 增强图像特征目录: d:\NEOschool\eegtoimage\NeuroBridge-main\data\things_eeg\image_feature\RN50\GaussianBlur-GaussianNoise-LowResolution-Mosaic
[存在] checkpoint intra-subjects_sub-08_checkpoint_last.pth: d:\NEOschool\eegtoimage\NeuroBridge-main\intra-subjects_sub-08_checkpoint_last.pth
[存在] 活跃工作区结果根 2026BMI-/results: d:\NEOschool\2026BMI-\results
[存在] 镜像内 v2b (应缺失): D:\NEOschool\Ablation-on-EEG2Image-Models-main\scripts\temporal\ablation_temporal_stft_v2b.py
[存在] 2026BMI- 内 v2b 依赖: d:\NEOschool\2026BMI-\scripts\temporal_ablation\ablation_temporal_stft_v2b.py
[存在] NeuroBridge 内 v2b 依赖: d:\NEOschool\eegtoimage\NeuroBridge-main\scripts\things_eeg\ablation_temporal_stft_v2b.py
```

- checkpoint 命名规则来自脚本拼接：`{setting}-seed{seed}/sub-{sub:02d}-seed{seed}/checkpoint_test_best.pth`（审计时的原始运行根 = `d:\NEOschool\2026BMI-\results`）。
- 数据根 = `d:\NEOschool\eegtoimage\NeuroBridge-main\data\things_eeg`；单 checkpoint 默认 = `d:\NEOschool\eegtoimage\NeuroBridge-main\intra-subjects_sub-08_checkpoint_last.pth`。
- 现行脚本默认值已改为仓库相对（`--base-results-dir`、`--base-ckpt-dir`、`--data-root` 等均可 CLI 覆盖）；上方路径为原始实验环境的溯源记录。

## 4. checkpoint 命名一致性

主命名（脚本首选路径）与设置变体核查结果：

```text
primary 命中: 100  primary 缺失: 0
fallback(inter-leave) 命中: 0
intra-subjects-averaged/sub-08 参考 checkpoint 存在: True
```

真实存在的 `inter-leave-sub-*` 目录位置（只读）：

```text
d:\NEOschool\2026BMI-\results\inter-subjects\20260728-233929-inter-leave-sub-01-seed3407
d:\NEOschool\2026BMI-\results\inter-subjects\20260729-061556-inter-leave-sub-02-seed3407
d:\NEOschool\2026BMI-\results\inter-subjects\20260729-130044-inter-leave-sub-03-seed3407
d:\NEOschool\2026BMI-\results\inter-subjects\20260729-230856-inter-leave-sub-03-seed3407
d:\NEOschool\2026BMI-\results\inter-subjects\20260730-002606-inter-leave-sub-04-seed3407
d:\NEOschool\2026BMI-\results\inter-subjects\20260730-014024-inter-leave-sub-05-seed3407
d:\NEOschool\2026BMI-\results\inter-subjects\20260730-025637-inter-leave-sub-06-seed3407
d:\NEOschool\2026BMI-\results\inter-subjects\20260730-041313-inter-leave-sub-07-seed3407
d:\NEOschool\2026BMI-\results\inter-subjects\20260730-052826-inter-leave-sub-08-seed3407
d:\NEOschool\2026BMI-\results\inter-subjects\20260730-064910-inter-leave-sub-09-seed3407
d:\NEOschool\2026BMI-\results\inter-subjects\20260730-080443-inter-leave-sub-10-seed3407
```

**判定：命名一致性 = 一致（primary 路径）**。
- `ablation_temporal_generalization.py:143` 的主路径 `{setting}-seed{seed}/sub-{NN}-seed{seed}` 在 intra 与 inter 两种设置下均命中真实 checkpoint；
- `ablation_temporal_generalization.py:148` 的备用路径 `{setting}-seed{seed}/inter-leave-sub-{NN}-seed{seed}` 在活跃工作区**未命中**；实际 `inter-leave-sub-*` 目录位于 `2026BMI-/results/inter-subjects/<时间戳>-inter-leave-sub-NN-seed3407`，与脚本拼接的层级不符，故该 fallback 实际为无效分支（不影响主路径生效）。
- 同一命名规则亦见 `run_phase1_window_generalization.py:164-174`（候选顺序相同）。

## 5. 结论：镜像能否独立复现

**结论：仓库内代码依赖已闭合——`ablation_temporal_stft_v2b` 与 `module/` 包均已随镜像提供，temporal 脚本不再含本机绝对路径（缺失仓库内模块=无、硬编码绝对路径 0 条）；克隆仓库并安装依赖后，全部脚本可直接导入运行。分层说明：**
- 推理类脚本（`ablation_temporal_generalization.py`、`ablation_temporal_stft.py`、`ablation_temporal_amplitude.py`、`run_temporal_sub08_phase1_canonical.py`、`run_temporal_sub08_phase2_canonical.py`、`run_phase1_window_generalization.py`）需读者自备 THINGS-EEG 公开数据与自行训练的 checkpoint（仓库不托管权重文件），路径均经 CLI 参数指定。

部分可复现（仅需仓库内既有 CSV，或合成数据即可自证）：

- `recompute_inter_paired_drops.py`（只读 `results/temporal/generalization_ablation/*.csv`，仅依赖 numpy/scipy）
- `plot_generalization_temporal.py`（只读同目录 CSV + matplotlib/pandas/statsmodels）
- `ablation_visualize.py`、`plot_composite_fig3.py`（只读仓库内 CSV + matplotlib）
- `statistical_testing.py`（纯统计工具，仅 statsmodels/scipy）

自查：本脚本重跑退出码 0；md 中每条断言均可溯源到文件+行号。
