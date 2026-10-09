# Temporal 数据与流程审计报告（只读审计）

> 本报告由只读审计脚本 `scripts/temporal/audit_temporal_data_and_pipeline.py` 的 Section 6 生成，
> 汇总 Section 1–6 的核验结果，覆盖 spec 疑点 A–E 与图注定量断言判定。
> 审计对象：`d:\NEOschool\Ablation-on-EEG2Image-Models-main`（只读，不修改任何既有结果文件）。

## 1. 审计概要

- 图注定量/结构性断言：共 **19** 条，PASS=15 / STALE=3 / MISMATCH=1。
- 判定规则：PASS=与当前代次数据源一致或结构性成立；STALE=仅旧代次源可复现且当前代次已变；MISMATCH=无源可对应或方向相反。
- 分图分布：Fig. 1(3): P3/S0/M0；Fig. 2(4): P2/S2/M0；Fig. 3(3): P2/S1/M0；Fig. 4(3): P2/S0/M1；Fig. 5(2): P2/S0/M0；Fig. 6(2): P2/S0/M0；Fig. 7(2): P2/S0/M0。
- 窗口标签口径：条件键沿用图注旧字符串（如 `T1_50-150ms`），真实样本切片为 T0 `[0,13)`=0–52ms、T1 `[13,38)`=52–152ms、T2 `[38,75)`=152–300ms；Fig.1 图注已直接使用真实边界 `52-152ms`，属口径说明而非错误。

## 2. 疑点 A–E 判定

| 疑点 | 判定 | 证据 |
| --- | --- | --- |
| A 图注数值与数据源代次不一致 | 确认 | 图注断言 STALE=3、MISMATCH=1（详见第 3 节）|
| B 既有信号级报告 2 项 FAIL | 确认（复现） | T6b: FAIL (DC_diff=4.52e-01); T7a: FAIL (E_in_orig=6.1086e-02, E_in_abl=7.2939e-03, ratio=0.1194)；归因见 2.1 节|
| C 镜像不可独立复现 | 排除 | 缺失仓库内模块=无；硬编码绝对路径 0 条|
| D 统计范式不统一 | 确认 | 口径不一致标记：INCONSISTENT_CORRECTION_SCOPE; INCONSISTENT_TEST_LEVEL; INCONSISTENT_UNIT|
| E 窗口标签与真实样本边界不一致 | 口径说明（非错误） | `T1_50-150ms` 实际样本切片 [13,38)=52–152ms；Fig.1 图注已用真实边界|

### 2.1 疑点 B 两项 FAIL 的归因（结论：阈值过严 / 口径差异，非实现缺陷）

- **T6b（DC preserved，|diff|<1e-4）**：短窗（len<nperseg=32）FFT 回退分支近似精确保均值（|DC_diff|≤6.9e-09）；STFT 分支（保留加窗 0Hz bin）的 |DC_diff| 随窗长增大而减小（最差 T2_150-300ms=4.52e-01 → T_full=6.36e-03）。是否保均值取决于分支口径，非算子缺陷。
- **T7a（in-box energy ratio<0.10）**：实测 ratio=0.1194；掩码执行于 nperseg=32/hop=16/nfft=64，而 in-box 度量基于可视化 STFT（nfft=128、窗宽 128ms，框宽 148ms、窗/框比=0.86）；同口径（段内 FFT）gamma 残留仅 1.07%，残留集中于框边缘帧窗支撑越界。阈值 0.10 过严 + 观测口径错配，非算子缺陷。

## 3. 图注断言判定表

| 断言 | 图 | 断言内容 | 数据源 | 复算值 | 判定 |
| --- | --- | --- | --- | --- | --- |
| F1a | Fig. 1 | V1 峰值窗口(52-152ms) 全频掩码 Δ≈0.190 | `results/temporal/phase3_full_freq_masking/phase3_window_masking_results.csv` | full_freq_T1.top1_drop=0.190 (图注=0.19) | **PASS** |
| F1b | Fig. 1 | N170/Thorpe 窗口(152-300ms) 全频掩码 Δ≈0.140 | `results/temporal/phase3_full_freq_masking/phase3_window_masking_results.csv` | full_freq_T2.top1_drop=0.140 (图注=0.14) | **PASS** |
| F1c | Fig. 1 | 晚期窗口(500-800ms) 解码价值微弱 | `results/temporal/phase3_full_freq_masking/phase3_window_masking_results.csv` | full_freq_T4.top1_drop=0.020 vs T1=0.190/T2=0.140 | **PASS** |
| F2a | Fig. 2 | 7 频段 × 5 时间窗 = 35 条件 | `(结构性)` | canonical 命中 35 条件=True（35/35） | **PASS** |
| F2b | Fig. 2 | 抑制性门控 T1×alpha Δ≈+0.185 | `results/temporal/temporal_stft_ablation/stft_ablation_results.csv | results/temporal/temporal_sub08_phase1_canonical/stft_ablation_results.csv` | legacy=0.185 canonical=0.176 (图注=0.185) | **STALE** |
| F2c | Fig. 2 | 自上而下反馈 T2×beta Δ≈+0.110 | `results/temporal/temporal_stft_ablation/stft_ablation_results.csv | results/temporal/temporal_sub08_phase1_canonical/stft_ablation_results.csv` | legacy=0.11 canonical=0.093 (图注=0.11) | **STALE** |
| F2d | Fig. 2 | 三个 gamma 子频段在所有时间窗近零 | `results/temporal/temporal_stft_ablation/stft_ablation_results.csv | results/temporal/temporal_sub08_phase1_canonical/stft_ablation_results.csv` | max|gamma drop| legacy=0.015 canonical=0.012 | **PASS** |
| F3a | Fig. 3 | 展示 Top-10 时频条件 | `results/temporal/temporal_sub08_phase1_canonical/stft_ablation_results.csv` | canonical 时频条件数=35 | **PASS** |
| F3b | Fig. 3 | 红/蓝柱分别表示 Top-1 / Top-5 下降 | `results/temporal/temporal_sub08_phase1_canonical/stft_ablation_results.csv` | top1_drop 列=True top5_drop 列=True | **PASS** |
| F3c | Fig. 3 | T1×alpha 与 T2×beta 为『仅有的严格显著(** q<0.05)』特征 | `results/temporal/generalization_inter_paired/inter_paired_condition_summary.csv | qcounts(单被试 McNemar)` | 单被试 McNemar 源时频条件 q<0.05=2; 多被试 inter_paired 标注 */** 条件数=13 | **STALE** |
| F4a | Fig. 4 | 振幅缩放因子 α: 0 → 2.0 | `results/temporal/temporal_sub08_phase2_canonical/intra-subjects-averaged/phase2_canonical_averaged.csv` | T1×alpha scaling 参数范围=[0.0, 2.0] | **PASS** |
| F4b | Fig. 4 | T1×alpha α 趋近 0(抑制) 时准确率反而略升 | `results/temporal/temporal_sub08_phase2_canonical/intra-subjects-averaged/phase2_canonical_averaged.csv` | top1(α=0)=0.52 vs top1(α=1.0)=0.696 | **MISMATCH** |
| F4c | Fig. 4 | 人为增强(α>1) 导致性能下降 | `results/temporal/temporal_sub08_phase2_canonical/intra-subjects-averaged/phase2_canonical_averaged.csv` | top1(α=2.0)=0.598 vs top1(α=1.0)=0.696 | **PASS** |
| F5a | Fig. 5 | 随机化比例 0.0 → 1.0 | `results/temporal/temporal_sub08_phase2_canonical/intra-subjects-averaged/phase2_canonical_averaged.csv` | T2×beta phase_rand 参数范围=[0.0, 1.0] | **PASS** |
| F5b | Fig. 5 | T2×beta 在 ratio=1.0 处性能陡降 | `results/temporal/temporal_sub08_phase2_canonical/intra-subjects-averaged/phase2_canonical_averaged.csv` | top1_drop: r=0.5→0.05, r=0.75→0.076, r=1.0→0.104 | **PASS** |
| F6a | Fig. 6 | T2×beta 在 SNR≥0dB 相对稳定，<0dB 后快速恶化 | `results/temporal/temporal_sub08_phase2_canonical/intra-subjects-averaged/phase2_canonical_averaged.csv` | SNR≥0 最大|drop|=0.016; SNR=-10 drop=0.075 | **PASS** |
| F6b | Fig. 6 | 覆盖多档信噪比 SNR | `results/temporal/temporal_sub08_phase2_canonical/intra-subjects-averaged/phase2_canonical_averaged.csv` | distinct snr_db=6 -> [-10.0, -5.0, 0.0, 5.0, 10.0, 20.0] | **PASS** |
| F7a | Fig. 7 | 对比 α=0 / ratio=1.0 / SNR=-10dB 三种最大强度扰动 | `results/temporal/temporal_sub08_phase2_canonical/intra-subjects-averaged/phase2_canonical_averaged.csv` | T1×alpha drops: scaling α0=0.176, phase_rand r1.0=0.287, noise -10dB=0.502 | **PASS** |
| F7b | Fig. 7 | 相位随机化在 T1×alpha 与 T1×beta 上一致造成更高损伤 | `results/temporal/temporal_sub08_phase2_canonical/intra-subjects-averaged/phase2_canonical_averaged.csv` | phase_rand vs scaling α0: T1×alpha 0.287>0.176; T1×beta 0.074>0.043 | **PASS** |

### 3.1 非 PASS 断言明细

- **F2b**（Fig. 2，STALE）：抑制性门控 T1×alpha Δ≈+0.185。数据源 `results/temporal/temporal_stft_ablation/stft_ablation_results.csv | results/temporal/temporal_sub08_phase1_canonical/stft_ablation_results.csv`；复算 legacy=0.185 canonical=0.176 (图注=0.185)。仅旧代次 temporal_stft_ablation 0.185 与图注一致；当前 canonical 0.176 已变
- **F2c**（Fig. 2，STALE）：自上而下反馈 T2×beta Δ≈+0.110。数据源 `results/temporal/temporal_stft_ablation/stft_ablation_results.csv | results/temporal/temporal_sub08_phase1_canonical/stft_ablation_results.csv`；复算 legacy=0.11 canonical=0.093 (图注=0.11)。仅旧代次 temporal_stft_ablation 0.110 与图注一致；当前 canonical 0.093 已变
- **F3c**（Fig. 3，STALE）：T1×alpha 与 T2×beta 为『仅有的严格显著(** q<0.05)』特征。数据源 `results/temporal/generalization_inter_paired/inter_paired_condition_summary.csv | qcounts(单被试 McNemar)`；复算 单被试 McNemar 源时频条件 q<0.05=2; 多被试 inter_paired 标注 */** 条件数=13。图注仅对单被试单次 McNemar 源(results/temporal/temporal_stft_ablation/mcnemar_fdr_results.csv)成立(恰 2 个)，多被试泛化下有 13 个条件显著，结论不一致
- **F4b**（Fig. 4，MISMATCH）：T1×alpha α 趋近 0(抑制) 时准确率反而略升。数据源 `results/temporal/temporal_sub08_phase2_canonical/intra-subjects-averaged/phase2_canonical_averaged.csv`；复算 top1(α=0)=0.52 vs top1(α=1.0)=0.696。实际 α=0(0.52) < α=1.0(0.696)，为下降而非上升

## 4. 审计产物清单

- `results/temporal/verification/audit/temporal_provenance_map.csv`（temporal_provenance_map.csv，行数=62）：溯源映射表（图件→脚本→输入 CSV→被试/种子）
- `results/temporal/verification/audit/temporal_recompute_checks.csv`（temporal_recompute_checks.csv，行数=14）：配对重算与线性恒等式核验
- `results/temporal/verification/audit/temporal_statistics_conventions.csv`（temporal_statistics_conventions.csv，行数=15）：统计口径对照与图注专项核验
- `results/temporal/verification/audit/temporal_reproducibility_gaps.md`（temporal_reproducibility_gaps.md，行数=147）：可复现性与依赖闭合清单
- `results/temporal/verification/audit/temporal_audit_report.md`（temporal_audit_report.md，行数=本报告）：本汇总报告（含 A–E 疑点判定）

## 5. 端到端自查

- 运行方式：`python -B scripts/temporal/audit_temporal_data_and_pipeline.py`（任意 cwd），退出码 0。
- Section 1 溯源表行数=62；Section 2 重算校验行数=14；Section 3 口径对照行数=15。
- Section 4 掩码算子复核：15 PASS / 2 FAIL（与既有报告一致 17/17 项；numpy 等价性 max|Δ|=3.576e-07）。
- Section 5 可复现性扫描：脚本 12 个；缺失仓库内模块=无；硬编码绝对路径 0 条。
- Section 6 图注断言判定：共 19 条，PASS=15 / STALE=3 / MISMATCH=1。
- 幂等性：报告不含时间戳，重跑产物字节不变；脚本不写回任何既有结果文件。
