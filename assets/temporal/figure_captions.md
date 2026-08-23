# Temporal Ablation Figure Captions

This document contains publication-ready captions for the temporal ablation figures. Each caption follows the structure: **Objective description of visual elements** + **Scientific interpretation/Conclusion**.

---

## Fig. 1. Phase 1: Full Frequency Masking
**English Caption:**
**Fig. 1. Phase 1: Full-Frequency Window Masking.** Displays the impact of entirely masking all frequency components within defined temporal windows. The extreme accuracy drop ($\Delta=0.190$) during the V1 peak window (52-152ms), followed by the N170/Thorpe window (152-300ms, $\Delta=0.140$), validates classic visual evoked potential (VEP) timelines. It demonstrates that early visual processing stages are the absolute bottleneck for the deep learning model's classification capability, whereas late cognitive processing (500-800ms) provides marginal decoding value.

**Chinese Caption:**
**图 1. 第一阶段：全频段时间窗掩码。** 展示了完全掩蔽特定时间窗内所有频率成分的影响。在 V1 峰值窗口（52-152ms）出现的极端准确率下降（$\Delta=0.190$），以及随后的 N170/Thorpe 窗口（152-300ms, $\Delta=0.140$），完美验证了经典的视觉诱发电位（VEP）时间线。这表明早期的视觉处理阶段是深度学习模型分类能力的绝对瓶颈，而晚期的认知处理阶段（500-800ms）提供的解码价值微乎其微。

---

## Fig. 2. Phase 2: STFT Time-Frequency Masking Heatmap
**English Caption:**
**Fig. 2. Phase 2: STFT Time-Frequency Masking Heatmap.** Illustrates the Top-1 accuracy drop ($\Delta$ Drop) after masking 7 frequency bands across 5 time windows (35 conditions in total). Red cells indicate high impairment (high importance for decoding), while green/white cells indicate low impact. Statistical significance is annotated with asterisks (* $p < 0.05$ uncorrected, ** $q < 0.05$ FDR corrected). Inhibitory gating (T1$\times$alpha, $\Delta=+0.185$) and top-down feedback (T2$\times$beta, $\Delta=+0.110$) dominate the decoding process, whereas all three gamma sub-bands remain near zero across all time windows, indicating they do not contribute meaningful task-relevant information.

**Chinese Caption:**
**图 2. 第二阶段：时频掩码热力图。** 展示了在 5 个时间窗内掩码 7 个频段（共 35 种条件）后的 Top-1 准确率下降量（$\Delta$ Drop）。红色区域表示严重的性能损伤（特征重要性高），绿色/白色区域表示影响较小。统计显著性通过星号标注（* $p < 0.05$ 原始显著，** $q < 0.05$ FDR 校正显著）。结果表明，抑制性门控（T1$\times$alpha, $\Delta=+0.185$）和自上而下的反馈机制（T2$\times$beta, $\Delta=+0.110$）主导了解码过程；相反，三个 Gamma 子频段在所有时间窗内均接近于零，表明它们未提供与任务相关的有效信息。

---

## Fig. 3. Phase 2: Top-10 Accuracy Drop
**English Caption:**
**Fig. 3. Phase 2: Top-10 Most Important Time-Frequency Features.** Displays the top 10 time-frequency conditions that caused the most severe accuracy drop when ablated. The red and blue bars represent the Top-1 and Top-5 accuracy drops, respectively. The T1$\times$alpha and T2$\times$beta combinations stand out as the only strictly significant features (** $q < 0.05$). This confirms that the model's visual decoding heavily relies on a highly specific spatiotemporal pattern—early alpha synchronization followed by delayed beta desynchronization—rather than broadband power shifts.

**Chinese Caption:**
**图 3. 第二阶段：排名前 10 的核心时频特征。** 展示了被消融后导致最严重准确率下降的前 10 个时频条件。红色和蓝色柱状分别代表 Top-1 和 Top-5 的准确率下降量。T1$\times$alpha 和 T2$\times$beta 组合作为仅有的严格显著特征（** $q < 0.05$）脱颖而出。这证实了模型在视觉解码时，严重依赖于高度特异性的时空模式（早期的 Alpha 同步与随后的 Beta 去同步），而非宽频带的能量偏移。

---

## Fig. 4. Phase 3A: Amplitude Scaling
**English Caption:**
**Fig. 4. Phase 3A: Amplitude Scaling Dose-Response Curve.** Shows the accuracy variations under different amplitude scaling factors ($\alpha: 0 \to 2.0$). A paradoxical effect is observed for T1$\times$alpha: as the scaling factor $\alpha$ approaches 0 (suppression), accuracy actually increases slightly, whereas artificial enhancement ($\alpha > 1$) degrades performance. This indicates that the early alpha band primarily acts as an inhibitory gate (a distractor or noise floor for this specific visual task), and suppressing its amplitude is functionally equivalent to denoising.

**Chinese Caption:**
**图 4. 第三阶段 A：振幅缩放剂量-反应曲线。** 展示了在不同振幅缩放因子（$\alpha: 0 \to 2.0$）下的准确率变化。对于 T1$\times$alpha 观察到了反常效应：当缩放因子 $\alpha$ 趋近于 0（抑制）时，准确率反而略有上升；而人为增强（$\alpha > 1$）则会导致性能下降。这表明，早期的 Alpha 频段主要充当抑制性门控（在该特定视觉任务中表现为干扰噪声），抑制其振幅在功能上等同于去噪过程。

---

## Fig. 5. Phase 3B: Phase Randomization
**English Caption:**
**Fig. 5. Phase 3B: Phase Randomization Impact.** Plots the Top-1 accuracy drop as the temporal phase structure is progressively destroyed (randomization ratio $0.0 \to 1.0$) while strictly preserving the power spectral density. The steep performance collapse at ratio 1.0 for T2$\times$beta reveals that the model does not merely rely on the gross energy of the beta band. Instead, it extracts fine-grained, phase-locked temporal dynamics, underscoring the necessity of precise temporal alignment for successful decoding.

**Chinese Caption:**
**图 5. 第三阶段 B：相位随机化影响曲线。** 描绘了在严格保持功率谱密度不变的前提下，随着时域相位结构被逐渐破坏（随机化比例 $0.0 \to 1.0$），Top-1 准确率的下降趋势。T2$\times$beta 在比例为 1.0 时出现的性能陡降表明，模型并非仅仅依赖 Beta 频段的宏观能量。相反，它提取了细粒度的、相位锁定的时间动态特征，突显了精准的时间对齐在成功解码中的必要性。

---

## Fig. 6. Phase 3C: Gaussian Noise Injection
**English Caption:**
**Fig. 6. Phase 3C: Gaussian Noise Robustness Test.** Illustrates the model's resilience to additive white Gaussian noise injected into specific time-frequency bins, mapped across varying Signal-to-Noise Ratios (SNR). The accuracy for T2$\times$beta remains relatively stable until the SNR drops below 0 dB, after which it deteriorates rapidly. This suggests that the learned representations are robust to moderate environmental or background noise, provided the core signal topology remains discernible.

**Chinese Caption:**
**图 6. 第三阶段 C：高斯噪声注入鲁棒性测试。** 展示了模型对注入到特定时频块中的加性高斯白噪声的抵抗能力，横轴为不同的信噪比（SNR）。T2$\times$beta 的准确率在 SNR 降至 0 dB 以下之前保持相对稳定，随后迅速恶化。这表明，只要核心信号拓扑结构保持可辨识，模型学习到的表征对中等程度的环境或背景噪声具有良好的鲁棒性。

---

## Fig. 7. Phase 3 Summary
**English Caption:**
**Fig. 7. Phase 3: Perturbation Types at Maximum Strength.** Provides a comparative summary of the Top-1 accuracy drop across three distinct perturbation strategies—Amplitude Scaling ($\alpha=0$), Phase Randomization (ratio=$1.0$), and Noise Injection (SNR=$-10$dB)—applied to key conditions. The consistently high impairment caused by phase randomization across both T1$\times$alpha and T1$\times$beta confirms that temporal phase structure encodes more critical discriminative information than pure amplitude modulation.

**Chinese Caption:**
**图 7. 第三阶段综合：最大强度下的扰动类型对比。** 对比总结了三种不同的扰动策略（振幅缩放 $\alpha=0$、相位随机化 ratio=$1.0$、噪声注入 SNR=$-10$dB）在关键条件下的 Top-1 准确率下降量。相位随机化在 T1$\times$alpha 和 T1$\times$beta 上引起的一致性严重损伤证实了，时域相位结构比纯粹的振幅调制编码了更多关键的判别性信息。
