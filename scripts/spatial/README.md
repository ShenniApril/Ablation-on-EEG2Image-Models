## 预处理
``` powershell
python preprocess_eeg.py `
  --sub_id 1 `
  --raw_data_dir "data\things_eeg\raw_eeg" `
  --output_dir "data\things_eeg\preprocessed_eeg_no_mvnn" `
  --rfreq 250 `
  --precision fp32
```
--sub_id: 1-10，分别对应10个被试。
--rfreq: 采样率，单位Hz。
--precision: 输出数据精度。



## 聚类
1. 从无 MVNN 的训练 EEG 中随机选取 trial。
2. 每个 trial 提取 GFP 局部峰值。
3. 每个 trial 最多保留若干个最强峰。
4. 使用 polarity-invariant modified K-means 聚类。
5. 每个 K 值进行若干次随机初始化。
6. 将模板 back-fit 到完整 test 和 train EEG。
7. 对短于若干 ms 的 microstate segment 进行平滑。

```powershell
python microstate_clustering_reference.py `
  --fit-eeg "data\things_eeg\preprocessed_eeg_no_mvnn\sub-01\train.npy" `
  --apply-eeg "data\things_eeg\preprocessed_eeg_no_mvnn\sub-01\test.npy" `
  --output-dir "results\microstate_sub01_no_mvnn_k4" `
  --n-states 4 `
  --max-fit-trials 5000 `
  --max-peaks-per-trial 12 `
  --min-peak-distance-ms 20 `
  --min-segment-ms 20 `
  --n-init 20 `
  --backfit-batch-size 256 `
  --backfit-train `
  --random-state 2025
```

### `--max-peaks-per-trial 12`

每个1秒 EEG trial 最多保留12个最强的 GFP 局部峰值。

1. 计算每个时间点的 GFP。
2. 找出所有局部峰值。
3. 按 GFP 大小从强到弱排序。
4. 最多选择前12个峰对应的头皮电压拓扑图参与聚类。

---

### `--min-peak-distance-ms 20`

同一个 trial 中，两个被检测到的 GFP 局部峰之间至少相隔20 ms。

---

### `--min-segment-ms 20`

模板 back-fitting 到完整 EEG 后，持续时间短于20 ms的 microstate segment 会被重新标记为相邻状态。

### 模板聚类结果评价

(neurobridge) PS D:\shenni\交大\论文\EEG2Image\NeuroBridge> python -c "import numpy as np; p=r'results/k4_t10000_peak10_seg10_init50_seed2025/test_microstate_features.npz'; f=np.load(p); c=f['coverage'].reshape(-1,4); print('mean coverage:',c.mean(0)); print('median:',np.median(c,0)); print('trial presence:',(c>0).mean(0)); print('sum:',c.mean(0).sum())"

### 绘图
(neurobridge) PS D:\shenni\交大\论文\EEG2Image\NeuroBridge> python plot_microstate_topographies.py ` 
   --templates "results\microstate_parameter_search_k4\k4_trials5000_peaks12_distance10ms_segment10ms_init50_seed2025\microstate_templates.npy" `         
   --info "data\things_eeg\preprocessed_eeg_no_mvnn\info.json" `                                   
   --output "results\microstate_sub08_no_mvnn_k4_new\microstate_topographies.png"

## 消融

### 1. 完整预处理
1. 使用固定K=4模板back-fitting
``` powershell
python -c "from pathlib import Path; import numpy as np; from microstate_clustering_reference import backfit_and_save_dataset; out=Path(r'results_sub01/backfit_sub08_templates_k4'); out.mkdir(parents=True, exist_ok=True); templates=np.load(r'results_sub08/microstate_sub08_no_mvnn_k4/microstate_templates.npy'); print(backfit_and_save_dataset(Path(r'data/things_eeg/preprocessed_eeg_no_mvnn/sub-01/test.npy'), out, 'test', templates, 4, 250.0, 20.0, 256))"
```
2. 得到每个时间点的microstate标签
3. 扰动目标状态对应的连续EEG片段
4. 使用训练集得到的固定MVNN矩阵进行白化
``` powershell
python export_mvnn_matrices.py `
  --subject 1 `
  --raw-data-dir "data\things_eeg\raw_eeg" `
  --pre-mvnn-info "data\things_eeg\preprocessed_eeg_no_mvnn\info.json" `
  --output-dir "results_sub01\mvnn_matrices_sub01" `
  --sessions 4 `
  --rfreq 250
```

### 2. 分别消融4个microstate模板

完整替换目标状态出现位置的空间拓扑，以state1为例：

```powershell
python microstate_ablation_reference.py `
  --condition topography `
  --state 0 `
  --control none `
  --output-dir "results\microstate_ablation_sub08"
```

### 3. 相对随机替换造成的下降

以State 1为例：

```powershell
python microstate_ablation_reference.py `
  --condition topography `
  --state 0 `
  --control random-segment `
  --random-state 2025 `
  --output-dir "results\microstate_ablation_sub08"
```

- 保持修改采样点数；
- 保持每个 trial 的目标 mask 结构；
- 优先寻找与原目标位置完全不重叠的循环平移；
- 如果目标 STATE coverage 太高而无法完全避开，则选择重叠最少的位置；
- 在新位置根据当地原 STATE 随机选择非当前模板。

由此确定哪个microstate模板对模型最重要。

### 4. 绘制

python plot_state_ablation.py

## 分析重要STATE的特征

### 1. Duration/Coverage缩短
1. 找到每个连续State 2 episode；
2. 找到它后面的microstate；
3. 将State 2末端25%的拓扑替换为后续状态模板；
4. 保留至少20 ms的State 2片段；
5. 排除没有following state的trial末端episode；
6. 保持修改时间点的GFP不变。
```powershell
$ratios = @(0.1, 0.25, 0.5, 0.75)
$seeds = @(2025, 2026, 2027, 2028, 2029)

foreach ($ratio in $ratios) {
    foreach ($seed in $seeds) {
        python microstate_ablation_reference.py `
          --condition duration `
          --state 1 `
          --ratio $ratio `
          --control random-segment `
          --random-state $seed `
          --output-dir "results\microstate_ablation_sub08"
    }
}
```
### 2. Occurrence removal

随机移除25%的完整State 2 episodes：

```powershell
python microstate_ablation_reference.py `
  --condition occurrence `
  --state 1 `
  --ratio 0.25 `
  --control none `
  --random-state 2025 `
  --output-dir "results\microstate_ablation_sub08"
```

### 3. 绘制
python plot_state4_feature_ablation.py