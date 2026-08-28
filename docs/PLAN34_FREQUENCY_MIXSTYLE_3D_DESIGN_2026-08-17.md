# Plan 34 3-D Frequency MixStyle 设计

## 1. 状态与问题

- 状态：source-side robustness 设计与可运行模块；不是新的 UDA 性能结论。
- 目标：在 source 图像及其受控频率环境中随机化 style-like spectral statistics，使分类器不依赖某一固定频谱外观。
- 与 C4 的区别：C4 用 `T_adapt` 的全局 discrepancy 对所有样本做同方向衰减；Frequency MixStyle 不读取 target prior，不估计“target 应被修正到哪里”，而在 source 支持范围内制造多样的频率 style。

直接的 Fourier amplitude mixing 已有 image-level 和 feature-level 先例。这里的候选贡献不是“使用 FFT”本身，而是：在 3-D MRI hidden feature map 中，使用 class-conditional donor、径向 band-wise amplitude-statistic mixing、source frequency-environment style bank、phase preservation 和可审计 identity/fallback controls。

## 2. 固定边界

- 主训练只使用 source 标签。target 不提供 prior、频带排序、mix coefficient 或 model selection。
- source donor 仅来自同一 diagnosis class；若一个 class 在 batch 内没有第二个样本，该样本回退 identity。
- donor 可来自 source 的 `original`、`lowpass`、`downsample_resample`、`mild_blur` 单视图环境；这不是 paired-view consistency loss。
- 模块只在 `model.train()` 下激活；validation、source-test 和任何 target inference 都是 identity forward。
- 不使用 paired scan/loss、pseudo-label、domain adversary、负 discrepancy 或 amplitude amplification。

## 3. 模块

令选定空间层的 feature 为：

```text
Z_i in R^[C,D,H,W]
F_i = rFFTN(Z_i) = A_i exp(j P_i)
```

其中 `A_i` 为非负 amplitude，`P_i` 为 recipient phase。把 rFFT 的径向频率空间划分为 `B={low, mid, high}`。对每个 sample/channel/band，计算 amplitude statistics：

```text
mu_i,b,c = mean(A_i,c over band b)
sigma_i,b,c = std(A_i,c over band b)
```

对同标签 donor `j` 和 per-sample/per-band `lambda_i,b ~ Beta(alpha, alpha)`：

```text
mu'_i,b,c = lambda_i,b mu_i,b,c + (1-lambda_i,b) mu_j,b,c
sigma'_i,b,c = lambda_i,b sigma_i,b,c + (1-lambda_i,b) sigma_j,b,c
A'_i,b,c = max(0, (A_i,b,c - mu_i,b,c) / sigma_i,b,c * sigma'_i,b,c + mu'_i,b,c)
Z'_i = irFFTN(A'_i exp(j P_i))
```

phase 始终来自 recipient。`full_amplitude` 控制则直接混合完整 amplitude：

```text
A'_i = lambda_i A_i + (1-lambda_i) A_j
```

两种模式都保持输出实数、finite，且在 `p=0`、evaluation mode、没有合法 donor 或 `lambda=1` 时严格回退 identity。

## 4. 模型位置

默认架构保持 `layer5_pixel + original_capm`：

```text
image -> layer1 -> layer2 -> layer3 -> FrequencyMixStyle3D -> layer4 -> layer5
      -> original CAPM(age, sex, education) -> classifier
```

实际插入层由全局图谱的稳定性选择；为避免在贴近 classifier 的位置直接重写 task feature，首个实现默认 `layer3`。`layer2`、`layer3`、`layer4` 是可配置候选，但不能在已见 target 结果后选择。

## 5. 最小对照矩阵

历史 MixStyle 结果不可直接复用，所有对照必须在当前 scan-filtered manifest、source split、optimizer、budget、selector 和 source frequency environments 下重跑：

| ID | 变体 | 用途 |
|---|---|---|
| M0 | `original_capm` | source CAPM clean baseline |
| M1 | `mixstyle` | 当前协议下的空间 MixStyle 对照 |
| M2 | `frequency_mixstyle_full` | class-conditional complete-amplitude mixing，FACT-like frequency control |
| M3 | `frequency_mixstyle_bandwise` | class-conditional 3-D band-wise amplitude-statistic MixStyle |

M2 与 M3 都使用同样的 donor policy、mix probability、Beta alpha、插入层、source data budget 和 environment augmenter。M3 不能仅靠比 M0 高一次 BA 宣称频率机制；它必须超过 M1/M2 的 source-side 机制门。

## 6. source-only 机制门

在触及任何新的 target test 之前，每个 M2/M3 checkpoint 必须满足：

1. clean `S_val` BA/CE 不劣于 M0 的预注册容差；
2. 在四个 source frequency environments 上报告 per-environment CE/BA 和 worst-environment risk；
3. feature relative RMS、logit JS 和 amplitude relative delta 证明干预并非 identity；
4. `applied_fraction`、合法 donor fraction、identity fallback fraction、每 band lambda 和 amplitude delta 可复算；
5. 同类 donor 不足时确实 fallback，不以跨 diagnosis donor 填充；
6. full-amplitude 与 band-wise 的差异只来自预注册的 mixing mode，而不是不同训练预算。

任一失败均是 **NO-GO for target confirmation**。此阶段的可用主张仅为“在 source 合成频率环境中评估的 robustness candidate”。

## 7. 目标确认与论文边界

当前 C0-C4 的 `T_test` 已被读取，不能成为新 M2/M3 的确认性 target test。只有获得从未参与本设计的新 NACC holdout 或新队列后，才冻结一个通过 source-only 门的候选并做一次 subject-disjoint final evaluation。

即使新的候选得到提升，允许的表述也只是固定 protocol 下的跨队列泛化潜力；不能称 field-strength causal correction、普遍 scanner harmonization 或 zero-shot domain generalization。

## 8. 运行配置

模块由 `Model/ablation/frequency_mixstyle.py` 提供，journal variants 为 `frequency_mixstyle_full` 与 `frequency_mixstyle_bandwise`。`frequency_mixstyle` config 必填或可显式设置：

```yaml
frequency_mixstyle:
  mix_stage: layer3
  probability: 0.5
  alpha: 0.1
  band_edges: [0.0, 0.15, 0.35]
  class_conditional: true
  use_source_frequency_environments: true
  lowpass_kernel: 3
  blur_sigma: 0.8
```

先使用 `experiments/train_journal.py --source-only` 运行 M0--M3 的 source-side screening。真实数据、冻结 manifest、checkpoint 和完整 output provenance 未在本仓库提交；合成测试通过不等于真实训练或泛化证据。
