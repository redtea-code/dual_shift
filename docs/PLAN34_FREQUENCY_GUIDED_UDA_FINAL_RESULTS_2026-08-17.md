# Frequency-Guided UDA：ADNI 1.5T → NACC 3T 最终结果（MCI vs AD）

- 更新：2026-08-17 UTC+8
- 状态：主方向预注册矩阵已完成；`C0–C4 × seeds 43/44` 共 10/10 个训练、冻结 checkpoint、source-test、target-test 与配对比较产物均通过审计。
- 主方向：ADNI 1.5T（source）→ NACC 3T（target）。这是唯一的主方向；本报告不包含、也不以 NACC→ADNI 结果支持任何结论。
- 数据协议：`scan_filtered_v1_2026-08-08`，MCI（0）vs AD（1），输入为 `160×196×160`。
- Backbone / route：ResNet10，`layers=[1,1,1,1]`；`layer5_pixel + original_capm`，表变量为 age、sex、education。
- 训练 seeds：43、44；source split seed 固定为 42。所有数字为 subject-mean 聚合；目标集全部为每个 seed 固定的 263-subject `T_test`。

## 1. 结论摘要

本次结果**不支持**“完整目标频谱引导（C4）在两个复现实验中稳定优于源初始化 CAPM 基线（C0）”这一主张：

- C4 相比 C0 的 target BA：seed43 为 `+0.023`（95% paired bootstrap CI `[-0.011, +0.061]`），seed44 为 `+0.019`（`[-0.027, +0.062]`）。两者的 CI 均跨 0。
- 虽然 C4 在两 seed 的点估计都高于 C0，均值 BA 为 `0.693`，高于 C0 的 `0.670`，但仅两 seed，且配对不确定性未支持 seed 内稳定差异；不能将均值差 `+0.023` 表述为可靠增益。
- 预注册的全部升级条件未满足：C4 在 seed43 落后 C1（环境 GroupDRO）`−0.027`，也落后 C2（uniform gate）`−0.030`；因此不能称 target-specific frequency ordering 必要，亦不能进入下一机制（如 SoftRegion）开发。
- 更严格的解释是：频率环境、门控容量和真实 band ordering 的效果均存在明显 seed 依赖。seed43 中 C2（uniform gate）最佳；seed44 中 C4 最佳。该矩阵是一个合规的**阴性/不确定 UDA 实验**，而非方法优效的确证。

## 2. 预注册问题、数据隔离与训练契约

目标是判断一个只读取未标注 `T_adapt` 图像与协变量的目标频谱先验，能否在跨队列 ADNI 1.5T→NACC 3T 的 MCI-vs-AD 任务上超越 source-selected `original_capm`。

每个 seed 在拟合之前划分为：

| 集合 | 用途 | 是否可读取标签 | 是否影响模型选择 / 配置 |
|---|---|---:|---:|
| `S_train` / `S_val` / `S_test` | source 训练、checkpoint 选择与 source 报告 | 是 | 仅 source；checkpoint 由 source validation 选择 |
| `T_adapt` | 建立 target feature-spectrum prior | 否 | 仅无标签频率先验 |
| `T_test` | 最终冻结 target 评估 | 仅评估时 | 否 |

`T_adapt` / `T_test` 按 subject ID 划分，`adaptation_fraction=0.5`，不按诊断分层；频率先验对每个 subject 仅保留 earliest visit（同日按 folder/path 打破平局）。两个 seed 的 `T_adapt` 均为 264 subjects，`T_test` 均为 263 subjects；审计确认二者 subject-disjoint。频率先验 JSON 均记录 `target_labels_read=false`、`target_metrics_read=false`。

所有 10 个最终 checkpoint 都用 source-validation selection 加 collapse guard 选择，并全部 eligible；每个最终产物进一步验证 prior、source split、target split 的 SHA-256 及 target subject digests 一致。因此 target 标签、target 指标和 target 预测均未参与 checkpoint、variant、preset、band edge 或强度选择。

## 3. 锁定的 C0–C4 比较矩阵

| ID | 代码变体 | 固定差异 | 作用 |
|---|---|---|---|
| C0 | `frequency_uda_baseline` | 无频率环境、无 gate | 同一 source CAPM 初始化的主基线 |
| C1 | `frequency_uda_env_dro` | 四种频率环境 + GroupDRO，无 gate | 分离环境鲁棒训练的效应 |
| C2 | `frequency_uda_uniform` | gate + 等权 band discrepancy | 控制 gate 容量 |
| C3 | `frequency_uda_permuted` | gate + 非恒等置换的真实 discrepancy | 检验 band ordering |
| C4 | `frequency_uda` | gate + 真实 target-spectrum discrepancy | 完整 target-spectrum-guided UDA |

C4 的 gate 在 layer4 与 layer5 之间施加有界、残差的 3D rFFT amplitude attenuation；相位保持不变，gate 不接收 domain ID。每个 source 样本恰好取一种频率环境：`original`、`lowpass`、`downsample_resample` 或 `mild_blur`；无 pseudo-label、无 target metric selection、无对抗域分类器。

## 4. Target test：完整逐 seed 结果

以下均为最终 `T_test` 的 subject-mean 指标。BA 方括号为 200 次 subject bootstrap 95% CI。Brier 与 ECE 越低越好；其余越高越好。

### seed 43（T_test = 263 subjects）

| 变体 | 冻结 epoch | BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 baseline | 2 | 0.640 [0.587, 0.689] | 0.783 | 0.341 | 0.938 | 0.649 | 0.174 | 0.047 |
| C1 env_dro | 14 | 0.723 [0.660, 0.778] | 0.833 | 0.529 | 0.916 | 0.738 | 0.154 | 0.082 |
| C2 uniform | 38 | **0.765 [0.720, 0.810]** | **0.837** | **0.682** | 0.848 | **0.765** | 0.159 | 0.103 |
| C3 permuted | 18 | 0.657 [0.602, 0.710] | **0.837** | 0.365 | 0.949 | 0.669 | 0.177 | 0.131 |
| C4 full UDA | 26 | 0.657 [0.604, 0.710] | 0.831 | 0.353 | **0.961** | 0.669 | 0.191 | 0.162 |

### seed 44（T_test = 263 subjects）

| 变体 | 冻结 epoch | BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 baseline | 18 | 0.700 [0.642, 0.759] | 0.786 | 0.515 | 0.884 | 0.708 | 0.204 | 0.143 |
| C1 env_dro | 42 | 0.633 [0.590, 0.682] | 0.789 | 0.303 | 0.963 | 0.626 | 0.254 | 0.243 |
| C2 uniform | 39 | 0.580 [0.545, 0.615] | 0.797 | 0.172 | **0.988** | 0.541 | 0.274 | 0.265 |
| C3 permuted | 31 | 0.714 [0.670, 0.769] | 0.810 | 0.525 | 0.902 | 0.724 | 0.184 | 0.128 |
| C4 full UDA | 37 | **0.729 [0.682, 0.786]** | **0.818** | **0.586** | 0.872 | **0.737** | **0.178** | **0.115** |

## 5. Target test：两个 seed 的描述性汇总

均值与 SD 跨 seed 43/44 计算，仅用于描述复现实验的中心与波动；它不是跨 seed 的显著性检验。

| 变体 | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 baseline | 0.670 ± 0.042 | 0.784 ± 0.002 | 0.428 ± 0.123 | 0.911 ± 0.038 | 0.678 ± 0.042 | 0.189 ± 0.021 | **0.095 ± 0.068** |
| C1 env_dro | 0.678 ± 0.063 | 0.811 ± 0.031 | 0.416 ± 0.160 | **0.940 ± 0.034** | 0.682 ± 0.079 | 0.204 ± 0.071 | 0.162 ± 0.114 |
| C2 uniform | 0.673 ± 0.131 | 0.817 ± 0.028 | 0.427 ± 0.361 | 0.918 ± 0.099 | 0.653 ± 0.159 | 0.217 ± 0.081 | 0.184 ± 0.114 |
| C3 permuted | 0.685 ± 0.040 | **0.824 ± 0.018** | 0.445 ± 0.114 | 0.926 ± 0.033 | 0.696 ± 0.038 | 0.180 ± 0.005 | 0.129 ± 0.002 |
| C4 full UDA | **0.693 ± 0.051** | **0.825 ± 0.009** | **0.469 ± 0.165** | 0.916 ± 0.063 | **0.703 ± 0.048** | **0.184 ± 0.009** | 0.138 ± 0.033 |

C4 的描述性均值相对 C0 为：BA `+0.023`、AUROC `+0.040`、sensitivity `+0.041`、specificity `+0.005`、macro-F1 `+0.025`、Brier `−0.004`，但 ECE `+0.043`（更差）。这说明即使按均值观察，C4 的改善也并非所有质量维度一致，尤其不能声称校准改善。

## 6. 预注册配对对比

同一 seed 内，每个比较严格配对同一批 263 个 `T_test` subjects；200 次 subject-row bootstrap。这里“BA”使用 `accuracy` 字段（在该固定 binary target 集与报告 BA 的数值相同）；正值代表前者更优。CI 跨 0 表示在此 bootstrap 评估下差异不稳定。

### C4 的主对比与机制分解

| 对比 | seed43 ΔBA [95% CI] | seed44 ΔBA [95% CI] | 判定 |
|---|---:|---:|---|
| C4 − C0（主对比） | +0.023 [−0.011, +0.061] | +0.019 [−0.027, +0.062] | 两 seed 点估计为正，但均不稳定 |
| C4 − C1（超过环境训练的 gate） | −0.027 [−0.061, +0.008] | +0.054 [+0.007, +0.100] | 方向相反；C4 不满足“每 seed 不差于 C1” |
| C4 − C2（target-specific vs uniform） | −0.030 [−0.084, +0.015] | +0.088 [+0.031, +0.146] | 方向相反；不支持稳定 target-specific weighting 效应 |
| C4 − C3（真实 vs 置换 ordering） | +0.008 [−0.019, +0.031] | +0.008 [−0.039, +0.046] | 两 seed 均接近 0；没有 ordering 必要性的证据 |

C4−C0 的 AUROC contrast 为 seed43 `+0.052 [+0.010, +0.093]`、seed44 `+0.032 [−0.010, +0.076]`。这比 BA 更乐观，但仍没有满足两个 seed 的稳定确认要求，且不应替代预注册 primary metric BA。

### 相对 C0 的其他控制组结果（ΔBA）

| 变体 − C0 | seed43 [95% CI] | seed44 [95% CI] |
|---|---:|---:|
| C1 env_dro − C0 | +0.050 [+0.015, +0.088] | −0.035 [−0.081, +0.008] |
| C2 uniform − C0 | +0.053 [−0.004, +0.107] | −0.069 [−0.123, −0.015] |
| C3 permuted − C0 | +0.015 [−0.023, +0.054] | +0.012 [−0.031, +0.046] |
| C4 full UDA − C0 | +0.023 [−0.011, +0.061] | +0.019 [−0.027, +0.062] |

这些控制表明，seed43 的高分主要由 C1/C2 路线取得，而 seed44 的高分由 C3/C4 路线取得；没有某一固定处理跨两个 seed 给出稳定 BA 优势。

## 7. Source-test 结果（未用于 target 选择）

Source-test 是 source 侧固定 test split（49 subjects），仅作为训练后的同域诊断。checkpoint 实际选择仅来自 source validation，且所有 checkpoint 均通过 collapse guard。

| 变体 | seed43 source-test BA / AUROC / Macro-F1 | seed44 source-test BA / AUROC / Macro-F1 |
|---|---:|---:|
| C0 | 0.660 / 0.771 / 0.657 | **0.769 / 0.826 / 0.756** |
| C1 | 0.661 / 0.830 / 0.649 | 0.691 / 0.809 / 0.685 |
| C2 | **0.756 / 0.826 / 0.708** | 0.688 / 0.818 / 0.705 |
| C3 | 0.720 / 0.818 / **0.738** | 0.690 / **0.852** / 0.693 |
| C4 | 0.720 / **0.824** / **0.738** | 0.754 / 0.831 / 0.737 |

同域 source-test 与 target-test 的排序并不一致，例如 seed43 的 C2 在 source-test 和 target-test 都较高，而 seed44 C2 在 target 上显著退化。因而 source 分数不能代替真实 target transfer 的判断。

## 8. 频率先验与可审计性

频率先验均由 frozen `original_capm` 的 pre-CAPM layer4 feature maps 提取，使用 `S_train`（144 subjects）和无标签 `T_adapt`（264 subjects）。频段为 low `[0,0.15)`、mid `[0.15,0.35)`、high `[0.35,∞)`；high band 定义为 rFFT 全域分区，故不同于 raw-image descriptive audit 的截断区间。

| seed | source low / mid / high fraction | T_adapt low / mid / high fraction | normalized discrepancy（low / mid / high） |
|---|---:|---:|---:|
| 43 | 0.465 / 0.266 / 0.269 | 0.464 / 0.260 / 0.276 | 0.163 / 0.925 / 1.000 |
| 44 | 0.462 / 0.252 / 0.286 | 0.463 / 0.248 / 0.289 | 0.269 / 1.000 / 0.790 |

两个 seed 的 discrepancy ordering 不一致（seed43：high > mid > low；seed44：mid > high > low）。这可解释 C4 及 C3 的 seed-dependent 行为是值得进一步研究的现象，但不能事后将其用作新的模型选择依据。特别是 C4−C3 均接近 0，也不支持当前 real band ordering 在此规模上产生可辨别的特异收益。

## 9. 预注册 GO / NO-GO 核对

| 条件 | 观察结果 | 是否满足 |
|---|---|---:|
| C4 在两 seed 均优于 C0 | ΔBA：+0.023、+0.019；但两个 paired CI 均跨 0 | 否（点估计正，但无稳定支持） |
| C4 不差于 C1 与 C2，且对二者均值为正 | seed43 C4−C1 = −0.027；C4−C2 = −0.030 | 否 |
| source validation collapse guard 通过 | 10/10 checkpoints eligible | 是 |
| prior/checkpoint/split hash、subject separation 与 target label blinding 均合规 | 10/10 审计通过；`T_adapt ∩ T_test = ∅`；prior 均记录 labels/metrics 未读取 | 是 |

**总判定：NO-GO。** 保留该实验为合规的 negative / inconclusive UDA result；不要基于本结果引入 SoftRegion 或第二个新机制。下一轮应优先解决 seed 间频率先验排序与性能反转的原因，且所有新假设须在读取新的 `T_test` 结果前锁定。

## 10. 局限与下一步

1. 仅两个训练 seed，且 source split 固定为 seed 42；均值差不能取代重复实验或 hierarchical / cross-seed 推断。
2. 该实验只评估 ADNI 1.5T→NACC 3T。其结果不能推断场强因果效应，因为 cohort、site、acquisition、预处理与生物学差异仍然混杂。
3. UDA 使用无标签 target images，故结论仅适用于 subject-disjoint、unlabeled-target-adaptation setting，不是 zero-shot domain generalization。
4. 200 次 bootstrap 可量化当前固定 target sample 的条件不确定性，但不会量化训练 seed 的总体不确定性。
5. 随后的工作应预先固定：更多训练 seeds、仅在 source side / `T_adapt` 上确定的先验稳定性诊断，以及重新锁定的 C4 对照矩阵；不得用本次 target 排名去选择 band edge、strength、layer 或新模块。

## 11. 产物与复现来源

- 预注册计划：`docs/FREQUENCY_UDA_EXPERIMENT_PLAN.md`
- 主配置模板：`journal_frequency_uda_scan_filtered_1p5t_mci_ad.yaml`
- resolved configs：`resolved_frequency_uda/final_v2/adni_to_nacc_seed{43,44}.yaml`
- source CAPM preparation、target split 与 prior：`outputs/frequency_uda_prepare/ADNI_to_NACC/seed_{43,44}/`
- 最终矩阵：`outputs/frequency_uda_final_v2/ADNI_to_NACC/seed_{43,44}/{frequency_uda_baseline,frequency_uda_env_dro,frequency_uda_uniform,frequency_uda_permuted,frequency_uda}/journal_metrics.json`
- subject-level target predictions：同目录下各变体的 `target_predictions.csv`
- 完整 paired bootstrap comparisons：`paired_comparisons.json`（位于 `outputs/frequency_uda_final_v2/ADNI_to_NACC/seed_{43,44}/`）
- 训练实现：`experiments/train_journal.py`；频率先验构建：`experiments/build_frequency_uda_prior.py`；频率环境：`training/frequency_environments.py`。
