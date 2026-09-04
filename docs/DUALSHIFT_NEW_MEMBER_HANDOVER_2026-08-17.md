# DualShift 项目新人交接说明

更新时间：2026-08-29
截至证据快照：`origin/main`，提交 `98032fa`
当前计划分支：`codex/frequency-uda-plan`，提交 `fcd7a11`

本文只记录截至 2026-08-29 已注册的实验计划、已获得的结果，以及这些结果能够支持的结论和启示。不包含 DS-041 之后提出的新方法方案，不把未执行的设想写成项目结论。结果优先引用 `main` 中的版本化报告；实验分支上的代码和 interim 文件不自动等同于主线证据。

## 0. 项目目标和证据边界

当前研究问题是：在 scan-filtered MCI vs AD 任务中，考察 CAPM 条件化、频域增强和无标签域适应是否能够改善 ADNI 1.5T 与 NACC 3T 之间的跨队列泛化。

当前不能将 cohort、site、manufacturer、sequence 和 field strength 的混杂解释为 scanner 或 field-strength 的独立因果效应。任何性能结果都必须限定在具体方向、数据划分、随机种子、预处理和模型配置下。

所有 target 结果均需要区分：

- 代码和协议是否可运行；
- adaptation 是否遵守 target-label-blind 边界；
- target 指标是否只是 internal frozen `T_test` exploratory 结果；
- 是否有足够的跨 seed、机制和 provenance 证据支撑方法主张。

## 1. 固定数据契约

| 集合 | 用途 | 允许访问的信息 |
|---|---|---|
| `S_train` | source 训练、source 统计或 source projector | source image、source label、允许的 source covariates |
| `S_val` | checkpoint/epoch selector | source image、source label；不得用 target metric 选择 |
| `S_test` | checkpoint 固定后的 source 诊断 | 仅用于训练后报告 |
| `T_adapt` | 无监督适配或 target summary | target image；不得访问 label、covariate、environment、prediction 或 metric |
| `T_test` | 最终 target 评估 | checkpoint 固定后读取；当前为历史 internal frozen holdout，属于 exploratory |

主方向是 `ADNI_to_NACC`。`NACC_to_ADNI` 是单独的方向性压力测试，不能与主方向平均后宣称双向有效。

target-label-blind 只适用于 adaptation/statistics/selection 阶段。最终 exploratory target report 可以读取 `T_test` label，但不能反过来影响 checkpoint、strength、variant 或协议。

## 2. 截至目前的实验计划登记

权威索引：[`docs/EXPERIMENT_INDEX.md`](https://github.com/redtea-code/dual_shift/blob/main/docs/EXPERIMENT_INDEX.md)。

| DS-ID | 实验计划 | 计划状态/结果状态 | 核心问题 |
|---|---|---|---|
| DS-034 | Scan-filtered CAPM / IE-CAPM / APIC v3_2 | RUNNING | feature-frequency 与 CAPM 主线是否有可复现跨队列信号 |
| DS-035 | FMM raw-image UDA baseline | COMPLETED | 完整 FMM 是否优于匹配的 source-only reference |
| DS-036 | Target-style transport + CAPM UDA probe | COMPLETED | target-style CAPM transport 是否改善 target 泛化 |
| DS-037 | Amplitude transport mechanism audit | COMPLETED | Fourier amplitude/phase transport 的 strength 和 phase 是否有稳定收益 |
| DS-038 | Domain/intensity GRL factorial audit | BLOCKED | domain GRL、intensity GRL 是否各自有效或互补 |
| DS-039 | CAPM-conditioned residual adaptation pilot | RUNNING / PILOT | CAPM 后的 bounded residual correction 是否能减少 batch-sensitive variation |
| DS-040 | CAPM-conditioned frequency GRL | PROPOSED（result record 已完成 screening） | raw-Fourier enhancement 与 full/residual GRL placement 是否安全、有效 |
| DS-041 | CAPM-conditioned source-free residual distribution alignment | COMPLETED STAGE B / EXPLORATORY | source-free 的 CAPM residual distribution alignment 是否改善 target 域 |

DS-039 至 DS-041 的共同设计前提是：CAPM 使用 `[age, sex, education]` 作为条件化路径，不能把 CAPM 自动解释为纯生物学或因果表示；残差或域适应分支只能作为待验证的功能分工。

### 2.1 当前残差路线中的 `P_task`

DS-039/041 中的 `P_task` 是 source-only 的任务支持子空间投影，不是由 target 标签或 target 预测拟合的投影。实现流程为：先得到冻结 CAPM 特征 `B4 = CAPM(F4, z)`，对每个 source subject 做 global average pooling，并使用冻结 classifier 的 diagnosis CE 梯度构造特征协方差；再取协方差前 `rank=32` 个特征方向组成矩阵 `U`，定义 `P_task = U U^T`；残差为 `R4 = (I - P_task) B4`。因此，`P_task` 只能被解释为 source classifier 下的 task-support proxy，`R4` 也不能自动被解释为纯 batch 或 scanner 信息。

## 3. 已有实验结果

### 3.1 DS-034：CAPM 与 feature-frequency 主线

结果记录：[`docs/results/DS-034/`](https://github.com/redtea-code/dual_shift/tree/main/docs/results/DS-034)。

- C4 相对 C0 的历史 target BA 正点估计在 seed43/44 约为 `+0.023` 和 `+0.019`，两 seed 的点估计方向一致；
- 现有不确定性区间跨过零，且 C4 与 C3 的差异约为 `+0.008`，frequency prior 的独立增量尚未完全分离；
- raw-volume FFT、spatial feature-map FFT 和 GAP/embedding 统计不是同一个分析对象，不能混用；
- 当前只能说 C4 有值得继续审计的 exploratory point estimate，不能称为稳定 frequency-domain adaptation 或 scanner-causal correction。

DS-034 仍是主线机制审计记录，不是已闭合的最终方法结论。

### 3.2 DS-035：FMM raw-image UDA baseline

结果记录：[`docs/results/DS-035/3SEED_ABLATION_2026-08-19.md`](https://github.com/redtea-code/dual_shift/blob/main/docs/results/DS-035/3SEED_ABLATION_2026-08-19.md)。方向为 ADNI 1.5T -> NACC 3T，3 seeds，15/15 cells 完成。

| Variant | Target BA mean ± SD | 相对 B0 |
|---|---:|---:|
| B0 control | 0.5827 ± 0.0456 | — |
| B1 full FMM | 0.6131 ± 0.0166 | +0.0305 |
| B1a no source FFT | 0.6187 ± 0.0681 | +0.0360 |
| B1b no attention | 0.6285 ± 0.0759 | +0.0458 |
| B1c no GRL | 0.5726 ± 0.0114 | -0.0101 |

结论：完整 FMM 相对匹配 reference 有方向性 target BA 增益，值得作为 baseline 保留；但 B1a/B1b 方差较大，B1c 同时关闭两个 GRL head，不能将差异归因于 domain 或 intensity 单个 head。该结果不支持 scanner 因果或单个 GRL 已被证明是增益来源。

### 3.3 DS-036：target-style CAPM UDA

结果记录：[`docs/results/DS-036/5SEED_COMPARISON_2026-08-19.md`](https://github.com/redtea-code/dual_shift/blob/main/docs/results/DS-036/5SEED_COMPARISON_2026-08-19.md)。ADNI 1.5T -> NACC 3T，5 seeds，10/10 paired runs 完成。

- CAPM control target BA：`0.6663 ± 0.0067`；
- target-style CAPM target BA：`0.6415 ± 0.0214`；
- 平均 BA 差值：`-0.0248`；
- AUC 从 `0.7473 ± 0.0170` 降至 `0.7185 ± 0.0124`；
- 5/5 seeds 的 target BA 和 AUC 差值均为负。

结论：当前 strength、loss 和冻结 selector 下，target-style transport 为 NO-GO。它不证明所有 Fourier transport 都无效，也不提供 scanner/field-strength 因果解释；source-side 改善不能替代 target 主终点。

### 3.4 DS-037：amplitude/phase transport

结果记录：[`docs/results/DS-037/README.md`](https://github.com/redtea-code/dual_shift/blob/main/docs/results/DS-037/README.md)。原计划 5 seeds，批准的执行修订为 42/43/44 三 seeds，18/18 cells 完成。

| Variant | Strength | Phase | Target BA mean ± SD | 相对 AT0 |
|---|---:|---|---:|---:|
| AT0 CAPM | 0.00 | none | 0.6488 ± 0.0391 | — |
| AT1 | 0.25 | source | 0.6184 ± 0.0431 | -0.0304 |
| AT2 | 0.50 | source | 0.6227 ± 0.0038 | -0.0261 |
| AT3 | 1.00 | source | 0.5925 ± 0.0283 | -0.0563 |
| AT4 | 0.50 | target | 0.6419 ± 0.0341 | -0.0069 |
| AT5 | 1.00 | target | 0.6283 ± 0.0426 | -0.0205 |

没有 transport variant 在三 seeds 上一致为正；AT3 在三 seeds 上均为负。结论：当前 strength/phase 选择 NO-GO，不能关闭所有频域方向，也不能把单个 AUC 或 seed 的提升写成稳定机制。

### 3.5 DS-038：domain/intensity GRL factorial

结果记录：[`docs/results/DS-038/README.md`](https://github.com/redtea-code/dual_shift/tree/main/docs/results/DS-038)。历史 G0-G3 × seeds 42-46 的 20/20 性能单元完成，但注册的机制诊断没有完整持久化，因此主线状态为 BLOCKED。

| Variant | Target BA mean ± SD | 相对 G0 |
|---|---:|---:|
| G0 no GRL | 0.5697 ± 0.0766 | — |
| G1 domain only | 0.6038 ± 0.0251 | +0.0341 |
| G2 intensity only | 0.5801 ± 0.0371 | +0.0104 |
| G3 both GRL | 0.6308 ± 0.0618 | +0.0611 |

paired target BA 并非所有 seeds 都为正：G1 为 4/5，G2 为 3/5，G3 为 4/5。因此不能声称单个 head 稳定有效，也不能声称两个 head 存在互补性。缺少 checkpoint-bound 的 discriminator/gradient/discrepancy 诊断时，性能矩阵只能作为 screening evidence。

### 3.6 DS-039：CAPM-conditioned residual adaptation

结果记录：[`docs/results/DS-039/README.md`](https://github.com/redtea-code/dual_shift/blob/main/docs/results/DS-039/README.md)。模型为 frozen `original_capm`、`layer4_pixel`，使用 source 统计和 subject-disjoint、image-only `T_adapt` 的 channel-wise bounded correction，`max_strength=0.25`。

| Direction | RA0 BA | RA1 BA | Δ BA | RA0 AUROC | RA1 AUROC | Δ AUROC |
|---|---:|---:|---:|---:|---:|---:|
| ADNI→NACC mean | 0.5844 | 0.5948 | +0.0104 | 0.7672 | 0.7681 | +0.0009 |
| NACC→ADNI mean | 0.5784 | 0.5848 | +0.0064 | 0.6912 | 0.6914 | +0.0002 |

ADNI→NACC 的两个 seed 均有小幅正差，但 NACC→ADNI 的平均增益主要由 seed42 驱动，seed43 基本不变。source-side 也存在 sensitivity 上升、specificity 下降的 operating-point trade-off。correction finite 且 bounded，但已有 artifact 没有完整的 paired before/after feature discrepancy 和明确 CAPM-anchor drift tolerance。

结论：DS-039 是小幅、方向/seed 敏感的 pilot，不能升级为稳定 BioCAPM-DA 或 scanner/biology 机制结论。

### 3.7 DS-040：CAPM-conditioned frequency GRL

结果记录：[`docs/results/DS-040/README.md`](https://github.com/redtea-code/dual_shift/blob/main/docs/results/DS-040/README.md)。注册矩阵为 P0、F0-F3、R1-R3，2 directions × 2 seeds，共 32/32 real-data screening cells；训练预算为每 cell 1 epoch。

- F0-F3 使用 raw-Fourier source/target-style enhancement；F1/F2/F3 将 domain/intensity GRL 作用于完整 `B4`；R1/R2/R3 作用于 `R4=(I-P_task)B4`；
- 结果明显依赖 direction 和 seed，部分 cell 的 target BA 接近 0.5；
- 没有稳定证据证明 residual placement 优于 full placement；
- 没有稳定证据证明 domain 与 intensity GRL 互补；
- mechanism history 虽存在于报告，但没有独立、完整、checkpoint-bound 的纵向诊断闭环。

结论：screening 已完成，但 formal GRL mechanism claim 不成立。当前结果只能支持保留实现用于受控审计，不能支持 validated adaptation 或 scanner/biology correction。

### 3.8 DS-041：CAPM-conditioned source-free residual distribution alignment

结果记录：[`docs/results/DS-041/README.md`](https://github.com/redtea-code/dual_shift/blob/main/docs/results/DS-041/README.md)，最新加入主线的结果提交为 `98032fa`。完整 real-data pilot 为 2 directions × 2 seeds × C0-C4，共 20/20 cells。

协议固定为 frozen `original_capm`、`layer4_pixel`、512 channels、source-only rank-32 projector、target image-only subject-disjoint summary、最大 transport strength 0.25。C0-C4 为：

- C0：frozen original-CAPM control；
- C1：full feature K=1 diagonal transport；
- C2：residual K=1 diagonal transport；
- C3：residual K=2 GMM transport；
- C4：C3 加 source-only synthetic perturbation audit。

主要结果：

- ADNI→NACC：C0 target BA `0.6598 ± 0.0194`，C2/C3/C4 `0.6317 ± 0.0132`；
- NACC→ADNI：C0 target BA `0.6157 ± 0.0697`，C2/C3/C4 `0.6330 ± 0.0634`；
- C1 与 C0 的 target 指标完全相同；
- C2、C3、C4 的 target 指标完全相同；
- C4 的四组 recovery fraction 均为负或接近零，prediction agreement 虽为 1.0，但 synthetic recovery gate 不通过；
- residual discrepancy 相对 full K=1 的归一化 gate 整体更低，但该 discrepancy 在每个 artifact 内按最大值归一化，不能直接当作跨 artifact 的绝对距离比较；
- source adapted diagnostic 基本保持 source decision，但这只能说明任务决策未明显破坏，不能证明 biological preservation。

结论：DS-041 完成当前登记的 exploratory pilot，但现有证据不足以支持扩大模型复杂度、调整 rank/strength 或增加 seed。该实验没有证明稳定 target-domain gain、GMM superiority 或 biology-preserving harmonization。

报告还存在一个需要保留的审计限制：部分 target 表格中的 `n` 与 confusion matrix 总数不一致，提示可能混用了 scan-level 和 subject-level 聚合。正式引用具体数值前，应从原始 `predictions.json` 统一重算指标。

## 4. 截至目前的结论和启示

### 4.1 没有跨任务、跨方向、跨 seed 都稳定有效的适配模块

DS-035 的完整 FMM 有方向性 baseline 信号，但 DS-036、DS-037、DS-038、DS-039、DS-040 和 DS-041 均显示明显的 seed 或 direction 依赖。当前不能把任一单独模块写成普遍有效的 UDA 方法。

### 4.2 domain GRL 和 intensity GRL 不能被默认视为互补

DS-038 的 factorial 结果中，domain-only、intensity-only 和 both 的 paired gain 都不是全 seed 一致为正；DS-040 的 CAPM-conditioned 版本也没有闭合 complementarity evidence。两个 head 的组合效果必须按 task、direction 和 mechanism audit 分开解释。

### 4.3 residual transport 主要实现了任务保护，不等于目标域性能改善

DS-039/041 说明 bounded residual correction 可以让 source decision 和 CAPM anchor 基本稳定，但这种稳定性本身不会产生 target ranking gain。当前观察到的 BA 变化部分表现为 sensitivity/specificity 或 operating point 的变化，而不是 AUROC 的可靠提升。

### 4.4 更复杂的分布模型不自动带来更好的下游结果

DS-041 中 C2、C3、C4 的分类输出完全相同，说明 K=2 GMM 虽然改变了统计摘要，但没有形成可观测的分类决策收益。均值/方差、GMM、Fourier amplitude 等统计层面的变化，必须和 frozen classifier 的 ranking、calibration、source preservation 一起判断。

### 4.5 CAPM、biology 和 batch effect 的关系仍是功能假设，不是因果结论

当前合理的表述是：CAPM 提供 demographic-conditioned diagnostic path，适配模块尝试处理与 cohort/image distribution 相关的变化。不能表述为 CAPM 是纯生物学表示、residual 是纯 batch/scanner 表示，或实验已经识别了 scanner/field-strength 的因果效应。

### 4.6 统计 discrepancy 下降必须和任务终点同时成立

DS-041 的 residual discrepancy gate 可以下降，但 ADNI→NACC target BA 同时下降；DS-039 的小幅 BA 上升也伴随 operating-point trade-off 和不完整的 before/after discrepancy audit。因此“分布更接近”不能单独作为 adaptation 成功证据。

### 4.7 当前最重要的证据质量问题是 provenance 和聚合一致性

所有结果必须同时保存 resolved config、manifest/split digest、checkpoint hash、代码版本、target access flags、subject-level aggregation 定义和 prediction artifact。若 `n`、混淆矩阵和 BA 不一致，应先视为报告审计问题，而不是继续解释模型机制。

## 5. 当前结果引用口径

可以支持：

- 在特定 scan-filtered ADNI/NACC exploratory protocol 下，FMM 和 DS-034/C4 曾出现正点估计；
- DS-036、DS-037 的当前 transport 配置没有稳定 target 增益；
- DS-038 完成了历史性能矩阵，但机制诊断不完整，主线状态为 BLOCKED；
- DS-039 是小幅、seed-sensitive residual pilot；
- DS-040 screening 完成但不支持 GRL placement 或 complementarity 的 formal claim；
- DS-041 完成了 source-free residual distribution alignment 的 Stage B pilot，但 target superiority、GMM superiority 和 biological preservation 均未被证明。

不能支持：

- scanner、manufacturer 或 field-strength 的独立因果效应；
- domain GRL 或 intensity GRL 单独具有稳定增益；
- 两个 GRL head 已被证明互补；
- residual 是纯 batch/scanner 信息，或 CAPM 是纯 biological information；
- source-test/source-validation 的提升等同于 UDA 成功；
- internal frozen `T_test` 是新的 confirmatory external generalization evidence；
- 仅凭 DS-041 的归一化 discrepancy、prediction agreement 或单个 seed BA 证明 harmonization 成功。

## 6. 分支和文件边界

### `main`

截至本次更新，远端 `origin/main` 为 `98032fa`，包含 DS-041 最新结果报告。主线结果文档是实验结论的优先引用来源。

### `codex/frequency-uda-plan`

当前计划分支提交为 `fcd7a11`。该分支包含 Plan 34/C4 计划以及若干未提交文件。dirty 文件、interim 结果和本地实现不能默认视为 `main` 已发布证据。

### DS-041 implementation branch

DS-041 实现已在 `codex/ds041-capm-residual-distribution` 分支提交 `d928ac6`，包含 residual distribution module、stats builder、pilot、测试和 YAML。该分支的 focused tests 为 `27 passed`；主线报告中的 real-data artifacts 位于远程机器的 `/zjs/AD_Project/ds041_runs`，大体积 checkpoint 不复制到 GitHub 文档。

## 7. 新成员核对清单

- [ ] 先确认当前 checkout、branch 和 commit，不把 dirty worktree 当成 main 证据；
- [ ] 先读对应 DS-ID 的 plan 和 result，再查看实现代码；
- [ ] 核对 `S_train/S_val/S_test/T_adapt/T_test` 的 subject split 和 aggregation unit；
- [ ] 确认 adaptation 阶段没有 target label、covariate、environment、prediction 或 metric；
- [ ] 确认 checkpoint 只由 source validation 选择；
- [ ] 同时查看 target BA/AUROC、sensitivity/specificity、Macro-F1、MCC、Brier/ECE 和 confusion matrix；
- [ ] 机制实验必须有 finite、identity、correction/discrepancy、anchor/preservation 和 checkpoint provenance；
- [ ] 发现聚合或 provenance 不一致时，先记录为审计限制，不升级为方法机制解释；
- [ ] 不用单个 seed、source-side 改善或 target point estimate 升级为稳定 UDA/因果结论。

这份 handover 只描述截至 2026-08-29 已登记的实验计划、结果和结论边界；未登记的方法不属于本文证据。
