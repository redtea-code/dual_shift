# FMM Baseline 完整消融实验报告

- 日期：2026-08-18
- 分支：`codex/fmm-baseline`，实验工作区 `dual_shift_fmm_baseline`
- 方向：ADNI 1.5T → NACC 3T
- 任务：MCI vs AD
- seed：42
- 矩阵：B0-ref、B1-fmm、B1a-no-source-fft、B1b-no-attention、B1c-no-grl
- 结果性质：T_test 已出现在历史 C0–C4 报告中，因此本报告中的 target 结果仅为 exploratory benchmark，不是新的 confirmatory claim。

## 1. 实验目的

本实验独立复现并拆解 Frequency Mixup Manipulation（FMM）raw-image UDA baseline，回答两个问题：

1. 完整 FMM 相对其 reference encoder 的 source/target 表现如何？
2. source Fourier synthesis、spatial attention consistency 和 GRL discriminators 分别是否贡献于结果？

本实验不是 C4 frequency-prior route 的延伸，也不把 FMM 的结果解释为 C4 机制的验证或反证。

## 2. 实验协议

固定条件：

- FMM 10-convolution 3D encoder，channels `[8,8,16,16,32,32,64,64,128,128]`。
- Adaptive pool shape `[2,3,2]`，classifier hidden width 64，dropout 0.5。
- 50 epochs，batch size 4，evaluation batch size 2。
- Adam，learning rate `1e-4`，weight decay `1e-4`，class-weighted CE。
- source split：490 train / 145 validation / 149 source test rows；subject-level metrics 为 48 / 49 个 validation/test subjects。
- target split：330 `T_adapt` rows、324 `T_test` rows；subject-level target test 263 subjects。
- `T_adapt` 仅提供 image 和 subject/domain membership，不提供 diagnosis label。
- S_val 是唯一 checkpoint selector；T_test 不参与训练、调参或 checkpoint 选择。
- FMM 配置：amplitude mixing interval `[0,1]`、intensity scale `[0.8,1.2]`、noise std `0.05`、unit domain/attention/intensity coefficients、GRL coefficient 1.0。

预检已完成：FMM 单元测试 `6 passed`；五个变体的 label-blind smoke 均 exit code 0。

## 3. 变体定义

| ID | variant | source stage | inter-domain stage | source FFT | attention | GRL | 目的 |
|---|---|---:|---:|---:|---:|---:|---|
| B0-ref | `b0_ref` | no | no | no | no | no | FMM reference encoder 的 source ERM 架构对照 |
| B1-fmm | `b1_fmm` | yes | yes | yes | yes | yes | 完整 FMM |
| B1a | `b1a_no_source_fft` | yes | yes | no | yes | yes | 移除 Stage-I source Fourier synthesis |
| B1b | `b1b_no_attention` | yes | yes | yes | no | yes | 移除 attention consistency |
| B1c | `b1c_no_grl` | yes | yes | yes | yes | no | 移除 intensity/domain 两个 GRL discriminator |

## 4. Source validation 结果

S_val 只用于 checkpoint 选择，是本实验的主要 source-side 选择指标。

| variant | best epoch | BA | Accuracy | AUC |
|---|---:|---:|---:|---:|
| B0-ref | 46 | 0.6429 | 0.6667 | 0.6893 |
| B1-fmm | 49 | 0.6393 | 0.6458 | 0.6732 |
| B1a-no-source-fft | 42 | **0.6786** | **0.7083** | **0.7357** |
| B1b-no-attention | 46 | 0.6429 | 0.6250 | 0.6821 |
| B1c-no-grl | 24 | 0.6500 | 0.6250 | 0.6893 |

B1 完整 FMM 没有超过 B0-ref。B1a 在 S_val 上最高，但这只说明移除 source Fourier synthesis 后的模型在当前 source validation split 上表现最好，不能推断其 target generalization 更好。

## 5. Source test 结果

Source test 在训练结束后作为诊断性结果读取，不参与 checkpoint 选择。

| variant | BA | Accuracy | AUC |
|---|---:|---:|---:|
| B0-ref | 0.4940 | 0.5306 | 0.6173 |
| B1-fmm | **0.6369** | 0.6327 | **0.7364** |
| B1a-no-source-fft | 0.5714 | 0.5918 | 0.6582 |
| B1b-no-attention | **0.6786** | **0.6531** | 0.7058 |
| B1c-no-grl | 0.6310 | 0.6122 | 0.6429 |

B0 的 source-test BA 明显低于其 S_val，说明 source split 之间存在较大波动。B1 和 B1b 的 source-test BA 较高，但不能仅凭单一 seed 的 source-test 结果确定组件贡献。

## 6. Exploratory target test 结果

以下结果是在 source validation checkpoint 已冻结、训练完成后才计算的 target test 指标。由于该 T_test 已在历史实验中出现，这些结果只作为探索性 benchmark。

| variant | Target BA | Accuracy | AUC |
|---|---:|---:|---:|
| B0-ref | 0.5403 | 0.6920 | 0.6912 |
| B1-fmm | 0.6009 | 0.7186 | 0.7748 |
| B1a-no-source-fft | **0.6929** | **0.7757** | **0.8260** |
| B1b-no-attention | 0.6647 | 0.7224 | 0.7964 |
| B1c-no-grl | 0.5606 | 0.6920 | 0.7142 |

相对 B0-ref 的 target BA 变化：

| variant | Δ target BA | Δ target AUC |
|---|---:|---:|
| B1-fmm | +0.0606 | +0.0836 |
| B1a-no-source-fft | **+0.1526** | **+0.1349** |
| B1b-no-attention | +0.1244 | +0.1053 |
| B1c-no-grl | +0.0203 | +0.0231 |

探索性结果中，B1a-no-source-fft 最高，完整 B1-fmm 次之，B1b-no-attention 也高于完整 B1；B1c-no-grl 接近 B0。这一模式不支持“完整 FMM 的所有组件都必须存在”的简单结论。

## 7. 组件分析

### 7.1 Source Fourier synthesis

B1a 去掉 source Fourier synthesis 后，S_val 反而最高，target exploratory BA/AUC 也最高：`0.6929 / 0.8260`。这表明在本次实现和配置下，source-stage Fourier synthesis 可能引入不利的训练扰动，或者其收益被其他 FMM 组件抵消。

但 B1a 的 source-test BA 低于 B1 和 B1b，且只有 seed42；不能将其直接解释为稳定优势。需要多 seed 和独立 target holdout 才能验证。

### 7.2 Attention consistency

B1b 去掉 attention consistency 后，source-test BA 为最高的 `0.6786`，target exploratory BA 为 `0.6647`，高于 B1 的 `0.6009`。在当前 seed 上，attention consistency 没有显示出增益，反而完整 B1 的 target 结果低于 B1b。

这只能说明当前 attention loss、attention pairing 或其权重未展现稳定收益，不能证明 attention consistency 在 FMM 中普遍无效。

### 7.3 GRL discriminators

B1c 去除两个 GRL discriminator 后，target exploratory BA 下降到 `0.5606`，接近 B0 的 `0.5403`，明显低于 B1 的 `0.6009`。这给出一个较明确的探索性信号：在当前配置中，GRL 相关分支可能是 FMM target-side 变化的主要贡献来源。

但 B1c 的 loss 结构也发生了变化：其 domain/intensity loss 为 0，不能把此结果归因于单一 discriminator；本变体同时去除了两个 GRL heads。

## 8. 训练 loss 与机制审计

| variant | final total | classification | domain | attention | intensity |
|---|---:|---:|---:|---:|---:|
| B0-ref | 0.0629 | 0.0629 | 0 | 0 | 0 |
| B1-fmm | 1.4511 | 0.0719 | 0.6822 | 0.0041 | 0.6929 |
| B1a-no-source-fft | 1.6573 | 0.2710 | 0.6908 | 0.0018 | 0.6937 |
| B1b-no-attention | 1.4935 | 0.1111 | 0.6894 | 0 | 0.6931 |
| B1c-no-grl | 0.0075 | 0.0067 | 0 | 0.0008 | 0 |

B1/B1a/B1b 的 domain/intensity discriminator loss 接近 `log(2)=0.693`，说明在最后记录点上 discriminator 接近二分类随机水平；这不是 discriminator“有效”的充分证据。需要进一步查看随 epoch 变化的 diagnostics 和 discriminator accuracy，不能只以 loss 接近 0.693 判定成功。

## 9. Leakage 与 provenance 审计

五个变体的 `audit.json` 均确认：

- `target_label_access_during_training: false`
- `target_labels_used_for_selection: false`
- `target_test_evaluation_after_fit: true`
- source train/val/test subject splits disjoint
- target adapt/test subject splits disjoint
- source 与 target subject sets disjoint
- sampled target subjects only appear for B1/B1a/B1b/B1c, not B0
- five variants record the same upstream FMM reference commit：`580625cee5bfc1474fe8700e530ade07ac5e9776`

每个变体均保存了 `summary.json`、`audit.json`、`config.yaml`、`predictions.json` 和 `best.pt`。配置 hash 在五个变体中一致：`e2244dd151d49f19ea6c8d3d0b889c1219c550355ac6c362f56dea1cdbfeb725`。

## 10. 结论与限制

### 结论

1. 完整 FMM B1 相比 B0-ref 在探索性 target test 上提高 BA `+0.0606`，提高 AUC `+0.0836`。
2. B1a-no-source-fft 是本次单 seed 探索性 target 结果最佳变体；它的 target BA/AUC 为 `0.6929/0.8260`。
3. B1b-no-attention 也超过完整 B1，说明当前 attention consistency 没有表现出稳定收益。
4. B1c-no-grl 接近 B0，提示 GRL 分支可能比 source FFT 或 attention 对当前 target 结果更重要，但该结论仍需多 seed 验证。
5. 没有任何 target label leakage；target labels 只在拟合完成后的探索性评估阶段读取。

### 限制

- 只有 seed42，不能作稳定性结论。
- T_test 已在历史 C0–C4 报告中出现，不是新的 unread holdout。
- 本实现中的若干 FMM 超参数来自预注册可运行选择，不应称为作者代码的 exact reproduction。
- B1c 同时移除两个 GRL discriminator，不能分离 domain GRL 与 intensity GRL 的独立作用。
- B1a/B1b 的结果不能排除 source-test 与 target-test 的抽样波动。
- 目前未运行 B2-capm 与 B3-fmm-core-capm 的 same-backbone comparison。

**最终判定：FMM 完整基线在当前探索性 ADNI→NACC 评估中优于其 B0 reference encoder，但最强结果来自去除 source Fourier synthesis 的 B1a，而不是完整 B1。结果支持继续进行多 seed 验证与更细粒度 GRL 分解，不支持直接宣称完整 FMM 已被验证为稳定的跨域方法。**

## 11. Artifact locations

- FMM 代码工作区：`/zjs/AD_Project/dual_shift_fmm_baseline`
- 配置：`fmm_baseline_scan_filtered_1p5t_mci_ad.yaml`
- 结果：`outputs/fmm_full/ADNI_to_NACC/seed_42/<variant>/`
- 设计文档：`docs/FMM_BASELINE_COMPARISON_DESIGN_2026-08-17.md`
- 上游 FMM reference commit：`580625cee5bfc1474fe8700e530ade07ac5e9776`
