# Target-style CAPM UDA 五 seed 对比实验报告

- 日期：2026-08-19
- 方向：ADNI 1.5T → NACC 3T
- 任务：MCI vs AD
- seeds：42、43、44、45、46
- 对比：`capm` control vs `target_style_capm`
- 结果性质：internal frozen `T_test` exploratory benchmark，不是新的 confirmatory claim

## 1. 摘要

本实验以相同的 layer4 CAPM backbone、source split、优化器和 source-validation checkpoint 规则，逐 seed 配对比较无 target adaptation 的 `capm` 与仅使用无标签 target image style statistic 的 `target_style_capm`。五个 seed 的 target BA 差值均为负：target-style transport 的平均 target BA 为 **0.6415 ± 0.0214**，低于 CAPM control 的 **0.6663 ± 0.0067**，平均差值为 **−0.0248**。因此，在当前 strength=0.5、loss 权重和协议下，未观察到 target-style CAPM 的 target-side 增益。

该结果不证明 target-style transport 普遍无效；它只表明本次预注册 probe 未通过继续进入 fresh target holdout 的 go/no-go 条件。

## 1.1 证据等级与报告口径

本报告严格区分四个层次：代码/协议可运行、target-style 机制路径执行、相对 CAPM 的性能变化、以及 scanner/field-strength 机制解释。本实验最多支持前三级的受限描述，不支持第四级因果解释。

- **计划单元**：1 个方向 × 5 个 seeds × 2 个 variants = 10 个 paired runs。
- **完成单元**：10/10；没有删除、替换或依据 target test 结果重跑的单元。
- **主终点**：external target subject-mean balanced accuracy；辅助终点为 AUC、macro-F1、sensitivity、specificity。
- **证据性质**：`T_test` 是 internal frozen holdout，且已出现在历史实验中；所有 target 结果均为 exploratory benchmark。
- **未执行控制**：预案中的 `interaction="image_only"` secondary control 未运行，因此不能分离 CAPM-specific increment 与 generic target-style increment。


## 2. 数据现实与可识别性边界

本实验使用 ADNI 1.5T → NACC 3T 的跨队列设置。NACC 的协议支持集主要为 3T，ADNI 与 NACC 在 cohort、site、manufacturer、sequence 和 field strength 上存在混杂；因此本实验可检验普通 image/CAPM 模型能否跨队列泛化，但不能识别 scanner、manufacturer 或 field strength 的独立因果效应，也不能证明模型学习了 1.5T→3T 的普遍校正机制。

按照主分支的数据现实规范，ADNI→NACC 与 NACC→ADNI 不是对称问题；本报告只研究前者，不把方向平均或外推到 NACC→ADNI。target-style 结果不能被表述为 scanner-causal、field-strength-causal 或对任意未见协议普遍有效。

## 2. 实验协议与数据边界

- `capm`：CAPM source-only control，target adaptation disabled。
- `target_style_capm`：clean source CE + transported source-feature CE + consistency loss。
- target feature：shared layer4 feature 的空间 FFT amplitude；保留 source phase。
- target feature 在 transport 前 detached；transport strength=`0.5`。
- source rows：490 train / 145 validation / 149 source test；subject-level n=48/49。
- target rows：330 `T_adapt` / 324 `T_test`；subject-level `T_test` n=263。
- checkpoint：仅由 source validation balanced accuracy 选择。
- target labels、target covariates 和 target metrics 不参与 adaptation loss 或 checkpoint selection。
- target covariates 仅用于最终 `T_test` CAPM inference，不传入 transport。
- 结果报告使用 subject-mean aggregation。
- 每个 run 的 `metrics.json` 内嵌 resolved config 与 protocol 字段；本报告不把实验输出目录中的运行产物当作 GitHub source artifact。
- 本次报告未保存独立的 frozen manifest hash、原始输入文件 hash、环境锁定文件或单独 `git_commit.txt`；这些 provenance 缺口限制了可复现性等级，不能写成完整 E0/E2 归档通过。

## 3. 执行门、完成度与失败单元

| 门/单元 | 状态 | 证据与边界 |
|---|---|---|
| target-style transport targeted tests | PASS | 最新提交对应测试曾报告 `4 passed`；覆盖 identity、shape/gradient detach 与 clean/mixed path |
| 主分支完整 E0 | 未执行/未归档 | 未在本实验报告中声称 main 规定的完整 pytest/py_compile E0 通过 |
| paired run matrix | PASS | 10/10 `metrics.json` 与 `best_checkpoint.pt` 存在 |
| source-validation-only checkpoint selection | PASS（按产物字段） | 每个 run 记录 `target_test_used_for_selection: false` |
| target-label blind adaptation | PASS（按产物字段） | 每个 run 记录 `target_labels_used_in_training: false` |
| image-only secondary control | 未执行 | 不能作 CAPM-specific 与 generic target-style 的分离结论 |
| fresh external target holdout | 未执行 | 当前 T_test 不是新的 confirmatory holdout |

没有实验单元因 target 指标被删除或替换；未执行单元保留在报告中，不从主矩阵中隐去。

## 3. Target test 主结果

| seed | CAPM BA | Target-style BA | Δ BA | CAPM AUC | Target-style AUC | Δ AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6648 | 0.6431 | -0.0217 | 0.7536 | 0.7052 | -0.0484 |
| 43 | 0.6746 | 0.6706 | -0.0039 | 0.7642 | 0.7296 | -0.0346 |
| 44 | 0.6697 | 0.6454 | -0.0243 | 0.7459 | 0.7106 | -0.0353 |
| 45 | 0.6565 | 0.6380 | -0.0184 | 0.7192 | 0.7136 | -0.0056 |
| 46 | 0.6661 | 0.6105 | -0.0556 | 0.7536 | 0.7335 | -0.0201 |
| **mean ± SD** | **0.6663 ± 0.0067** | **0.6415 ± 0.0214** | **-0.0248** | **0.7473 ± 0.0170** | **0.7185 ± 0.0124** | **-0.0288** |

### 3.1 其他 target test 指标

| variant | BA mean ± SD | AUC mean ± SD | macro-F1 mean ± SD | sensitivity mean ± SD | specificity mean ± SD |
|---|---:|---:|---:|---:|---:|
| capm | 0.6663 ± 0.0067 | 0.7473 ± 0.0170 | 0.6686 ± 0.0117 | 0.5271 ± 0.0481 | 0.8056 ± 0.0546 |
| target_style_capm | 0.6415 ± 0.0214 | 0.7185 ± 0.0124 | 0.6374 ± 0.0147 | 0.4988 ± 0.1508 | 0.7843 ± 0.1098 |

target-style 的 target BA 在 5/5 个 seed 上低于 matched CAPM control；target AUC 也在 5/5 个 seed 上下降。target-style 的 sensitivity 均值略低（0.4988 vs 0.5271），specificity 均值略低（0.7843 vs 0.8056），未显示出 balanced trade-off。

## 4. Source-side 结果

### 4.1 Source validation（checkpoint selector）

| variant | BA mean ± SD | AUC mean ± SD | macro-F1 mean ± SD | sensitivity mean ± SD | specificity mean ± SD |
|---|---:|---:|---:|---:|---:|
| capm | 0.6937 ± 0.0140 | 0.7199 ± 0.0257 | 0.6516 ± 0.0193 | 0.8000 ± 0.0815 | 0.5875 ± 0.0713 |
| target_style_capm | 0.6969 ± 0.0392 | 0.7199 ± 0.0176 | 0.6352 ± 0.0598 | 0.8625 ± 0.1355 | 0.5312 ± 0.1547 |

### 4.2 Source test（训练后诊断）

| variant | BA mean ± SD | AUC mean ± SD | macro-F1 mean ± SD | sensitivity mean ± SD | specificity mean ± SD |
|---|---:|---:|---:|---:|---:|
| capm | 0.6848 ± 0.0375 | 0.7379 ± 0.0286 | 0.6393 ± 0.0291 | 0.8000 ± 0.0685 | 0.5697 ± 0.0136 |
| target_style_capm | 0.7104 ± 0.0389 | 0.7561 ± 0.0414 | 0.6432 ± 0.0595 | 0.8875 ± 0.1202 | 0.5333 ± 0.1296 |

target-style source validation BA 平均变化为 `+0.0031`；source test BA 平均变化为 `+0.0256`。source validation 的平均 BA 略升，但 SD 从 `0.0140` 增至 `0.0392`，且 sensitivity/specificity 变化显示预测阈值行为不稳定。

## 5. 逐 seed paired difference

| seed | Δ target BA | Δ target AUC | Δ source-val BA | Δ source-test BA | target-style best epoch | CAPM best epoch |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.0217 | -0.0484 | -0.0469 | -0.0587 | 6 | 21 |
| 43 | -0.0039 | -0.0346 | +0.0625 | +0.0634 | 48 | 23 |
| 44 | -0.0243 | -0.0353 | -0.0469 | +0.0152 | 26 | 15 |
| 45 | -0.0184 | -0.0056 | +0.0312 | +0.0938 | 18 | 18 |
| 46 | -0.0556 | -0.0201 | +0.0156 | +0.0142 | 34 | 27 |

The five-seed paired target BA difference is negative in every seed, so the result does not rely on a single unfavorable or favorable seed.

## 6. Protocol and leakage audit

All 10 runs report the same target-data boundary fields:
- `target_labels_used_in_training: false`
- `target_covariates_used_in_transport: false`
- `target_test_used_for_selection: false`
- `target_image_information: layer4_spatial_fft_amplitude_only`
- `source_phase_preserved: true`
- source train rows=490, target adapt rows=330, target test rows=324
- target test subjects=263 for every run

The saved protocol does not expose a target diagnosis label to the adaptation view. The target-style branch uses target image only to derive a detached feature-space amplitude statistic.

审计结论只覆盖运行器保存的字段，不替代独立 manifest/E0 证据。尤其是：报告没有独立记录 ADNI/NACC manifest 的 field-strength 计数、paired-origin 计数、manifest SHA256、源码 commit 文件或环境版本。因此，本报告可以报告 target-label boundary 和 subject-level split 的运行时字段，但不能声称完成主分支规范要求的完整 manifest provenance audit。

## 7. Interpretation and decision

### Findings
1. The matched five-seed comparison does not support a positive target-style CAPM increment under the locked configuration.
2. The target-style variant reduced mean target BA by `0.0248` and mean target AUC by `0.0288` relative to CAPM.
3. Source-side effects are mixed: source-test BA increased on average, but target performance decreased consistently; source improvement therefore does not establish UDA benefit.
4. The transport may have shifted the source/target decision threshold: target-style sensitivity and specificity both varied substantially across seeds.

### 可支持的表述

> 在当前 ADNI 1.5T→NACC 3T、固定 strength=0.5 和冻结 source-validation selector 下，五 seed paired exploratory comparison 未显示 target-style feature transport 优于 matched CAPM control。

### 禁止的表述

不得将本实验写成“target-style transport 已被证明无效”“模型学会了 scanner/field-strength 因果响应”或“该方法对任意未见协议普遍有效”。也不得用 source-test 改善替代 target 主终点，或把单个 seed 的机制/性能变化升级为稳定机制结论。

### Go/no-go
The pre-specified rule required a positive five-seed paired target BA difference without unacceptable source-side or subgroup failure. The observed paired difference is negative in all five seeds, so this probe is **NO-GO** for a fresh confirmatory target holdout or promotion as an improvement over CAPM.

This is not evidence for scanner-causal or field-strength-causal explanations, and it does not establish that Fourier feature transport is universally invalid. Any follow-up should be a separately locked diagnostic/control (for example `image_only`, strength sensitivity, or transport source-label agreement) and must not be selected from these target-test scores.

## 8. Limitations

- Current `T_test` is an internal frozen holdout that has appeared in historical experiments; target results are exploratory.
- Five seeds share the same source split policy and target split seed policy from the configuration; this is repeated training randomness, not five independent cohort draws.
- This run records subject-aggregated metrics; the JSON schema provides `n`/`n_subjects`, not a separate scan-level metric field.
- The experiment does not include the optional `interaction="image_only"` secondary control, so it cannot separate CAPM-specific effects from a generic target-style effect.
- The transport strength and loss weights were locked before evaluation and were not tuned using target test results.
- The report does not include the independent E0 artifacts required by the main-branch multi-machine plan: frozen manifest audit, manifest hashes, resolved config hash file, environment record, and `git_commit.txt`.

## 9. Artifact locations and completeness

- Worktree: `/zjs/AD_Project/dual_shift_target_style_uda`
- Results: `outputs/target_style_capm_uda_scan_filtered_1p5t_mci_ad/ADNI_to_NACC/seed_<42-46>/<variant>/metrics.json`
- Plan: `docs/TARGET_STYLE_CAPM_UDA_EXPERIMENT_PLAN_2026-08-19.md`
- Compared variants: `capm`, `target_style_capm`
