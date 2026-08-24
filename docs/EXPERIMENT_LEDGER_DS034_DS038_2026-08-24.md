# DS-034 至 DS-038 实验总览

- **整理日期**：2026-08-24
- **范围**：DS-034 至当前 DS-038 扩展实验
- **主指标**：target subject-mean balanced accuracy（BA）；除非特别说明，均为冻结 checkpoint 后的 target 评估
- **统计口径**：跨 seed 报告均值 ± 样本标准差（SD）
- **重要说明**：ADNI→NACC 与 NACC→ADNI、MCI–AD 与 NC–MCI 不合并统计；不完整矩阵不作为完整 factorial 结果解释。

## 一、总体结论

1. **DS-035 是目前最清晰的 FMM 探索性正向信号**：完整 FMM B1 相对 B0 在 3 个 seed 上 target BA 均为正，平均提升 `+0.0305`，但不能据此拆分 domain GRL 与 intensity GRL 的独立贡献。
2. **DS-036 与 DS-037 均未支持 target-style amplitude transport**：DS-036 的 target-style CAPM 在 5/5 seed 低于 CAPM；DS-037 的所有 strength/phase 变体均未通过预设的正向改进规则。
3. **DS-038 仍主要是机制审计与筛查，不是最终采用依据**：ADNI→NACC 的 MCI–AD 原注册矩阵只有 `9/20` 单元可验证；NC–MCI 两 seed 矩阵已具备 `8/8` 产物，但统计功效不足；NACC→ADNI MCI–AD 扩展只有 `4/8` 单元可从稳定 artifact root 复核。
4. 所有 target 结果均来自历史使用过的内部冻结 holdout，因而只能作为 **exploratory benchmark**，不能作为新的 confirmatory claim。
5. 当前数据中 cohort、site、manufacturer、acquisition、预处理和 field strength 存在混杂；这些实验不支持 scanner、manufacturer 或 field-strength 的因果解释。

## 二、实验总表

| 实验单元 | 任务 / 方向 / seeds | 可验证完成度 | 主要问题与方法 | 核心结果 | 结论与状态 |
|---|---|---:|---|---|---|
| **DS-034 · E2 source screen** | MCI–AD；ADNI→NACC 与 NACC→ADNI；seed 42 | 35/36 | ResNet18，多 preset、多 CAPM/Transformer 变体；source-validation BA + collapse guard 选 checkpoint | 最佳 source BA：ADNI→NACC 的 layer4 `conv_gate` 为 `0.688`；NACC→ADNI 的 layer4 `image_only` 为 `0.786` | 缺少 layer5 ADNI→NACC `transformer_self` 的正式 journal artifact；不能视为完整 source screen |
| **DS-034 · E3 frozen target matrix** | MCI–AD；双方向；seed 42 | 36/36 | 冻结 E2 checkpoint 后 target 盲测 | 最佳 target BA：ADNI→NACC layer4 `transformer_self` 为 `0.692`；NACC→ADNI layer5 `original_capm` 为 `0.677` | 完整探索性 target 矩阵；没有跨方向、跨尺度的唯一稳定赢家 |
| **DS-034 · NACC→ADNI validation** | MCI–AD；NACC→ADNI；seeds 43/44 | 12/12 | ResNet18，三个 preset，仅比较 image-only 与 original CAPM | CAPM 相对 image-only 的 BA 差：layer3 `−0.039`、layer4 `+0.029`、layer5 `+0.033` | seed43/44 未复现严格层级递增趋势；CAPM 增益符号不跨 seed 稳定 |
| **DS-034 · ResNet10 L4 six variants** | MCI–AD；NACC→ADNI；seeds 43/44 | 12/12 | ResNet10 layer4_pixel，六种变体 | original CAPM `0.662 ± 0.023`；image-only `0.632 ± 0.039`；平均差 `+0.030` | 在固定 ResNet10/layer4/NACC→ADNI 条件下有初步支持，但仍为小规模探索性结果 |
| **DS-034 · L4 pixel vs patch2** | MCI–AD；NACC→ADNI；seeds 43/44 | 12/12 | 匹配比较 layer4 pixel 与 patch2 tokenization | Patch−Pixel BA：original CAPM `−0.036`、transformer self `−0.007`、transformer cross `−0.013` | pixel 在该狭窄 tokenization 对照中更好；不能外推到其他 layer、方向或 backbone |
| **DS-034 · feature-frequency audit** | MCI–AD；双方向；seeds 43/44 | 12 个 paired cells | 深层 feature 频谱摘要与 domain classifier；同时比较 target BA | domain AUROC 范围 `0.793–0.952`；`Δ target BA` 与 `Δ domain BA` 的相关系数 `r=+0.034` | 存在明显 cohort/domain 频谱信息，但没有稳定证据表明降低 domain separability 会带来 target BA 增益 |
| **DS-034 · C0–C4 frequency UDA** | MCI–AD；ADNI→NACC；seeds 43/44 | 10/10 | C0 baseline、C1 env-DRO、C2 uniform、C3 permuted、C4 full target-spectrum UDA | C4 `0.693 ± 0.051`；C0 `0.670 ± 0.042`；C4−C0 每个 seed 均为 `+0.023`，但 paired bootstrap CI 均跨 0 | **NO-GO**：不能确认 full target-spectrum UDA 稳定优于 C0，也不能进入下一机制开发 |
| **DS-035 · FMM baseline ablation** | MCI–AD；ADNI→NACC；seeds 42–44 | 15/15 | B0-ref、B1-fmm、B1a-no-source-fft、B1b-no-attention、B1c-no-grl | B0 `0.5827 ± 0.0456`；B1 `0.6131 ± 0.0166`；B1−B0 `+0.0305`，3/3 seed 为正 | 已完成的探索性 baseline/component screen；B1a/B1b 均值较高但波动较大；B1c 同时移除两个 GRL，不能完成单 head 归因 |
| **DS-036 · target-style CAPM** | MCI–AD；ADNI→NACC；seeds 42–46 | 10/10 | CAPM control vs target-style CAPM；strength=`0.5`，source phase | CAPM `0.6663 ± 0.0067`；target-style `0.6415 ± 0.0214`；paired BA 差 `−0.0248`，5/5 seed 为负 | **NO-GO**：当前锁定 transport 配置没有 target-side 增益；不等于证明该方法普遍无效 |
| **DS-037 · amplitude transport audit** | MCI–AD；ADNI→NACC；seeds 42–44 | 18/18 | AT0–AT5；strength `0/0.25/0.5/1.0`；source/target phase | AT0 `0.6488 ± 0.0391`；AT1 `0.6184`；AT2 `0.6227`；AT3 `0.5925`；AT4 `0.6419`；AT5 `0.6283`。所有 transport 变体均值差为负 | **NO-GO**：没有 strength/phase 组合满足三 seed 正向、源标签保持和机制一致性的共同规则 |
| **DS-038 · MCI–AD GRL factorial** | MCI–AD；ADNI→NACC；注册 seeds 42–46 | 9/20 | G0 no-GRL、G1 domain-only、G2 intensity-only、G3 both-GRL；checkpoint-bound diagnostics 和 frozen probe | common seeds 42/43：G1 `+0.0748 ± 0.0043`；G2 `−0.0401 ± 0.0486`；G3 `−0.0357 ± 0.0298`（相对 G0） | **BLOCKED**：缺失 11 个可验证单元，不能作五 seed factorial 或 GRL 采用结论 |
| **DS-038 · NC–MCI label extension** | NC–MCI；ADNI→NACC；seeds 42/43 | 8/8 artifact | 同一 G0–G3 factorial；label mapping `{1:NC, 2:MCI}`，positive class=MCI | G0 `0.5537 ± 0.0167`；G1 Δ`+0.0185 ± 0.0017`；G2 Δ`+0.0203 ± 0.0048`；G3 Δ`+0.0205 ± 0.0284` | 两 seed 探索性筛查；AUC、source-test 和 frozen probe 方向不完全一致，不能据此采用 GRL |
| **DS-038 · MCI–AD reversed-direction extension** | MCI–AD；NACC→ADNI；seeds 42/43 | 4/8 currently verifiable | G0–G3；方向和 seed 数均为原注册协议外扩展 | 当前稳定 artifact root 仅能确认 G0/G3；历史 extraction 中 G1 Δ`−0.0174`、G2 Δ`−0.0069`、G3 Δ`−0.0250` | **BLOCKED**：G1/G2 产物缺失，历史表不能当作可复核的完整 factorial 结果；不得与 ADNI→NACC 合并 |

## 三、按实验阶段的详细说明与结果

### DS-034：ResNet/CAPM/频域 UDA 基线与结构筛查

DS-034 的目标是建立 MCI–AD 跨队列 UDA 基线，比较不同 feature scale、CAPM/Transformer 交互结构、pixel/patch tokenization 和频域表征。实验同时覆盖 ADNI→NACC 与 NACC→ADNI，但两者不是对称问题：NACC 主要是单一 3T 支持集，NACC→ADNI 更接近 unsupported-protocol stress test。

主要结果：

- seed42 的完整冻结 target 矩阵为 36/36，但 E2 source screen 有 1 个正式 artifact 缺失。
- target 最优结果依赖方向和 preset：ADNI→NACC 的最佳单元是 layer4 transformer-self，BA=`0.692`；NACC→ADNI 的最佳单元是 layer5 original CAPM，BA=`0.677`。
- NACC→ADNI seeds43/44 没有确认 seed42 所观察的“层数越深越好”趋势。
- ResNet10 layer4 六变体中，original CAPM 在两个 seed 均高于 image-only，平均 BA 差为 `+0.030`，但该结论仅限于固定方向、backbone 和 feature scale。
- pixel 相对 patch2 在三个受影响变体中均取得更高跨 seed BA，但该差异无法区分 patch 聚合效应和 token 数量变化效应。
- C0–C4 频率 UDA 中，C4 的点估计高于 C0，但两 seed 的 paired uncertainty 均跨 0，因此不能称为稳定方法增益。

### DS-035：FMM baseline 三 seed 消融

DS-035 对完整 FMM 及其组件进行五变体、三 seed 消融：

- B0-ref：reference encoder；
- B1-fmm：完整 FMM；
- B1a：移除 source-stage Fourier synthesis；
- B1b：移除 attention consistency；
- B1c：同时移除 domain/intensity 两个 GRL discriminator。

核心结果：

| Variant | Target BA mean ± SD | Target AUC mean ± SD | Δ BA vs B0 |
|---|---:|---:|---:|
| B0-ref | 0.5827 ± 0.0456 | 0.7156 ± 0.0509 | — |
| B1-fmm | 0.6131 ± 0.0166 | 0.7936 ± 0.0225 | +0.0305 |
| B1a-no-source-fft | 0.6187 ± 0.0681 | 0.8031 ± 0.0474 | +0.0360 |
| B1b-no-attention | 0.6285 ± 0.0759 | 0.7907 ± 0.0311 | +0.0458 |
| B1c-no-grl | 0.5726 ± 0.0114 | 0.6980 ± 0.0825 | −0.0101 |

B1 是当前最稳定的正向信号：3/3 seed 均高于 B0，但 seed44 的增益只有 `+0.0011`。B1a 和 B1b 的均值更高，却伴随更大的 seed 波动；不能称为稳定替代方案。B1c 接近或低于 B0，只能说明联合 GRL 路径可能有贡献，不能说明 domain GRL 或 intensity GRL 哪一个起作用。

### DS-036：target-style CAPM 五 seed 对比

DS-036 在相同 layer4 CAPM backbone 下，逐 seed 比较 source-only CAPM 与仅使用无标签 target image style statistic 的 target-style CAPM。transport strength 固定为 `0.5`，source phase 保留。

| Variant | Target BA mean ± SD | Target AUC mean ± SD | 结果 |
|---|---:|---:|---|
| CAPM control | 0.6663 ± 0.0067 | 0.7473 ± 0.0170 | baseline |
| Target-style CAPM | 0.6415 ± 0.0214 | 0.7185 ± 0.0124 | BA `−0.0248`；AUC `−0.0288` |

target-style BA 和 AUC 均在 5/5 seed 低于 CAPM control。source-side 指标并未显示相同方向：source-test BA 平均上升 `+0.0256`，但这不能替代 target 主终点，也不能证明 UDA 有效。

结论是当前锁定配置 NO-GO，不是“target-style transport 已被证明普遍无效”。

### DS-037：amplitude strength/phase mechanism audit

DS-037 将 target-style transport 拆为六种条件：

- AT0：无 transport；
- AT1–AT3：source phase，strength=`0.25/0.5/1.0`；
- AT4–AT5：target phase，strength=`0.5/1.0`。

| Variant | Strength | Phase | Target BA mean ± SD | Δ BA vs AT0 |
|---|---:|---|---:|---:|
| AT0 | 0.00 | none | 0.6488 ± 0.0391 | — |
| AT1 | 0.25 | source | 0.6184 ± 0.0431 | −0.0304 ± 0.0296 |
| AT2 | 0.50 | source | 0.6227 ± 0.0038 | −0.0261 ± 0.0417 |
| AT3 | 1.00 | source | 0.5925 ± 0.0283 | −0.0563 ± 0.0180 |
| AT4 | 0.50 | target | 0.6419 ± 0.0341 | −0.0069 ± 0.0508 |
| AT5 | 1.00 | target | 0.6283 ± 0.0426 | −0.0205 ± 0.0657 |

AT3 在三个 seed 均低于 AT0，paired BA 差为 `−0.0741`、`−0.0565`、`−0.0381`。AT1、AT2、AT4、AT5 的 seed 方向不一致，不能用孤立 seed 或 AUC 正值覆盖 BA 的预设判断规则。因此 DS-037 对 strength/phase 选择为 NO-GO。

### DS-038：Domain/Intensity GRL Factorial Audit

DS-038 将 FMM 的联合 GRL 消融拆分为四个条件：

- **G0 `no_grl`**：domain/intensity GRL 均关闭；
- **G1 `domain_only`**：仅 domain GRL 开启；
- **G2 `intensity_only`**：仅 intensity GRL 开启；
- **G3 `both_grl`**：两个 GRL head 均开启。

统一状态符号：

- `※`：尚未完成、当前目录缺失完整 artifact、仅有历史提取值，或不能形成可复核的完整统计；
- `✓`：当前 artifact 集合已通过基本完整性检查；
- `⚠`：虽然有 artifact，但该结果仍属于 exploratory/interim，不能支持正式采用或机制因果结论。

所有 DS-038 target 结果均来自历史使用过的内部 holdout，不能作为新的 confirmatory claim。三组实验分别报告，不跨任务或方向合并。

#### A. MCI–AD，ADNI→NACC

**协议与矩阵。** 任务为 MCI vs AD，方向为 ADNI 1.5T → NACC 3T，原注册矩阵为 G0–G3 × seeds 42–46，共 20 cells。当前可复核的性能矩阵为 **9/20**；common seeds 42/43 的四个 variant 均有结果，G0 另有 seed44。其余单元统一记为 `※`，不能写成完整五-seed factorial 结果。

| Variant | 可复核 seeds | Target BA mean ± SD | Target AUC mean ± SD | Source-test BA mean ± SD | 状态 |
|---|---|---:|---:|---:|---|
| G0 `no_grl` | 42, 43（44 另有结果） | 0.5853 ± 0.0350 | 0.6763 ± 0.1188 | 0.6250 ± 0.0337 | ⚠ interim |
| G1 `domain_only` | 42, 43 | 0.6601 ± 0.0392 | 0.8114 ± 0.0089 | 0.6786 ± 0.0000 | ⚠ screening |
| G2 `intensity_only` | 42, 43 | 0.5452 ± 0.0136 | 0.7643 ± 0.0507 | 0.6667 ± 0.0505 | ⚠ screening |
| G3 `both_grl` | 42, 43 | 0.5496 ± 0.0051 | 0.7218 ± 0.0054 | 0.5982 ± 0.0042 | ⚠ screening |
| 缺失注册单元 | `※` seeds/cells not independently verifiable | `※` | `※` | `※` | BLOCKED |

common-seed paired target BA relative to G0：

| Comparison | Seed-level Δ BA | Mean Δ BA ± SD | Interpretation |
|---|---|---:|---|
| G1 − G0 | `+0.0717`, `+0.0778` | `+0.0748 ± 0.0043` | 两个 common seeds 均为正，但仅为 two-seed screening |
| G2 − G0 | `−0.0057`, `−0.0745` | `−0.0401 ± 0.0486` | 负向 |
| G3 − G0 | `−0.0146`, `−0.0568` | `−0.0357 ± 0.0298` | 负向 |

**机制诊断。** 已持久化的诊断包括 head loss/accuracy/BA/AUC、GRL coefficient、shared encoder feature gradient norm、discriminator parameter gradient norm、epoch/step/best-checkpoint binding，以及 frozen-feature domain probe。可复核的 probe 摘要如下：

| Variant / seed | Frozen probe BA | Frozen probe AUC | MMD proxy | 结论 |
|---|---:|---:|---:|---|
| G0 / 42 | 0.8579 | 0.9308 | 0.3232 | reference |
| G0 / 43 | 0.8118 | 0.8859 | 0.1044 | reference |
| G1 / 42 | 0.8165 | 0.9000 | 0.2500 | probe BA/AUC 下降 |
| G1 / 43 | 0.8272 | 0.9251 | 0.2667 | 未显示同方向稳定下降 |
| G2 / 42 | 0.8606 | 0.9398 | 0.5166 | separability 未改善 |
| G2 / 43 | 0.8112 | 0.9520 | 0.8672 | separability 未改善 |
| G3 / 42 | 0.8275 | 0.9252 | 0.1987 | 部分指标下降 |
| G3 / 43 | 0.8473 | 0.9314 | 0.1571 | 不能形成稳定机制证据 |

结论：G1 在两个 common seeds 的 target BA 均高于 G0，但完整注册矩阵缺失，且 frozen probe 没有稳定、单调的跨 seed 下降。因此 G1 只能称为 **screening signal**，不能称为已验证的 domain GRL 候选。G2/G3 不支持正向采用。

#### B. NC–MCI，ADNI→NACC

**协议与标签。** 该扩展沿用 DS-038 的四种 GRL 条件，任务改为 NC vs MCI：raw diagnosis `{1: NC, 2: MCI}`，model label `{0: NC, 1: MCI}`，positive class 为 MCI。注册 seeds 为 42/43，共 8 cells。当前 output root 的 8 cells 均具备 `best.pt`、`summary.json`、`audit.json`、`config.yaml`、`predictions.json`、`status.json=complete`，因此 artifact completion 为 **8/8 ✓**；但统计功效仍只有 two-seed，整体状态为 `⚠ interim`。

官方 artifact summary（按报告原有 subject-level 指标）：

| Variant | Seeds | Target BA mean ± SD | Target AUC mean ± SD | Source-test BA mean ± SD | 状态 |
|---|---|---:|---:|---:|---|
| G0 `no_grl` | 42, 43 | 0.5537 ± 0.0167 | 0.6106 ± 0.0215 | 0.6734 ± 0.0505 | ⚠ |
| G1 `domain_only` | 42, 43 | 0.5722 ± 0.0150 | 0.6165 ± 0.0075 | 0.6403 ± 0.0395 | ⚠ |
| G2 `intensity_only` | 42, `※`43 | 0.5656 | 0.6189 | 0.5893 | `※` single-seed summary in the contemporaneous report |
| G3 `both_grl` | 42, `※`43 | 0.5825 | 0.6256 | 0.6205 | `※` single-seed summary in the contemporaneous report |

原报告记录的 common-seed paired BA：G1 seed42 `+0.0197`、seed43 `+0.0173`，均值 `+0.0185 ± 0.0017`；G2/G3 当时只有 seed42 common comparison，分别为 `+0.0237` 与 `+0.0406`，这些单 seed 差值统一标记为 `※`，不能当作稳定改善。

基于当前可读取的两 seed raw predictions 重新计算的五项指标如下。该表与官方 summary 的 BA/AUC 存在轻微口径差异，故只作为统一五指标复核：

| Variant | BA mean ± SD | AUROC mean ± SD | Macro-F1 mean ± SD | Sensitivity mean ± SD | Specificity mean ± SD |
|---|---:|---:|---:|---:|---:|
| G0 `no_grl` | 0.5538 ± 0.0188 | 0.6118 ± 0.0218 | 0.5461 ± 0.0280 | 0.2215 ± 0.0382 | 0.8862 ± 0.0005 |
| G1 `domain_only` | 0.5738 ± 0.0161 | 0.6163 ± 0.0062 | 0.5651 ± 0.0067 | 0.4861 ± 0.0668 | 0.6615 ± 0.0346 |
| G2 `intensity_only` | 0.5725 ± 0.0081 | 0.6126 ± 0.0100 | 0.5739 ± 0.0092 | 0.3848 ± 0.0214 | 0.7602 ± 0.0377 |
| G3 `both_grl` | 0.5715 ± 0.0120 | 0.6104 ± 0.0231 | 0.5725 ± 0.0139 | 0.3053 ± 0.0632 | 0.8378 ± 0.0392 |

五指标解释：G1 的 BA 增益主要来自 Sensitivity 上升，同时 Specificity 明显下降；G2 的 BA/Macro-F1 改善伴随同样的类别权衡；G3 没有稳定超过单头 variants，AUROC 也未同步改善。因此 NC–MCI 的正向结果不能仅解释为 GRL 机制成功。

**机制与审计。** 8/8 artifact 均应以完整 artifact 为准；当前报告性诊断中已记录 target-label blind、source-validation checkpoint、persisted mechanism diagnostics 和 frozen-feature probe。可直接复核的早期 interim probe 摘要显示：G1 的 probe BA/AUC 在 seed42 低于 G0、在 seed43 高于 G0；G2/G3 的机制方向曾只有单 seed 可用，统一记为 `※`。因此 NC–MCI 仍为 **two-seed interim screening**，不支持采用 domain GRL、intensity GRL 或互补性结论。

#### C. MCI–AD，NACC→ADNI

**协议偏离。** 这是方向扩展，不是原注册矩阵：NACC 3T → ADNI 1.5T，MCI vs AD，seeds 42/43，G0–G3 共 8 cells。AD 为 positive class。该方向必须与 ADNI→NACC 分开报告。

当前稳定 output root 的 contemporaneous report 只确认 G0/G3 的 4/8 cells；G1/G2 的完整 artifact 在该报告中被标记为缺失，因此本节对官方可复核性使用统一 `※`。另一个本地运行目录目前可见 8/8 个 artifact 集合，但尚未完成与 interim report 一致的 provenance/diagnostic schema 复核；这些额外文件不自动升级正式状态。

官方 interim performance snapshot：

| Variant | 官方可复核状态 | Target BA mean ± SD | Target AUC mean ± SD | Source-test BA mean ± SD | 状态 |
|---|---|---:|---:|---:|---|
| G0 `no_grl` | 42,43 | 0.6099 ± 0.0419 | 0.6528 ± 0.0386 | 0.8024 ± 0.0690 | ⚠ |
| G1 `domain_only` | `※` 历史提取，当前报告缺失 | 0.5925 ± 0.0460 | 0.6553 ± 0.0324 | 0.7948 ± 0.0494 | `※` |
| G2 `intensity_only` | `※` 历史提取，当前报告缺失 | 0.6030 ± 0.0743 | 0.6641 ± 0.1035 | 0.8014 ± 0.0205 | `※` |
| G3 `both_grl` | 42,43 | 0.5849 ± 0.0163 | 0.6267 ± 0.0557 | 0.8028 ± 0.0213 | ⚠ |

官方 paired target BA relative to G0：G1 `−0.0144, −0.0203`，mean `−0.0174 ± 0.0042`；G2 `+0.0160, −0.0299`，mean `−0.0069 ± 0.0324`；G3 `−0.0430, −0.0069`，mean `−0.0250 ± 0.0255`。其中 G1/G2 的完整可复核性统一为 `※`，历史数值不得当作当前稳定 artifact 的完整 factorial 结论。

若仅查看当前本地 predictions 的统一五指标重算快照，结果为：

| Variant | BA mean ± SD | AUROC mean ± SD | Macro-F1 mean ± SD | Sensitivity mean ± SD | Specificity mean ± SD |
|---|---:|---:|---:|---:|---:|
| G0 `no_grl` | 0.6113 ± 0.0319 | 0.6559 ± 0.0315 | 0.5957 ± 0.0464 | 0.8281 ± 0.1865 | 0.3944 ± 0.2503 |
| G1 `domain_only` | 0.6184 ± 0.0550 | 0.6792 ± 0.0603 | 0.6128 ± 0.0745 | 0.9030 ± 0.0240 | 0.3339 ± 0.1340 |
| G2 `intensity_only` | 0.6271 ± 0.0911 | 0.6780 ± 0.1025 | 0.6075 ± 0.1270 | 0.9044 ± 0.0975 | 0.3499 ± 0.2796 |
| G3 `both_grl` | 0.6221 ± 0.0340 | 0.6455 ± 0.0631 | 0.6251 ± 0.0394 | 0.8690 ± 0.0599 | 0.3753 ± 0.0081 |

上述五指标表仅是当前本地 predictions 的计算快照，与官方 interim report 的 BA/AUC 不完全一致；因此不替代官方表，也不解除 G1/G2 的 `※` 状态。方向性上，GRL variants 的 Sensitivity 较高但 Specificity 较低，表现为明显的分类阈值权衡。

**机制诊断与最终边界。** NACC→ADNI interim report 记录了：best-checkpoint-bound head records、GRL coefficient=1.0、shared encoder/discriminator gradient norms，以及 frozen-feature probe。G1 在 seed42 的 probe BA/AUC 下降、seed43 未下降；G2 的 probe 方向跨 seed 反转；G3 的 probe AUC 两 seed 下降但 MMD 混合，且 target BA 低于 G0。故当前不支持保留 domain GRL、intensity GRL 或其互补性。由于该方向的协议偏离和 artifact 复核不完整，统一标记为 `※ interim/block`。

#### D. DS-038 统一结论

1. **MCI–AD ADNI→NACC**：G1 在 common seeds 的 target BA 有正向 screening signal，但注册矩阵不完整，不能形成五-seed factorial 结论。
2. **NC–MCI ADNI→NACC**：8/8 artifact 已存在，但只有两 seed；G1/G2/G3 的 BA 增益伴随 Sensitivity/Specificity trade-off，不能据此采用 GRL。
3. **MCI–AD NACC→ADNI**：方向和 seed 数均为扩展协议；G1/G2 的正式可复核状态用 `※` 表示，已有 paired 结果不支持稳定 GRL 增益。
4. 三组实验均不支持 scanner、manufacturer 或 field-strength 的因果解释，也不支持把不同任务/方向合并为一个 DS-038 总体效果。
5. 在所有 `※` 单元恢复并完成统一 schema、provenance、checkpoint binding 和 paired statistics 之前，DS-038 总体状态保持 **BLOCKED / INTERIM**。

## 四、审计与统一解释边界

### Target-label 与 checkpoint 审计

已完成并有运行时审计字段的实验均记录：

- target labels 未用于 training；
- target labels 未用于 checkpoint selection；
- checkpoint 由 source-validation BA（及适用的 collapse guard）选择；
- target-test 在 checkpoint 冻结后读取；
- source/target subject split 保持 disjoint；
- 运行级别保留 configuration hash、dataset hash 或 subject digest 的实验，才能达到相应的可审计等级。

这些审计只能证明 target-label boundary 没有被直接突破，不能消除历史 target holdout 重复使用带来的 confirmatory 限制。

### 不能支持的主张

当前 DS-034 至 DS-038 结果不能支持：

- “模型学习了 scanner 或 field-strength 的因果响应”；
- “从 NACC 3T 学到了普遍的 ADNI 1.5T 校正”；
- “某个 GRL head 在所有方向、任务和 seed 上稳定有效”；
- “双向平均证明方法对所有未见协议均有效”；
- “target BA 的均值优势自动等于机制成功”。

## 五、正向增益实验的五项分类指标复核

为避免仅依据 target BA 判断“正向增益”，对可读取的原始 `predictions.json` 进行了统一复核。除特别说明外，采用以下口径：同一 subject 的多次扫描先对预测概率取均值；以 `probability >= 0.5` 转换为类别；跨 seed 报告均值 ± 样本标准差（SD）；不合并不同任务、方向或数据集。五项常用指标为：

1. **Balanced Accuracy（BA）**：`(Sensitivity + Specificity) / 2`，本项目的主指标，适合类别不均衡场景。
2. **AUROC**：衡量模型对正负样本的整体排序能力，不依赖单一分类阈值。
3. **Macro-F1**：正类与负类 F1 的平均值，用于同时考察两类的识别质量。
4. **Sensitivity（Recall / TPR）**：`TP / (TP + FN)`，表示正类被正确识别的比例。
5. **Specificity（TNR）**：`TN / (TN + FP)`，表示负类被正确识别的比例。

### DS-035：FMM 正向增益

DS-035 为 MCI–AD、ADNI→NACC、seeds 42–44。以下为从原始 target predictions 按 subject 聚合后重算的结果：

| Variant | BA mean ± SD | AUROC mean ± SD | Macro-F1 mean ± SD | Sensitivity mean ± SD | Specificity mean ± SD |
|---|---:|---:|---:|---:|---:|
| B0-ref | 0.5769 ± 0.0417 | 0.6952 ± 0.0465 | 0.5421 ± 0.0707 | 0.2841 ± 0.2853 | 0.8696 ± 0.2090 |
| B1-fmm | 0.6069 ± 0.0192 | 0.7805 ± 0.0174 | 0.5919 ± 0.0214 | 0.2641 ± 0.0642 | 0.9498 ± 0.0321 |
| B1a-no-source-fft | 0.6120 ± 0.0701 | 0.7821 ± 0.0581 | 0.5909 ± 0.1038 | 0.2804 ± 0.1661 | 0.9435 ± 0.0413 |

相对 B0，B1 的重算结果在 BA、AUROC、Macro-F1 和 Specificity 上为正，但 Sensitivity 略低；因此其增益主要体现为整体排序能力和负类识别改善，而不是两类 Recall 同步改善。B1a 的均值也为正，但跨 seed 波动更大，不能作为稳定替代方案。

官方 DS-035 汇总报告中的 B1 target BA 为 `0.6131 ± 0.0166`，本节按原始预测重新计算的结果为 `0.6069 ± 0.0192`。该差异说明官方 summary 与本次 subject 聚合/汇总口径并不完全相同；正式结论应保留官方产物值，并将本节作为统一五指标的独立复核，不应混合两种口径进行统计推断。

### DS-034：C4 full frequency UDA

C4 相对 C0 的 ADNI→NACC、MCI–AD、seeds 43/44 描述性结果为：

| Variant | BA mean ± SD | AUROC mean ± SD | Macro-F1 mean ± SD | Sensitivity mean ± SD | Specificity mean ± SD |
|---|---:|---:|---:|---:|---:|
| C0 baseline | 0.670 ± 0.042 | 0.784 ± 0.002 | 0.678 ± 0.042 | 0.428 ± 0.123 | 0.911 ± 0.038 |
| C4 full UDA | 0.693 ± 0.051 | 0.825 ± 0.009 | 0.703 ± 0.048 | 0.469 ± 0.165 | 0.916 ± 0.063 |
| C4 − C0 | +0.023 | +0.040 | +0.025 | +0.041 | +0.005 |

五项指标的点估计均为正，但 paired bootstrap uncertainty 跨 0；因此 C4 只能描述为正向点估计，不能升级为稳定增益结论。另有 ECE 变差，说明分类性能点估计改善不等于概率校准改善。

### DS-038：NC–MCI 正向筛查

NC–MCI、ADNI→NACC、seeds 42/43 的原始 predictions 复核如下：

| Variant | BA mean ± SD | AUROC mean ± SD | Macro-F1 mean ± SD | Sensitivity mean ± SD | Specificity mean ± SD |
|---|---:|---:|---:|---:|---:|
| G0 no-GRL | 0.5538 ± 0.0188 | 0.6118 ± 0.0218 | 0.5461 ± 0.0280 | 0.2215 ± 0.0382 | 0.8862 ± 0.0005 |
| G1 domain-only | 0.5738 ± 0.0161 | 0.6163 ± 0.0062 | 0.5651 ± 0.0067 | 0.4861 ± 0.0668 | 0.6615 ± 0.0346 |
| G2 intensity-only | 0.5725 ± 0.0081 | 0.6126 ± 0.0100 | 0.5739 ± 0.0092 | 0.3848 ± 0.0214 | 0.7602 ± 0.0377 |
| G3 both-GRL | 0.5715 ± 0.0120 | 0.6104 ± 0.0231 | 0.5725 ± 0.0139 | 0.3053 ± 0.0632 | 0.8378 ± 0.0392 |

G1 的 BA 增益主要来自 Sensitivity 上升，同时 Specificity 明显下降；G2 的 BA 与 Macro-F1 提升也伴随 Sensitivity/Specificity 权衡；G3 没有稳定超过单头 variants，AUROC 亦未显示同步改善。因此这些结果更接近 threshold trade-off，而不是所有分类质量维度共同改善。

### 五指标综合判断

- **较完整的正向信号：DS-035 B1。** BA、AUROC、Macro-F1 和 Specificity 同时改善，但 Sensitivity 没有改善；应称为探索性 FMM 信号，而不是全面分类性能提升。
- **点估计正向但不稳定：DS-035 B1a、DS-034 C4。** B1a 的 seed 波动较大，C4 的 paired uncertainty 跨 0，均不满足直接采用条件。
- **存在类别权衡：DS-038 NC–MCI G1/G2/G3。** BA 上升伴随 Sensitivity 与 Specificity 此消彼长，不能只依据 BA 均值宣称 GRL 有效。

后续实验报告应固定同时呈现 `BA + AUROC + Macro-F1 + Sensitivity + Specificity`，并附 seed-level 数值、mean ± SD、相对 matched baseline 的 paired difference，以及 Sensitivity/Specificity 是否发生方向性偏移。

## 五、推荐后续动作

1. 若继续 DS-038，应先恢复或重跑缺失 artifacts，并在共同 seeds 上重新计算 paired statistics。
2. 机制报告应优先检查 best-checkpoint 的 head diagnostics、encoder/discriminator gradient norm、feature discrepancy 和 frozen probe，而不是只重复查看 target BA。
3. 若要形成 confirmatory claim，应使用未被历史实验使用过的 target holdout，并在读取 target-test 之前锁定 seed、变体、checkpoint selector 和分析规则。
4. NACC→ADNI 结果应始终单独报告为 unsupported-protocol stress direction，不与 ADNI→NACC 做无条件平均。

## 六、主要证据来源

- DS-034：`docs/results/DS-034/`
- DS-035：`docs/results/DS-035/README.md`、`docs/results/DS-035/3SEED_ABLATION_2026-08-19.md`
- DS-036：`docs/results/DS-036/README.md`、`docs/results/DS-036/5SEED_COMPARISON_2026-08-19.md`
- DS-037：`docs/results/DS-037/README.md`
- DS-038 ADNI→NACC：`docs/results/DS-038/README.md`
- DS-038 NC–MCI：`docs/results/DS-038_NC_MCI_2SEED.md`
- DS-038 MCI–AD NACC→ADNI：`review/analysis/39_ds038_mci_ad_nacc_to_adni_mci_ad_2seed_2026-08-24.md`
- 数据现实与主张边界：`docs/SCAN_AWARE_DATA_REALITY_AND_CLAIM_BOUNDARY.md`
