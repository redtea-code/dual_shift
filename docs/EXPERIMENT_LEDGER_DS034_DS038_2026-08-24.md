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

### DS-038：domain/intensity GRL factorial

DS-038 的设计是将 B1c 的联合 GRL 消融拆成四种互相独立的条件：

- G0：domain 和 intensity GRL 均关闭；
- G1：仅 domain GRL；
- G2：仅 intensity GRL；
- G3：两个 GRL 均打开。

诊断实现目前能够保存：

- 每个 head 的 loss、accuracy、balanced accuracy、AUC；
- GRL coefficient；
- shared encoder feature gradient norm；
- discriminator parameter gradient norm；
- epoch、step、`is_best_checkpoint`；
- frozen-feature domain probe 的 BA、AUC、MMD proxy 和 feature norm。

#### MCI–AD，ADNI→NACC

原注册矩阵为 G0–G3 × seeds42–46，共 20 个 cells，目前仅 9/20 可验证。common seeds42/43 的 screening 结果为：

| Variant | Target BA mean ± SD | Δ BA vs G0 |
|---|---:|---:|
| G0 | 0.5853 ± 0.0350 | — |
| G1 | 0.6601 ± 0.0392 | +0.0748 ± 0.0043 |
| G2 | 0.5452 ± 0.0136 | −0.0401 ± 0.0486 |
| G3 | 0.5496 ± 0.0051 | −0.0357 ± 0.0298 |

G1 的 target BA 在两个 common seeds 均高于 G0，但完整注册矩阵缺失，且 frozen probe 没有显示跨 seed 的稳定单调降低。因此当前只能称为 screening signal，不能称为 domain GRL 候选已被验证。

#### NC–MCI，ADNI→NACC

该扩展使用 seeds42/43、8 个 cells，label contract 为 `{1:NC, 2:MCI}` → `{0:NC, 1:MCI}`，positive class 为 MCI。当前八个 artifact 均已存在：

| Variant | Target BA mean ± SD | Δ BA vs G0 |
|---|---:|---:|
| G0 | 0.5537 ± 0.0167 | — |
| G1 | 0.5722 ± 0.0150 | +0.0185 ± 0.0017 |
| G2 | 0.5740 ± 0.0119 | +0.0203 ± 0.0048 |
| G3 | 0.5742 ± 0.0117 | +0.0205 ± 0.0284 |

这些数字显示两 seed 下四种设置的 target BA 差异都较小。G3 没有稳定超过单头 variants；AUC 差异也不一致，故不能支持两个 GRL head 的互补性或采用结论。

#### MCI–AD，NACC→ADNI

这是一个方向扩展，不应与 ADNI→NACC 汇总。当前稳定 output root 只能验证 seed42/43 的 G0/G3，共 4/8 个 cells。历史 extraction snapshot 中记录的相对 G0 差异为：G1 `−0.0174`、G2 `−0.0069`、G3 `−0.0250`，但 G1/G2 缺少当前可复核 artifacts，因此报告必须保持 interim/block 状态。

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
