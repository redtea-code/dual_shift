# APIS v2 Scan 参数泛化验证计划

日期：2026-08-03  
状态：提案；只有通过 Phase 0 数据门控后才启动确认性训练  
主问题：scan/acquisition 参数能否在不泄漏诊断信息的前提下，稳定提高跨队列 subject-level balanced accuracy？

## 1. 现有证据与决策

当前实现已经具备 source-train-only 的 `AcquisitionDescriptorEncoder`、缺失值掩码、观测协议原型和 acquisition shuffle 负对照接口。现有实验不支持把随机风格混合视为充分答案：MCI vs AD 的 target subject-mean BA 中，MixStyle 相对 ce_only 在已完成配对 seed 上没有一致方向。

| 任务 | 方向 | 可配对 seed | MixStyle - ce_only 的平均 ΔBA | 当前判断 |
|---|---|---:|---:|---|
| MCI vs AD | ADNI->NACC | 42, 43 | -0.0077 | 无正向趋势 |
| MCI vs AD | NACC->ADNI | 42, 43 | +0.0139 | 小幅、未证实 |

同时，3090 的场强表中 NACC->ADNI 有 `field_strength=unknown`、`n=1` 的记录。该层的样本数过小，不能用于机制或泛化结论。下一步必须先回答“scan 参数是否确实定义了可校正的 domain shift”，而不是继续调大 APIS 的随机干预强度。

## 2. 固定原则

1. 诊断标签不得用于 target 域训练、协议编码器拟合、归一化统计或阈值选择。
2. acquisition 词表、连续变量均值/标准差和协议原型只在 source-train 拟合。
3. 诊断主终点固定为 target `subject_mean` balanced accuracy；scan-level 仅为补充。
4. 所有变体共享 backbone、图像预处理、6/2/2 subject split、epoch、优化器、seed 和 source-only checkpoint 选择规则。
5. 每一个正向 scan-aware 结论必须同时有 acquisition shuffle 负对照；否则只能说明“模型变复杂后性能变化”。

## 3. Phase 0：scan 数据可用性与混杂审计

### 3.1 产物

每个任务和方向生成 `scan_parameter_audit_<task>.csv` 与一份 Markdown 摘要，至少包含：

- `manufacturer`、`scanner_model`、`field_strength`、`sequence_family`、`tr_ms`、`te_ms`、`ti_ms`、`flip_angle`、`slice_thickness`、`pixel_spacing_x/y`、`acceleration`；
- 各队列、类别、split 的非缺失数、缺失率、分位数/类别频数；
- 单 subject 多 scan 是否参数冲突，以及采用的 earliest-visit/scan 聚合规则；
- `unknown`、`Other`、`Missing` 的来源和数量；
- manifest 路径、SHA-256、生成 commit 和匹配质量（`exact_image`、`same_day_inferred`、`site_fallback`）。

### 3.2 必过门槛

- pooled target 的场强分层样本数必须可解释；`unknown` 不能静默进入 1.5T/3T 对比。
- 任一 field-strength 单元 `n_subjects < 20` 时，只报告样本数，不计算或解释 BA/AUC。
- 所有连续参数必须确认单位一致；例如 TR/TE 不可混合秒与毫秒。
- source-train 之外的数据不得影响 acquisition vocab、归一化或 prototype bank。

未通过时：先修 manifest/指标导出，不训练任何新模型。

## 4. Phase 1：证明 scan 参数携带何种信息

在每个方向仅用 source-train 拟合，source-val 调参；target 只评估。

| 探针 | 输入 | 预测目标 | 目的 | 解释 |
|---|---|---|---|---|
| Domain probe | scan 参数 + 缺失掩码 | 队列/环境 | 判断 acquisition shift 是否存在 | AUC 高才值得做条件校正 |
| Label probe | scan 参数 + 缺失掩码 | 诊断标签 | 检查参数-诊断混杂 | AUC 高是风险信号，不是模型优势 |
| Image domain probe | 图像特征 | 队列/环境 | 检查图像中剩余域信息 | 与 scan probe 对照 |
| Completeness probe | 缺失掩码 | 队列/诊断 | 检查“是否缺失”本身的泄漏 | 高性能需限制或分层 |

决策：若 domain probe 接近随机，则停止 scan-aware 主线；若 label/completeness probe 很高，则禁止直接 concat，并采用受约束调制及更强负对照。

## 5. Phase 2：小规模机制筛选

先用 seed 42、43，在两个任务和两个方向上完成以下最小矩阵；不改主终点。

| variant | 作用 | scan 参数使用方式 |
|---|---|---|
| `ce_only` | 图像基线 | 不使用 |
| `mixstyle` | protocol-unaware 基线 | 不使用 |
| `metadata` | 直接拼接负/弱基线 | 图像 GAP 与 acquisition concat |
| `apis_v2` | 当前观测协议残差干预 | encoder + prototype residual |
| `apis_v2_shuffle` | 关键负对照 | 同 APIS，但 batch 内置乱 acquisition embedding |
| `film_scan` | 条件调制基线 | 由 acquisition embedding 产生有界 FiLM |
| `apis_scan` | 候选方法 | APIS residual + 有界 scan FiLM，source-only |

`film_scan` 和 `apis_scan` 的调制必须满足：

```text
gamma = 1 + alpha * tanh(gamma_raw)
beta  = alpha * tanh(beta_raw)
alpha <= 0.1（筛选阶段固定）
```

这避免模型把 acquisition embedding 当成直接诊断捷径。`apis_scan` 只有在相对 `apis_v2_shuffle` 有一致收益时，才可解释为 scan 条件的有效贡献。

## 6. Phase 3：确认性矩阵与声明门槛

只有 Phase 0/1/2 满足预先条件时，冻结模型与配置，运行 seed 42--46 的完整矩阵：

```text
CN_vs_AD, MCI_vs_AD
× ADNI_to_NACC, NACC_to_ADNI
× ce_only, mixstyle, apis_v2, apis_v2_shuffle, film_scan, apis_scan
```

每个比较按相同 subject、相同 seed、相同方向配对。输出规范长表和配对差值表。

### 6.1 成功条件

“有效”仅在某一 task×direction 满足以下全部条件时成立：

1. APIS-Scan 相对 ce_only 与 MixStyle 的 ΔBA 均大于 0；
2. subject-level paired bootstrap 95% CI 不含 0；
3. 相对 `apis_v2_shuffle` 的 ΔBA 同方向；
4. AUC、ECE 和任一类别 sensitivity/specificity 未出现预注册的灾难性退化。

“普遍提高”仅在四个 task×direction 单元中至少三个满足“有效”，且第四个单元的 paired ΔBA CI 不支持实质性负向退化时成立。若仅 CN vs AD 通过，主张必须限定为该任务，不能外推到 MCI vs AD。

## 7. 推荐执行顺序

1. 修复 `unknown` 场强记录并生成 Phase 0 审计表。
2. 运行 domain/label/completeness probes，决定是否允许 direct-concat 参加后续对照。
3. 实现或启用 `film_scan`、`apis_v2_shuffle`、`apis_scan`，只跑 seed 42/43 机制筛选。
4. 由预先定义的筛选结果冻结一个候选方法；不得查看 seed 44--46 target 后继续改结构。
5. 执行完整确认性矩阵、paired bootstrap 和规范主表导出。

## 8. 禁止事项

- 不得以 target BA 选择 scan 参数、调制强度或 checkpoint；
- 不得删除失败 seed、`unknown` 场强或不利方向后再计算均值；
- 不得把 AUC 的局部提升替代 BA 主终点；
- 不得把 metadata concat 基线的塌缩当作 APIS 机制成功；
- 不得把内部队列分层结果直接称为“跨队列泛化改善”。

