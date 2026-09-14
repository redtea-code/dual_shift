# DS-044 方向感知的退化建模与任务表征保护实验计划

> 实验编号：DS-044
> 计划类型：方向感知域适应机制验证
> 主要方向：NACC → ADNI  1.5T 目标域压力测试
> 对照方向：ADNI → NACC  暂不扩展，仅保留为未来方向
> 任务：scan-filtered MCI vs AD 二分类
> 状态：D0–D3 COMPLETE / NEXT MECHANISM PLAN REGISTERED
> 前置实验：DS-043 CAPM-GRL 多模态拼接与频域/残差对抗基线

## 1. 研究背景与问题

DS-043 显示，当前 CAPM-GRL 方法在 ADNI→NACC 方向上具有一定收益，但在 NACC→ADNI 方向上没有复现该收益，部分 GRL 变体出现负迁移。一个待验证的解释是：NACC 的 3T MRI 可能保留更多高频和局部结构信息，而 ADNI 的 1.5T MRI 具有更明显的信息受限或退化特征。

在这种情况下，对完整特征使用 GRL 可能同时削弱：

1. 与 scanner/acquisition 有关的非任务信息；
2. 对 MCI/AD 判别有用的高频或局部结构信息。

因此，DS-044 不把 NACC→ADNI 视为 ADNI→NACC 的简单反向复制，而是验证一种方向感知的适配策略：

> 对 source 进行显式的 target-like 退化，对低频采集统计进行选择性对齐，同时保护高频任务表征，并使用受限的非对称 adapter 控制负迁移。

## 2. 研究假设

> **DS-044 第一轮结果更新（2026-09-08）**：D0–D3 已完成 15/15 个运行，但未满足预注册的联合成功标准。D0–D3 保留为已完成的历史机制检验；以下新增假设构成后续唯一主线。ADNI→NACC 暂不扩展。

### H5：Prior/calibration/conditional-shift 假设

DS-044 第一轮中，D1 的 AUROC 有所上升，但 BA、F1 和 MCC 没有同步改善；D3 改变了 sensitivity/specificity，却进一步降低 BA/MCC。这种模式更符合 **label-prior shift、decision calibration drift 和 label-conditional shift 的混合影响**，而不是单一的全局图像退化：

```text
P_T(y) != P_S(y)
P_T(t) != P_S(t)
P_T(y | x, t) != P_S(y | x, t)
```

因此，后续不再默认 `x_T ~= D(x_S)`，而按复杂度递增验证：

```text
P0
 └── P0-Prior   (prior-aware logit correction)
       └── P0-Cal     (source-registered calibration correction)
             └── P0-CShift   (uncertainty-gated class-conditional alignment)
```

该假设可被证伪：如果 prior correction、calibration correction 和 uncertainty-gated conditional alignment 均不能在至少 2/3 seeds 上稳定改善 target BA/F1/MCC，则停止新增适配模块，不再继续堆叠 adapter 或频率分支。


### H1：Source degradation 假设

将 NACC 3T source 图像退化为 ADNI 1.5T-like 视图，并让模型在退化视图上进行监督训练，可以减少 source/target 信息结构不匹配。

```text
NACC 3T source -> 1.5T-like degradation -> source supervised training
```

### H2：选择性对齐假设

NACC→ADNI 的主要可适配差异更可能位于低频强度、粗粒度空间结构和采集统计，而不是全部高层诊断特征。因此，低频选择性对齐应比 full-feature GRL 更安全。

### H3：任务表征保护假设

高频和局部结构可能携带疾病判别信息。对高频分量进行强域对齐会增加负迁移风险；保留 source task loss、logit consistency 和 feature preservation 可以改善反向迁移稳定性。

### H4：非对称适配假设

NACC→ADNI 更接近“信息丰富 source → 信息受限 target”的单向变化。方向性 bounded adapter 比强制两个域在完整特征空间中完全靠拢更适合该过程。

### 2.1 对原 H1–H4 的状态判定

- **H1（source degradation）**：D0 未稳定优于 P0；不支持当前固定退化组合可稳定缓解方向差异。
- **H2（低频选择性对齐）**：D1 仅提高部分排序指标，未改善联合任务指标；不支持其作为独立有效机制。
- **H3（高频任务表征保护）**：D2 的 F1 有局部改善，但 BA/MCC 下降；不支持稳定保护收益。
- **H4（bounded asymmetric adapter）**：D3 的 BA/MCC 低于 P0；不支持继续增加 adapter 容量。

这些判定是“当前实现和协议下未获支持”，不是证明相应机制在所有实现中不可能有效。

## 3. 总体实验原则

- 使用与 DS-043 一致的 scan-filtered manifest、任务定义、subject split、训练预算和 checkpoint 协议。
- 正式 seeds 固定为 `42、43、44`。
- 当前只注册 NACC→ADNI；ADNI→NACC 不属于本轮任务，不运行、不调参、不与 NACC→ADNI 结果合并排名。
- `T_adapt` 只提供无标签 target MRI；target label 只能在冻结模型后的最终评估阶段读取。
- source validation 可用于 checkpoint 和超参数规则选择；不得使用 target-test label、prediction 或 metric 调参。
- 第一阶段不扩展 table concat 变体，table 仅按既有 CAPM contract 使用。
- 所有新增模块必须有独立 ablation，避免同时改变 degradation、alignment 和 adapter 后无法解释增益来源。

## 4. 输入与数据合同

### 4.1 数据集合

| 集合 | NACC→ADNI 可访问信息 | 用途 |
|---|---|---|
| `S_train` | NACC 3T MRI、table、source label | 监督训练与 source task preservation |
| `S_val` | NACC 3T MRI、table、source label | checkpoint、超参数和停止规则 |
| `S_test` | NACC 3T MRI、table、source label | 冻结模型后的 source 报告 |
| `T_adapt` | 无标签 ADNI 1.5T MRI，必要时使用无标签统计 | 低频统计/无监督适配 |
| `T_test` | ADNI 1.5T MRI、table，最终读取 label | 冻结模型后的 target 评估 |

所有集合按 subject 划分，纵向 scan 不得跨集合。`T_adapt` 和 `T_test` 必须 subject-disjoint。

### 4.2 图像输入

```text
x_s^3T       [B, 1, 160, 196, 160]   原始 NACC source
x_s^deg      [B, 1, 160, 196, 160]   退化后的 source view
x_t^1.5T     [B, 1, 160, 196, 160]   无标签 ADNI target view
```

table contract 沿用 DS-043：

```text
t = [age, sex, education]
m = [age_missing, sex_missing, education_missing]
```

DS-044 不加入 site、scanner、field strength、manufacturer、sequence 或 diagnosis 作为模型输入。

## 5. 核心数据流

### 5.1 完整方向感知数据流

```text
NACC source x_s^3T --------------------+
                                       |
        +--> degradation D(x_s^3T) ---+--> shared CAPM task encoder --> h
        |                                                            |
        |                                     +----------------------+------------------+
        |                                     |                                         |
ADNI target x_t^1.5T --> shared encoder --> h_low / h_high                         table
                                              |         |                              |
                                      low-frequency   high-frequency              CAPM condition
                                      alignment       task preservation                  |
                                              |         |                              |
                                              +----+----+------------------------------+
                                                   |
                                      direction-specific bounded adapter
                                                   |
                                             classifier -> y_hat
```

### 5.2 CAPM 特征分解

```text
x -> ResNet/CAPM -> h [B, 512, D, H, W]
                   |
                   +--> h_low  = LowPass(h)
                   +--> h_high = h - h_low
```

第一版中，`LowPass` 可以使用与输入空间一致的固定 Gaussian/average low-pass 操作。低频 cutoff、kernel 和 degradation strength 必须在 source validation 或预注册规则中确定，不能使用 target-test label 选择。

### 5.3 NACC source degradation

```text
x_s^3T -> D(x_s^3T) = x_s^deg
```

退化模块按单因素增量设计：

1. `D_blur`：Gaussian blur / low-pass；
2. `D_resample`：下采样后上采样，模拟空间细节损失；
3. `D_highfreq`：高频幅度衰减；
4. `D_noise`：轻度 SNR 和 intensity variation。

第一轮只允许一个固定退化组合，避免把多个变换的效果误判为方法收益。所有退化操作保持标签不变，不改变 subject split。

## 6. 实验矩阵

### 6.1 NACC→ADNI 第一轮历史矩阵（已完成）


| 实验 | 组成 | 目的 |
|---|---|---|
| `P0` | 原始 NACC 3T source-only | 反向方向基线 |
| `D0` | `P0 + CE(f(D(x_s)), y_s)` | 验证 source degradation 是否有帮助 |
| `D1` | `D0 + low-frequency alignment` | 验证选择性低频对齐 |
| `D2` | `D1 + high-frequency task preservation` | 验证保护高频任务表征 |
| `D3` | `D2 + bounded asymmetric adapter` | 验证方向性受限 adapter |

### 6.2 后续主矩阵：NACC→ADNI 校正与条件适配

| 实验 | 组成 | 进入条件 | 目的 |
|---|---|---|---|
| `P0-Prior` | P0 + label-prior / prior-aware logit correction | 直接运行；不读 target-test label | 检验 class-prior shift 是否解释 BA/F1 漂移 |
| `P0-Cal` | P0 + source-registered temperature/threshold calibration | P0-Prior 后；规则只由 source validation 注册 | 检验 ranking 尚可但 decision calibration 漂移 |
| `P0-CShift` | P0 + uncertainty-gated class-conditional alignment | 仅在前两步不能解释现象时运行 | 检验 label-conditional representation shift |

`P0-Prior` 使用无标签 `T_adapt` 的预测分布估计 prior 修正；`P0-Cal` 的 temperature/threshold 只能由 source train/validation 规则确定，不能用 target-test 指标选取；`P0-CShift` 只允许高置信 target pseudo-label 参与 class-conditional alignment，低置信样本被排除。

### 6.3 ADNI→NACC 方向（deferred）

ADNI→NACC 的 C0–C3 仅作为未来方向性对照的候选，不属于当前 DS-044 注册任务。本轮不运行、不使用其结果选择 NACC→ADNI 模块，也不把 DS-043 正向结果与本轮结果合并排名。

## 7. 后续新主线的数据流与损失

### 7.1 `P0-Prior`：label-prior / prior-aware correction

冻结 P0 encoder/classifier，在 `T_adapt` 上只读取 MRI 和允许使用的无标签 table，得到 target prediction distribution。使用 source validation 注册的 correction rule 对 logits 进行 prior-aware 调整：

```text
z'_k = z_k + log(pi_T(k) / pi_S(k))
```

其中 `pi_T` 的估计规则、平滑项和最大修正幅度必须预先注册；不得用 `T_test` label、BA、F1 或 MCC 选择。

### 7.2 `P0-Cal`：source-registered calibration correction

在 source validation 上拟合 temperature scaling 或固定 threshold rule，然后冻结规则并应用于 target。该实验的目的不是用 target label 调 threshold，而是判断 D1 中“AUROC 上升但 BA/F1 下降”是否属于 decision calibration drift。报告 AUROC、BA、F1、MCC、Brier score、ECE 和 positive prediction rate。

### 7.3 `P0-CShift`：uncertainty-gated class-conditional alignment

仅当 prior/calibration correction 不能解释结果时运行。冻结或近似冻结 P0 task path，按 source-registered confidence threshold 选择 target pseudo-label：

```text
T_adapt -> P0 -> confidence gate -> pseudo-label y_hat_t
                         |
                         +--> class-conditional prototype/alignment loss
                         |
                 low-confidence samples: no alignment
```

只允许高置信样本参与同类 prototype 或 conditional alignment；pseudo-label coverage、confidence threshold、alignment weight 和停止规则均由 source-only simulation 或 source validation 确定。该模块不得读取 target-test label。

### 7.4 新主线的复杂度与停止顺序

```text
P0 -> P0-Prior -> P0-Cal -> P0-CShift
```

每一步先完成 3 seeds 的最小实验并检查预注册标准；不满足条件时，不因均值中单一指标改善而继续增加模型复杂度。

## 8. 原方向感知数据流与损失（历史 D0–D3）

### 8.1 `P0`：原始 source-only

```text
x_s^3T, t_s -> CAPM -> h_s -> classifier -> CE(y_s)
x_t^1.5T, t_t -> CAPM -> h_t -> final target evaluation only
```

### 8.2 `D0`：退化 source 监督训练

```text
x_s^3T ------------------------------> h_s -> classifier -> CE(y_s)
D(x_s^3T) = x_s^deg -----------------> h_deg -> classifier -> CE(y_s)
```

```text
L_D0 = CE(f(x_s^3T), y_s)
     + CE(f(D(x_s^3T)), y_s)
```

`D0` 不加入 target label，不使用 target-test 预测选择参数。

### 8.3 `D1`：低频选择性对齐

```text
h_s^deg -> h_s,low  ------------------+
                                      | low-frequency alignment
h_t     -> h_t,low  ------------------+

h_s,high / h_t,high -> task path, no strong full-feature GRL
```

```text
L_D1 = L_D0
     + λ_low L_align(h_s,low, h_t,low)
```

`L_align` 第一版可选 CORAL、linear-kernel MMD 或低频域判别器，但一次实验只使用一种对齐机制。建议优先使用稳定、可审计的低频统计对齐，而不是直接对完整特征使用 GRL。

### 8.4 `D2`：高频任务表征保护

```text
h_s,high -> source task / preservation
h_s,high -> degraded-source consistency
h_t,high -> target augmentation consistency only
```

```text
L_preserve = ||logits(x_s^3T) - logits(D(x_s^3T))||_2^2
           + ||h_s,high - h_s,high^deg||_2^2
```

```text
L_D2 = L_D1 + λ_preserve L_preserve
```

高频分量不承担强 domain confusion。其主要约束是 source label、原始/退化视图 consistency 和有限的 feature preservation。

### 8.5 `D3`：非对称 bounded adapter

```text
h -> Adapter_NACC_to_ADNI(h) = Δh
h_adapted = h + α Δh
```

```text
L_adapter = λ_delta ||Δh||_2^2
           + λ_logit ||f(h_adapted) - f(h)||_2^2
```

完整损失：

```text
L_D3 = L_D2 + L_adapter
```

adapter 约束：

- 初始接近 identity；
- `α` 固定为小值或由 source validation 选择；
- 第一阶段优先冻结 backbone，仅训练 adapter 和必要的浅层统计模块；
- source validation 性能明显下降时，降低 `α` 或停止适配；
- adapter 不读取 target label，不使用 target-test metric 选择。

## 9. 双方向统一框架（未来方向，不属于当前注册任务）

DS-044 的长期目标不是让两个方向使用完全相同的损失权重，而是共享任务保护原则、数据合同和评价协议，同时允许方向使用不同适配机制。

| 方向 | shift 假设 | 推荐适配 | 高频处理 |
|---|---|---|---|
| ADNI→NACC | 采集风格/域差异可能占主导 | 可保留 F1/F3 作为 baseline，增加低频保护版本 | 弱保护或轻度对齐 |
| NACC→ADNI | 3T source 到 1.5T target 的信息退化 | source degradation + low-frequency alignment + bounded adapter | 强任务保护，不做 full-feature GRL |

统一框架：

```text
shared task encoder
    -> low-frequency alignment branch
    -> high-frequency task-preservation branch
    -> direction-specific bounded adapter
    -> classifier
```

方向本身是预先给定的实验条件，不包含 target diagnosis，因此方向专用 adapter 不违反 label-blind UDA。

## 10. 成功标准与失败判定

### 10.1 主任务指标

每个变体、每个 seed、每个方向均报告：

- AUROC；
- balanced accuracy；
- accuracy；
- Precision；
- Sensitivity/Recall；
- Specificity；
- F1-score；
- MCC。

同时记录：

- source validation/test preservation；
- adaptation magnitude；
- low-frequency discrepancy；
- high-frequency energy/statistics；
- original/degraded source consistency；
- target augmentation consistency。

### 10.2 预注册成功标准

一个新增机制只有同时满足以下条件，才认为具有初步支持：

```text
target AUROC 或 BA 改善
+ F1 和 MCC 不下降
+ source performance 没有明显下降
+ 至少 2/3 seeds 方向一致
+ adaptation magnitude 没有异常增大
```

如果只有 AUROC 上升而 BA、F1 或 MCC 下降，应标记为 threshold/calibration shift，不能判定为稳定收益。

### 10.3 机制判定

- `D0 > P0`：支持 source degradation 假设；
- `D1 > D0`：支持低频选择性对齐；
- `D2 > D1`：支持高频任务表征保护；
- `D3 > D2`：支持 bounded asymmetric adapter；
- 若 `D3` 仍不优于 `P0`，则不能继续声称 3T→1.5T 退化是主要原因，应转向检查 cohort shift、label-conditional shift 和数据协议差异。

## 11. 必要诊断与数据分析

在解释模型结果前，必须对两个 cohort 进行独立诊断：

### 图像/频率诊断

- 原始和预处理后 voxel intensity 分布；
- low-frequency energy；
- high-frequency energy；
- gradient magnitude；
- blur/SNR proxy；
- feature covariance；
- 原始与退化 source 的频谱差异。

### cohort 与任务诊断

- MCI/AD 类别比例；
- age、sex、education 分布；
- 缺失模式；
- 每个 split 的 subject 数量；
- 每个 subject 的 scan 数量；
- diagnosis/date matching 和排除数量；
- 预处理、分辨率和 manifest 版本。

“3T 更清晰、1.5T 信息更少”只能作为待检验假设，不能在没有这些诊断前写成因果结论。

## 12. 协议边界

明确禁止：

- 用 ADNI target-test label 选择 degradation strength；
- 用 target-test BA/AUROC 选择 adapter strength、cutoff 或 checkpoint；
- 以 domain discriminator loss 下降单独证明 batch/scanner effect 被消除；
- 将 high-frequency branch 直接命名为纯 biological representation；
- 将 residual/adapter branch 直接命名为纯 batch representation；
- 把 NACC→ADNI stress-test 结果与 ADNI→NACC 主结果未经分层地合并排名；
- 在第一轮同时改变 backbone、degradation、alignment、selector 和 table 输入。

## 13. 建议实施顺序（更新版）

```text
已完成：Phase A: 诊断 NACC/ADNI 的频率、协变量和类别差异
    |
    v
已完成：Phase B–D: P0 -> D0 -> D1 -> D2 -> D3
    |
    v
下一阶段：P0-Prior -> P0-Cal -> P0-CShift（仅 NACC→ADNI，逐步闸门）
    |
    v
停止条件：三类校正均失败后停止新增适配模块；ADNI→NACC 延后
```

当前后续只运行：

```text
NACC→ADNI: P0-Prior -> P0-Cal -> P0-CShift
ADNI→NACC: deferred
```

## 14. 预期贡献与结论边界

如果 D0–D3 逐步改善，可以支持：

> NACC→ADNI 的适配收益更依赖 source degradation、低频采集统计对齐和任务表征保护，而不是完整特征域混淆。

如果只有 D0 有效，说明退化建模可能比对抗对齐更重要。

如果 D1 有效但 D2 无效，说明低频对齐有益，但高频保护没有额外收益或实现方式需要调整。

如果 D2 有效但 D3 无效，说明任务保护是关键，而非对称 adapter 不是必要组件。

如果所有 D 系列都不优于 P0，则应拒绝把“3T source 信息过丰富”作为唯一解释，进一步检查 cohort composition、label-conditional shift、预处理差异和 target sample size。

无论结果如何，DS-044 不能单独证明：

- 方法已经消除了 scanner/batch effect；
- 3T/1.5T 差异完全由分辨率造成；
- high-frequency 表征是纯生物学信息；
- bounded adapter 是纯采集偏差修正；
- 方法在所有 source-target 方向上都具有普适收益。

## 15. 结果文件规划

计划中的运行与结果文件：

- 配置：`journal_ds044_direction_aware_scan_filtered.yaml`；
- 代码入口：`experiments/run_ds044_direction_aware.py`；
- 主输出：`outputs/ds044_direction_aware/NACC_to_ADNI/`；
- 方向对照输出：暂不生成；ADNI→NACC 为 deferred。
- 汇总：`outputs/DS-044_SUMMARY_METRICS.json`；
- 报告：`DS-044_EXPERIMENT_REPORT.md`。

本文件已于 2026-09-08 更新：D0–D3 结果已生成并归档；后续只注册 NACC→ADNI 的 P0-Prior、P0-Cal、P0-CShift 主线。ADNI→NACC 暂不扩展。

新主线的成功标准：至少 2/3 seeds 的 target BA、F1、MCC 同方向改善，AUROC 不下降，source validation 不下降，且不使用 target-test label。若 P0-Prior、P0-Cal、P0-CShift 均失败，则停止新增适配模块，并将该方向判定为当前 label-blind 框架下不可稳定修正。
