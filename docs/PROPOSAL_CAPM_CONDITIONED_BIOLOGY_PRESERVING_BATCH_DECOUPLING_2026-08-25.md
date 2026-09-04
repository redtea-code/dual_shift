# Proposal: CAPM-Conditioned Biology-Preserving Batch Decoupling

更新时间：2026-08-25
状态：PROPOSAL；基于 DS-038 两 seed interim/exploratory 结果，尚未实现、尚未验证
适用主方向：`ADNI_to_NACC`，MCI vs AD
证据入口：[DS-038 双任务双方向两 Seed 泛化审计](https://github.com/redtea-code/dual_shift/blob/main/docs/results/DS-038_TWO_TASK_TWO_DIRECTION_2SEED.md)

本提案不是对 DS-038 的结果重命名。DS-038 的 target holdout 曾用于历史实验，所有 target 结果仍是 exploratory；MCI-AD 的原注册五 seed 矩阵也没有完成。提案只把当前最清晰的 screening signal 转化为一个待验证的结构假设。

## 1. 提案动机

AD MRI 中的生物信号与批次/队列效应可能发生耦合。对全部视觉特征施加无条件 domain alignment，可能同时削弱诊断相关结构。当前 CAPM 使用少量人口学变量进行条件化，因此可以作为一个**诊断支持表征锚点**；但在没有额外识别实验前，不能把 CAPM 输出称为纯生物因果特征。

DS-038 的最新两 seed 结果给出更具体的约束：在 MCI-AD、ADNI->NACC 中，G1 `domain-only` 的 target BA 为 `0.6568`，G0 为 `0.5834`，差值 `+0.0734`；G2 `intensity-only` 差值为 `-0.0394`，G3 `both-GRL` 差值为 `-0.0347`。G1 的 target AUROC 也由 `0.6774` 增至 `0.8113`，但 frozen-feature probe 尚未显示稳定的 domain-separability 下降，因此这只能说明 domain-only 是当前主任务的 screening candidate，不能说明批次效应已经被成功去除。

在 NC-MCI、ADNI->NACC 中，G1/G2/G3 的 BA 增益主要伴随 sensitivity 上升和 specificity 下降；NACC->ADNI 两个方向是协议外压力测试，且存在接近全正类预测。由此，本方案不把两个 GRL 视为互补组件，也不把 GRL 结果扩展为跨任务、跨方向普遍规律，而是提出：**批次适配应在 CAPM 条件下作用于任务相关性较低的残差，并允许 identity fallback。**

## 2. 核心假设

对第 `s` 个空间特征尺度，令 `F_s` 为 backbone 特征，`z` 为允许使用的人口学变量，定义：

```text
B_s = CAPM_s(F_s, z)
```

`B_s` 不是已证明的纯生物表示，而是 CAPM-conditioned diagnostic-support representation。方法希望学习一个有界的批次残差校正：

```text
B_s -> B_s - bounded_batch_residual(B_s, source, T_adapt)
```

适配应满足：

1. source/target 的批次残差差异下降；
2. source diagnosis loss 和 CAPM anchor 保持；
3. 校正幅度受控，避免把疾病相关结构当成批次效应删除；
4. 未知或证据不足的方向保留 identity fallback。

### 2.1 DS-038 对本提案的约束

DS-038 只支持以下较窄的设计先验：`domain-only` 值得作为 MCI-AD ADNI->NACC 的候选动作，`intensity-only` 不应作为默认动作，`both-GRL` 不应作为主模型。它不支持把 G1 的 gain 直接解释为 domain alignment，也不支持使用 target BA 选择适配强度或子空间。因此新模块必须将 domain adaptation 限定在 CAPM-conditioned residual 上，并把 residual discrepancy、diagnostic preservation 和 anchor drift 作为共同判定量。

## 3. 方法草案

### 3.1 CAPM-conditioned residual adaptation

先计算 CAPM 调整后的空间特征 `B_s`。再由 source 与无标签 `T_adapt` 的统计量估计批次差异 `Delta_s`，例如 channel mean/std、robust Wasserstein summary 或保留空间维度时的 feature-map discrepancy。

适配器输出有界残差：

```text
R_s = alpha_s * M_s * RMS(B_s) * tanh(H_s(Delta_s, B_s))
B_s_adapt = B_s - R_s
```

其中 `M_s` 是批次差异与诊断保护共同决定的 mask，`alpha_s` 是固定上界或 source-validation 选定的强度。适配器不接收 target label、target covariate、environment、prediction 或 target metric。

### 3.2 Biology-preservation constraint

保护项不把所有 CAPM 特征强制保持不变，而是约束适配前后的诊断支持：

```text
L_bio = source classification preservation
      + CAPM-anchor consistency
      + bounded residual penalty
```

可选地，在 source-only 阶段从 `B_s` 构造低秩诊断支持子空间 `P_bio`，仅允许 `(I - P_bio)` 方向承担批次校正。`P_bio` 只由 source image/label 和 CAPM 输出构造，不能使用 target label。

### 3.3 非互补 GRL 的处理

domain/intensity 两个 GRL 不再同时作为主模块。第一版只保留一个 **domain-residual adapter**；intensity GRL 作为负控或后续单独消融，不进入默认路径。若保留域判别器，它只作用于批次残差或其保护子空间的补空间：

```text
L_residual_domain = DomainLoss((I - P_bio) * B_s_adapt)
```

不对完整 `B_s` 做无条件对抗对齐。若 source-only vulnerability/discrepancy gate 没有足够证据，则使用 preservation 或 identity，而不是强行启用 GRL。即使 domain-only 在主任务中通过 screening，也必须证明它在 residual 上降低 discrepancy，同时没有明显损害 source diagnosis 或 CAPM anchor。

## 4. feature scale 设计

feature size 的创新不应只是增加 channel、改变 layer 或修改 patch size。建议把不同尺度赋予不同功能：

```text
layer3 / patch-level:  建立较稳定的 CAPM-conditioned diagnostic anchor
layer4 / pixel-level:  表达空间批次残差并执行 bounded domain adaptation
layer5 / global:        作为 identity、分类和负迁移监控路径
```

候选跨尺度结构为：

```text
B3 = CAPM_layer3(F3, z)
B4 = CAPM_layer4(F4, z)
R4 = (I - P_bio) * B4
R4_adapt = residual_adapter(R4, Down(B3), Delta4)
logits = classifier(B3, B4 - R4 + R4_adapt)
```

第一实现不应同时学习 layer3、layer4、layer5 三个自由适配器。建议先固定 layer4 spatial feature map 作为适配位置，并以 layer3 CAPM 表征作为保护条件；`layer3_patch2`、`layer4_pixel` 和 `layer5_pixel` 的既有 DS-034 结果只能用于设计背景，不能被解释为该跨尺度结构已经获胜。

这是一项待验证的结构假设，不代表 `layer3` 或 `layer4` 已被证明是正确尺度。DS-034 的既有尺度结果只能作为设计背景，不能直接证明该跨尺度机制有效。

## 5. 与现有路线的差异

| 路线 | 主要操作 | 本提案的差异 |
|---|---|---|
| FMM/DyMix | raw-image amplitude/phase 与 GRL | 不在原始图像上直接重混；以 CAPM-conditioned feature 为锚点 |
| DS-038 两个 GRL | domain/intensity 并列 factorial | 只把 MCI-AD ADNI->NACC 的 G1 作为 screening candidate；不采用 G3，不默认启用 intensity GRL |
| 当前 CAPM/IE-CAPM | demographic-conditioned spatial modulation | 继续利用 CAPM 输出约束后续批次残差校正 |
| SAMix/OT diagnostics | 频率或特征 shift 的标签盲度量 | 仅借鉴 discrepancy 量化，不直接按 target test 重训或选模 |
| image harmonization | 直接修改像素或生成图像 | 保持 feature-level、bounded、source-preserving 的校正 |

可使用的论文表述是：

> We propose CAPM-conditioned biology-preserving batch decoupling, which performs bounded domain adaptation on the residual feature component after demographic conditioning instead of enforcing unconditional invariance on the entire diagnostic representation.

不能表述为已经识别了 scanner/field-strength 的独立因果效应，也不能把 CAPM 输出直接称为纯生物信号。

## 6. 最小验证矩阵

在计算资源受限时，先使用当前两 seed 假设做工程筛选。主方向只保留 `ADNI_to_NACC`、MCI vs AD；NC-MCI 与 NACC->ADNI 只能作为独立的压力测试，不参与主模型选择：

1. `CAPM`；
2. `CAPM + global domain adaptation`（负控，验证无条件对齐是否伤害诊断）；
3. `CAPM + same-scale domain residual adaptation`（首选候选）；
4. `CAPM + cross-scale BioCAPM-DA`（结构候选）；
5. `CAPM + residual intensity adaptation`（负控/单独消融，不进入默认主模型）。

其中第 2 项应明确区别于 FMM 的 raw-image G1：这里的 `global domain adaptation` 只作为“完整 CAPM feature 无条件适配”的负控，不把 FMM 的训练结果直接当作新方法结果。

每个条件必须保留相同 source split、source-validation selector、训练预算和 target protocol。报告：

- source validation/test 与 target exploratory BA/AUC；
- sensitivity、specificity、subject-level paired seed difference；
- domain-probe/discrepancy before and after adaptation；
- CAPM anchor drift；
- residual RMS 与 identity/fallback 比例；
- source diagnosis preservation；
- target-label-blind audit、manifest/config/hash 和 checkpoint provenance。

主方向的首要成功模式不是单一 target BA 上升，而是同时满足：

```text
target BA/AUROC 不低于 CAPM control；
residual discrepancy 下降；
CAPM anchor drift 受控；
source diagnosis preservation 通过；
residual RMS 非零但有界；
没有由 sensitivity/specificity 单向偏移造成的假增益。
```

若 residual discrepancy 下降但 source diagnosis 或 CAPM anchor 明显受损，则判定为负迁移。若只 target BA 上升而没有 preservation/残差证据，不足以支持 biology-preserving 机制。

## 7. 当前边界与下一步

这是一个独立的新方法提案，不改写 DS-034--038 的结果，也不授权将已有 C4/FMM/GRL 结果解释为该方法的先验验证。当前最重要的证据边界是：MCI-AD ADNI->NACC 的 G1 只提供候选方向，NC-MCI 的 BA 增益存在 sensitivity/specificity trade-off，NACC->ADNI 是 unsupported-protocol stress direction。下一步应先完成：

1. 明确 `B_s` 的张量层级和可用 feature cache；区分 raw-volume FFT、spatial feature-map 和 GAP embedding；
2. 设计不访问 target label 的 `Delta_s` 与 `P_bio` 构造；
3. 先做 frozen checkpoint/no-retrain 机制探针，再决定是否训练新适配器；
4. 预注册 matched ablation 与 identity fallback；
5. 只有在 source preservation、残差 discrepancy 和 target 结果方向一致时，才进入更大 seed 或论文主模型。

6. 对齐报告口径：最新两 seed 报告把四组扩展矩阵写为 `8/8`，旧版 `docs/results/DS-038/README.md` 的 diagnostics rerun 仍记为 `9/20`。实现新模块前，必须固定 output root、artifact generation、manifest hash 和 checkpoint schema，不能混用两代产物。

## 8. 简要数学定义

以下公式给出方法的最小数学描述；其中“protected”表示 source-task/CAPM 支持约束，不宣称该子空间是纯生物因果空间。

### 8.1 特征与受保护分解

令

```text
F4 = E_theta(x),                         # layer4 feature map
p0 = f_CAPM(F4, z),                      # clean CAPM prediction
G  = E_S[ grad_F4 l_CE(p0, y) grad_F4 l_CE(p0, y)^T ],
P  = TopQProjector(G),                   # source-only protected projector
F_prot = P F4,
R      = (I - P) F4.                     # residual feature
```

`P` 只由 source image/label 和冻结 CAPM 路径构造，不使用 target label、target metric 或 `T_test`。

### 8.2 无标签目标域频率先验

对频带 `k`，由 `S_train` 与无标签 `T_adapt` 的 feature-map 统计量计算：

```text
delta_k = |mu_S,k - mu_T,k|
          / sqrt((sigma_S,k^2 + sigma_T,k^2) / 2 + eps),

rho_k   = P_boot(k is a stable shifted band),
delta_tilde_k = rho_k * delta_k / (max_j delta_j + eps).
```

若某频带的 bootstrap 支持不足，则令 `rho_k = 0`，该频带回退为 identity。

### 8.3 残差级频率适配

仅对 `R` 进行有界的频率衰减：

```text
R_adapt = IFFT_k[(1 - a * delta_tilde_k) * FFT_k(R)],
F4_adapt = F_prot + R_adapt,
0 <= a <= a_max.
```

最终预测为：

```text
p_adapt = f_CAPM(F4_adapt, z).
```

与当前 C4 的差别是：当前 gate 作用于完整 `F4`，新方案只作用于 `(I - P)F4`。

### 8.4 训练目标

初始阶段仅更新低参数 residual gate，冻结 backbone、CAPM 和 classifier：

```text
L = L_CE(p_adapt, y_s)
  + lambda_a KL(stopgrad(p0) || p_adapt)
  + lambda_r ||F4_adapt - F4||_2^2 / (||F4||_2^2 + eps).
```

其中 `L_CE` 保持 source diagnosis，`KL` 限制 CAPM anchor 漂移，`lambda_r` 限制校正幅度。主方案不加入 global GRL；residual-only GRL 只作为后续消融。

### 8.5 访问约束与判定量

```text
delta = delta(S_train, T_adapt images only),
T_adapt ∩ T_test = empty,
target labels/metrics/predictions are not used before final evaluation.
```

机制审计至少要求：

```text
D_residual(S, T_adapt) decreases,
D_anchor(p0, p_adapt) <= tau_anchor,
source performance after adaptation >= source baseline - tau_source,
gate activity is non-zero but bounded.
```

若残差差异下降但 anchor 或 source diagnosis 明显受损，则判定为过度校正/负迁移；若 gate 接近 identity，则不能声称残差适配产生了作用。
