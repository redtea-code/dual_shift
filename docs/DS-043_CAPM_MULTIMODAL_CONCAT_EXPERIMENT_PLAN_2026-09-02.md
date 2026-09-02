# DS-043 研究计划/副地图：CAPM + 保守多模态拼接

状态：**PROPOSED / PLAN WRITTEN / NO DS-043 RESULTS**
主问题：在不改变既有 CAPM-GRL 机制的前提下，只有 `age`、`sex`、`education` 时，轻量 table concat 是否值得作为 MRI 分类的辅助输入？

本计划主动收敛实验规模：保留原有 8 个 CAPM-GRL 变体，并只增加一个 `P0-M`。不注册 `F1-M` 至 `R3-M`，不引入 CMRP、CORAL、relation loss、cross-attention 或新的多模态 UDA 矩阵。

## 1. 模型输入与输出

### 1.1 数据合同

主方向为 `ADNI_to_NACC`，任务为 scan-filtered MCI vs AD，正式重跑使用 seeds `42,43,44`。所有 split 按 subject 固定，`T_adapt` 与 `T_test` subject-disjoint。

| 集合 | 可访问字段 | 用途 |
|---|---|---|
| `S_train` | MRI、table、source label | source 分类训练；CAPM-GRL 的 source 分支 |
| `S_val` | MRI、table、source label | 唯一 checkpoint/停止选择数据 |
| `S_test` | MRI、table、source label | 冻结 checkpoint 后的 source 报告 |
| `T_adapt` | 无标签 target MRI；原有 8 个变体按 image-only contract 使用 | target-style 生成和 domain GRL |
| `T_test` | target MRI、table，最终才读取 label | 冻结模型后的最终 target 报告 |

严格 UDA 规则：target label、prediction、metric 和模型排名不能进入 adaptation 或 selector。`P0-M` 是 source-only 多模态可行性基线，不使用 `T_adapt`；它在最终 `T_test` 推理时需要 target table。

### 1.2 公共输入和张量尺寸

```text
MRI:        x [B, 1, 160, 196, 160]
table:      t [B, 3] = [age, sex, education]
missing:    m [B, 3] = [age_missing, sex_missing, education_missing]
source y:   y [B]
output:     logits [B, 2]
```

在 `layer4_pixel`、`layers=(1,1,1,1)` 的匹配设置下，ResNet layer4 输出为：

```text
F4 [B, 512, 5, 7, 5]
B4 = CAPM(F4, t) [B, 512, 5, 7, 5]
GAP(B4) = h_M [B, 512]
```

实际空间尺寸随 `input_shape` 改变；`[B,512,5,7,5]` 是本计划默认 MRI 几何下的已核对尺寸。

### 1.3 P0-M 的 table 输入

`P0-M` 不使用 table MLP、attention 或 relation module，只使用 source-fitted preprocessing 后的直接向量：

```text
t_tilde = [age_norm, sex_code, education_norm,
           age_missing, sex_missing, education_missing]
t_tilde [B, 6]
```

age 和 education 的缩放、缺失填补和 sex 编码沿用现有 manifest/preprocessor；不把 site、scanner、field strength、manufacturer 或 diagnosis 字段加入 table。

## 2. 模型的数据流

### 2.1 原始 CAPM source 基准 P0

```text
x_s [B,1,160,196,160]
  -> ResNet conv1/maxpool/layer1-layer4
  -> F4 [B,512,5,7,5]
  -> table-free SpatialGate3d
  -> original CAPM(F4, t_s)
  -> B_s [B,512,5,7,5]
  -> GAP [B,512]
  -> classifier [B,2]
  -> CE(y_s)
```

这是 `P0 = capm_erm`。table 在 CAPM 条件化中使用，但不作为独立分类输入；`T_adapt` 不参与训练。

### 2.2 频域控制分支 F0

训练时额外生成：

```text
x_i  = intensity_transform(x_s)
x_t* = amplitude_mix(x_s, x_t)
```

对应的 forward 为：

```text
x_s,  t_s -> CAPM -> B_s  -> classifier -> CE(y_s)
x_i,  t_s -> CAPM -> B_i  -> classifier -> CE(y_s)
x_t*, t_s -> CAPM -> B_t* -> diagnostic feature path
```

F0 加入 source clean/source-intensity 分类、attention consistency 和 CAPM anchor。由于 domain/intensity GRL 系数均为零，`B_t*` 在当前实现中主要用于诊断，不通过有效 target loss 更新 encoder。

### 2.3 完整 CAPM 特征上的 GRL：F1-F3

F1、F2、F3 不改变疾病分类器的输入，均仍为完整的 `B4 -> GAP -> classifier`。

**F1：domain GRL**

```text
B_s  -> GAP -> GRL -> domain discriminator(label=source)
B_t* -> GAP -> GRL -> domain discriminator(label=target-style)
```

GRL 前向为 identity，反向把 domain loss 对 backbone/CAPM 的梯度乘以负系数。target-style 图像因此产生真正的 target-dependent adaptation gradient。

**F2：intensity GRL**

```text
B_s -> GAP -> GRL -> intensity discriminator(label=clean source)
B_i -> GAP -> GRL -> intensity discriminator(label=intensity source)
```

F2 只比较 source clean 与 source intensity，不产生 source-target domain 对抗梯度；target-style 分支不决定 F2 的训练更新。

**F3：双 GRL**

```text
B_s, B_t* -> domain GRL
B_s, B_i  -> intensity GRL
B_s, B_i  -> diagnosis classifier
```

F3 同时施加两种对抗梯度，作为 factorial control，不预先假设两种 GRL 互补。

### 2.4 CAPM residual 上的 GRL：R1-R3

先从 P0 的 source CE 梯度拟合并冻结任务支持 projector：

```text
P_task: [512] -> [512]
R4 = (I - P_task) B4
```

projector 不读取 target 数据，也不由 target label 拟合。

**R1：residual domain GRL**

```text
B_s, B_t* -> R_s, R_t*
R_s, R_t* -> GAP -> domain GRL
B_s       -> GAP -> diagnosis classifier
```

**R2：residual intensity GRL**

```text
B_s, B_i -> R_s, R_i
R_s, R_i -> GAP -> intensity GRL
B_s, B_i -> GAP -> diagnosis classifier
```

**R3：residual 双 GRL**

```text
R_s, R_t* -> domain GRL
R_s, R_i  -> intensity GRL
B_s, B_i  -> diagnosis classifier
```

R1-R3 的核心变化是 GRL 的输入从完整 `B4` 改为 residual `R4`；疾病分类器仍然看到完整 CAPM 特征。因而 full/residual 比较的是对抗梯度的作用范围，不是两个不同的诊断表示。

### 2.5 保守多模态基线 P0-M

`P0-M` 只在 P0 source-only 分类路径上增加 table concat：

```text
x_s, t_s
  -> ResNet -> F4 [B,512,5,7,5]
  -> SpatialGate3d -> original CAPM(F4,t_s)
  -> B_s [B,512,5,7,5]
  -> GAP -> h_M [B,512]

t_s + missingness
  -> source-fitted normalization
  -> t_tilde [B,6]

concat(h_M, t_tilde)
  -> z_M [B,518]
  -> linear classifier [B,2]
  -> CE(y_s)
```

该 head 的参数量只比原始 `512 -> 2` 分类器增加 12 个权重，不改变 MRI backbone、CAPM、GRL 或 projector。为了从原始 CAPM 平滑起步，建议将新分类器的 MRI 权重复制自 P0，table 权重初始化为零，bias 复制自 P0。

P0-M 的 target 流只在最终冻结评估出现：

```text
x_t, t_t -> CAPM -> GAP h_M_t
t_t      -> t_tilde_t
[h_M_t, t_tilde_t] -> classifier -> target logits
```

它不是 target-adaptation 变体，也不使用 target label。043 不继续注册 `F1-M` 至 `R3-M`，避免在 table 贡献尚未确认时把实验规模扩大 8 倍。

## 3. 涉及的模块

| 模块 | 043 中的作用 | 证据状态 |
|---|---|---|
| `ScaleTableInteractionAblation3D` | `original_capm` 的 ResNet、layer4 feature map、CAPM 条件化和 pooling | 已有实现，可复用 |
| `SpatialGate3d` | CAPM 前的共享空间注意力 | 已有实现，可复用 |
| `GradientReversal` | 反转 domain/intensity 对抗梯度 | 已有实现，可复用 |
| `domain_discriminator` | 区分 source 与 target-style | 已有实现，可复用 |
| `intensity_discriminator` | 区分 clean source 与 intensity source | 已有实现，可复用 |
| `TaskSupportProjector` | 用 source CE 梯度构造 frozen residual 子空间 | 已有实现，可复用 |
| `make_frequency_batch` | 生成 `x_i` 与 `x_t*` | 已有实现，可复用 |
| `compute_capm_frequency_losses` | source CE、GRL、attention、anchor 和诊断量 | 已有实现，可复用 |
| source/target loaders | source label、image-only `T_adapt`、冻结 `T_test` | 已有协议实现 |
| `P0-M` concat head | `h_M [B,512]` 与 `t_tilde [B,6]` 拼接后线性分类 | **043 proposed，尚未实现** |
| CMRP relation/CORAL | 本计划不使用 | 明确排除 |

## 4. 每个模块的具体实现

### 4.1 8 个既有变体

所有 8 个变体共享 `layer4_pixel` CAPM backbone、频域变换参数、source split、optimizer、训练预算和 source-validation selector。它们的差异只在频域分支、GRL 开关和 GRL 输入范围。

| 变体 | 相对 P0 的变化 | target 训练作用 | GRL 输入 | 推理时变化 | 实验问题 |
|---|---|---|---|---|---|
| `P0` | 无 | 不使用 | 无 | 原始 CAPM 分类 | source-only CAPM 锚点 |
| `F0` | 加 `x_i`、`x_t*`、attention/anchor | target-style 主要为 control/diagnostic | 无 | 与 P0 相同 | 频域增强本身是否改变结果 |
| `F1` | 增加 domain discriminator | `x_t*` 产生 domain GRL | 完整 `B4` | GRL/discriminator 移除 | 完整 CAPM domain 对抗是否有用 |
| `F2` | 增加 intensity discriminator | 仅 source intensity 对抗 | 完整 `B4` | GRL/discriminator 移除 | 强度不变性是否有用 |
| `F3` | 同时增加两类 GRL | target-style domain + source intensity | 完整 `B4` | GRL/discriminator 移除 | 双 GRL 是否互补 |
| `R1` | F1 的 GRL 改接 residual | `x_t*` 产生 residual domain 梯度 | `R4=(I-P_task)B4` | 分类仍使用完整 `B4` | 限制 domain 梯度是否更安全 |
| `R2` | F2 的 GRL 改接 residual | source intensity residual 对抗 | `R4` | 分类仍使用完整 `B4` | 限制 intensity 梯度是否更安全 |
| `R3` | F3 的两个 GRL 改接 residual | residual domain + intensity | `R4` | 分类仍使用完整 `B4` | residual 双 GRL 是否值得保留 |

`P0` 先用 source validation 选出 checkpoint；F0-F3/R1-R3 从同一个 seed-specific P0 checkpoint warm-start。历史 DS-038/DS-040 数字只作为背景，不直接填入 043 结果表。

### 4.2 P0-M 的具体实现

建议在 CAPM-GRL 模型中增加一个独立的 `use_table_concat` 开关，而不是重写 CAPM 或 GRL：

```text
use_table_concat=False -> P0/F0-F3/R1-R3 的原始 MRI classifier
use_table_concat=True  -> P0-M 的 [GAP(B4), t_tilde] classifier
```

实现要求：

1. CAPM 的 table condition 仍然保留；concat table 是第二次使用同一 table，不替换 CAPM；
2. concat 前只使用 source-fitted normalization、sex code 和 missingness flags；
3. 新 head 的 table 权重零初始化，确保初始预测接近 P0；
4. P0-M 不启用 target adaptation、不加入 pseudo-label、不加入 target entropy；
5. P0-M 与 P0 使用同一 source split、训练预算、selector 和 target-test subjects；
6. P0-M 的 target table 只在最终推理读取，需在结果 provenance 中单独记录为 multimodal inference access。

### 4.3 代码与计划边界

- 8 个 CAPM-GRL 变体来自 DS-040 reference implementation，043 正式运行前需要在当前实验 revision 中确认其源码、配置和 artifact inventory 一致；
- `P0-M` 当前只是计划项，不能在文档中标成已实现；
- 当前 CMRP 的 independent MRI/table encoder、relation、CORAL 和 bounded adapter 不属于 043；
- 043 的 batch/scanner 解释仍是辅助假说，domain discriminator loss 或 discrepancy 下降不能单独构成成功标准。

## 5. 实验结果与分析

### 5.1 预注册主比较

主 baseline 为 `P0`。所有结果必须使用同一 `ADNI_to_NACC`、MCI vs AD、subject split、source-validation selector 和 seeds `42,43,44`。以下是运行前的结果表，`NR` 表示尚未运行。

#### Target 结果表（subject-level，ADNI→NACC，seeds=42,43,44）

| 变体 | ACC | AUROC | BA | Macro-F1 | Sensitivity | ΔAUROC vs P0 | 证据状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| `P0` (baseline) | NR | NR | NR | NR | NR | — | proposed / not run |
| `F0` | NR | NR | NR | NR | NR | NR | proposed / not run |
| `F1` | NR | NR | NR | NR | NR | NR | proposed / not run |
| `F2` | NR | NR | NR | NR | NR | NR | proposed / not run |
| `F3` | NR | NR | NR | NR | NR | NR | proposed / not run |
| `R1` | NR | NR | NR | NR | NR | NR | proposed / not run |
| `R2` | NR | NR | NR | NR | NR | NR | proposed / not run |
| `R3` | NR | NR | NR | NR | NR | NR | proposed / not run |
| `P0-M` | NR | NR | NR | NR | NR | NR | proposed / not run |

`ΔAUROC` 必须先按 seed 计算再汇总；若最终主终点注册为 BA，则另行报告 `ΔBA`，不能用 AUROC 替代 BA。

### 5.2 判定规则

1. **table 可行性**：只有当 `P0-M` 在 target BA/AUROC 不低于 P0、source preservation 没有明显恶化，且多个 seed 方向一致时，才考虑把 concat head 扩展到后续 UDA 变体。
2. **frequency control**：`F0` 相对 `P0` 只解释频域增强/attention control，不能解释为 GRL 效果。
3. **domain GRL**：比较 `F1-F0` 和 `R1-F0`；若 residual 只保持性能而 full 降低，只能支持“对抗位置值得研究”，不能称 residual 为纯 biological 或纯 batch 子空间。
4. **intensity GRL**：比较 `F2-F0` 和 `R2-F0`；它是 source intensity invariance 对照，不等同于 target-domain UDA。
5. **双 GRL**：只有 `R3` 同时优于 `R1`、`R2` 且 seed-consistent，才讨论条件互补；不能由 discriminator loss 接近 `log(2)` 推断成功。
6. **主任务优先**：target 任务指标、source preservation、GRL/encoder 梯度和 feature discrepancy 必须联合解释；任何单独的 discrepancy 下降都不能替代 target 任务终点。

### 5.3 结果解释边界

043 能够回答：

- 原有 CAPM-GRL 8 个变体在统一 protocol 下的相对表现；
- full 与 residual GRL 的梯度范围差异；
- 三个 table 变量经过保守 concat 后是否提供可重复的增量信息。

043 不能单独证明：

- table 包含足够的独立疾病信息；
- CAPM 输出是纯 biological representation；
- residual 是纯 batch/scanner representation；
- GRL 已经去除了 scanner 或 batch effect；
- 单个 seed 或 target point estimate 说明方法稳定有效。

若 `P0-M` 没有稳定增益，043 到此停止多模态 concat 扩展，保留原有 8 个 MRI-UDA 变体作为主线。若 `P0-M` 有稳定但幅度较小的增益，也只把 table 定义为弱条件信息，不重新设计一套复杂多模态模型。
