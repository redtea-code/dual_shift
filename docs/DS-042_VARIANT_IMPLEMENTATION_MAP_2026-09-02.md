# DS-042 各变体具体实现
更新时间：2026-09-02

本文件把 DS-042 配置中的 13 个 `cmrp_*` 变体映射到当前代码的具体行为。它描述的是实现事实，不把变体名称自动解释成已验证的机制结论。

## 1. 公共实现

所有 CMRP 变体都由 `Model/ablation/cmrp_uda.py::CMRPUDA3D` 构造，由 `experiments/train_journal.py` 注册并由 `training/cmrp_uda_loop.py` 训练。

### 1.1 输入

```text
MRI: [B, 1, D, H, W]
table: [B, 3] = [age, sex, education]
missingness: [B, 3] = [age_missing, sex_missing, education_missing]
```

source batch 额外提供 `label`；target adaptation batch 只提供 MRI、table、missingness 和 subject metadata，不向损失函数提供 target label。

### 1.2 表示路径

```text
x_MRI -> image_backbone -> mri_encoder -> mri_raw
t    -> table_encoder  -> table_raw

mri_raw   -> mri_shared,   mri_specific
table_raw -> table_shared, table_specific

[mri_shared, table_shared, mri_specific, table_specific,
 mri_shared * table_shared]
    -> fusion -> joint -> classifier -> logits
```

`shared/specific` 是可消融的投影分解，不在代码中被定义为 biological 或 batch 表示。`mri_specific` 和 `table_specific` 可经过零初始化的 bounded residual adapter。

### 1.3 公共损失

```text
L_task       = CE(source_logits, source_label)
L_rel_s      = 1 - cosine(mri_shared_s, table_shared_s)
L_rel_t      = 1 - cosine(mri_shared_t, table_shared_t)
               + 1 - cosine(mri_shared_t, mri_shared_t_aug)
L_align      = CORAL(指定的 source/target 表示)
L_identity   = ||joint_adapt_s - stopgrad(joint_clean_s)||² / ||joint_clean_s||²
L_prox       = adapter 参数平方均值

L_total = λ_task L_task + λ_s L_rel_s + λ_t L_rel_t
          + λ_a L_align + λ_i L_identity + λ_p L_prox
```

默认 YAML 权重为：`lambda_task=1.0`、`lambda_source_relation=0.1`、`lambda_target_relation=0.1`、`lambda_alignment=0.1`、`lambda_identity=0.1`、`lambda_prox=1e-4`。target augmentation 是对 target MRI 加 `target_aug_noise=0.01` 的高斯噪声。

## 2. 变体总览

| 变体 | image | table | target adaptation | alignment | source relation | target relation | identity/prox | 代码操作 |
|---|---|---|---|---|---|---|---|---|
| `cmrp_source_only` | 使用 | 使用 | 不使用 | 无 | 开启 | 无 | 开启 | 不创建 target loss；target 只用于最终评估 |
| `cmrp_uda` | 使用 | 使用 | 使用 | MRI-specific（配置默认） | 开启 | 开启 | 开启 | 完整 CMRP-UDA v0 |
| `cmrp_joint_uda` | 使用 | 使用 | 使用 | joint | 开启 | 开启 | 开启 | 对 `joint` 做 CORAL |
| `cmrp_mri_specific` | 使用 | 使用 | 使用 | MRI-specific | 开启 | 开启 | 开启 | 对 `mri_specific` 做 CORAL |
| `cmrp_table_specific` | 使用 | 使用 | 使用 | table-specific | 开启 | 开启 | 开启 | 对 `table_specific` 做 CORAL |
| `cmrp_both_specific` | 使用 | 使用 | 使用 | MRI + table specific | 开启 | 开启 | 开启 | 两个 specific CORAL 相加 |
| `cmrp_no_source_relation` | 使用 | 使用 | 使用 | 默认 MRI-specific | 关闭 | 开启 | 开启 | source relation 置零 |
| `cmrp_no_target_relation` | 使用 | 使用 | 使用 | 默认 MRI-specific | 开启 | 关闭 | 开启 | target relation 置零 |
| `cmrp_no_alignment` | 使用 | 使用 | 使用 | 无 | 开启 | 开启 | 开启 | alignment 置零 |
| `cmrp_no_identity` | 使用 | 使用 | 使用 | 默认 MRI-specific | 开启 | 开启 | 关闭 | identity 和 proximal 同时置零 |
| `cmrp_shuffled_table` | 使用 | 打乱 | 使用 | 默认 MRI-specific | 关闭 | 关闭 | 开启 | batch 内随机置换 table 和 missingness |
| `cmrp_missingness_only` | 使用 | 数值置零，保留 missingness | 使用 | 默认 MRI-specific | 关闭 | 关闭 | 开启 | table 数值置零，missing flags 保留 |
| `cmrp_table_only` | 关闭图像信息 | 使用 | 使用 | 默认 MRI-specific | 关闭 | 关闭 | 开启 | `use_image=False` |

## 3. 每个变体的具体行为

### 3.1 `cmrp_source_only`

这是 **CMRP 模型族的 source-only matching control**，不是普通 `image_only` CE baseline。

- source 训练使用 `L_task + λ_s L_rel_s + λ_i L_identity + λ_p L_prox`；
- 不建立 target loss，`target_adapt_loader` 传入 `None`；
- target 若存在，只在冻结 checkpoint 后作为 `T_test` 评估；
- MRI 和 table 都参与 source 训练；
- 用于回答：同一 CMRP 架构在没有 target adaptation 时的结果是什么。

因此论文中应称为“CMRP source-only control”，不能直接称为“纯 source ERM”。

### 3.2 `cmrp_uda`

这是完整方法候选。

- source 使用有标签 MRI-table 对；
- target 使用无标签 MRI-table 对；
- `L_rel_s` 和 `L_rel_t` 都开启；
- 默认 `alignment_scope=mri_specific`，即 `CORAL(mri_specific_s, mri_specific_t)`；
- `L_identity` 比较 adapted source joint 与 clean source joint；
- bounded MRI/table adapter 通过 `L_prox` 限制更新幅度。

### 3.3 `cmrp_joint_uda`

只改变 alignment 位置：

```text
L_align = CORAL(joint_s, joint_t)
```

其余 source relation、target relation、identity/proximal 和 target augmentation 与完整方法相同。它是整体 joint alignment 对照，不是 relation-preserving 的等价实现。

### 3.4 `cmrp_mri_specific`

显式固定：

```text
L_align = CORAL(mri_specific_s, mri_specific_t)
```

它与当前 YAML 默认 `cmrp_uda` 的 alignment 行为相同。保留该变体的作用是让主表中的“指定 alignment scope”不依赖配置默认值。

### 3.5 `cmrp_table_specific`

只对 table-specific 表示做 target/source CORAL：

```text
L_align = CORAL(table_specific_s, table_specific_t)
```

MRI 仍参与任务、relation 和 fusion；并不是 table-only 模型。

### 3.6 `cmrp_both_specific`

同时对两种 specific 表示对齐：

```text
L_align = CORAL(mri_specific_s, mri_specific_t)
          + CORAL(table_specific_s, table_specific_t)
```

它测试“双 specific 对齐”是否优于单一 modality-specific 对齐，但不能把 specific 直接解释为 batch 表示。

### 3.7 `cmrp_no_source_relation`

- 计算 source paired relation，但在汇总损失前强制设为零；
- target relation、alignment、identity/proximal 保留；
- 用于测量 source MRI-table 配对约束移除后的变化。

这是 loss ablation，不改变输入配对，不等同于 shuffled-table。

### 3.8 `cmrp_no_target_relation`

- source relation 保留；
- target relation 和 target MRI augmentation consistency 一起被移除；
- target MRI/table 仍用于 alignment；
- identity/proximal 保留。

它测试 target 无标签关系约束的边际作用，而不是移除整个 target adaptation。

### 3.9 `cmrp_no_alignment`

- target adaptation loader 仍被读取；
- target relation 和 augmentation consistency 仍开启；
- source relation、identity/proximal 仍开启；
- 只有 `L_align` 被置零。

因此它是“无 CORAL 的 target relation adaptation”，不是 source-only。

### 3.10 `cmrp_no_identity`

代码中同时关闭：

```text
L_identity = 0
L_prox = 0
```

source relation、target relation 和 alignment 仍保留。该变体回答受限 adapter 的 identity/proximal 保护是否必要。

### 3.11 `cmrp_shuffled_table`

在 `_table_inputs` 中对每个 batch 生成随机 permutation：

```text
table = table[permutation]
missingness = missingness[permutation]
```

- MRI 不变，table 与 missingness 脱离原 subject；
- source relation 和 target relation 被关闭，因为真实 pair 已被破坏；
- MRI-specific alignment 和 identity/proximal 仍保留；
- 用于检验真实 MRI-table 配对是否比随机配对更有用。

当前实现的注意点：`_forward(..., variant="cmrp_shuffled_table")` 每次调用都会重新置换 table，因此 clean identity anchor 也会使用另一组随机 permutation。这个负控的含义仍可用于探索，但若要把 identity 差异严格归因于 adapter，应该增加固定 permutation 或 clean-table 路径后重跑。

### 3.12 `cmrp_missingness_only`

数值协变量被置零，但 missingness flags 原样输入 table encoder：

```text
table = zeros_like(covariates)
table_input = [0, 0, 0, age_missing, sex_missing, education_missing]
```

- source relation 和 target relation 关闭；
- MRI-specific alignment、identity/proximal 保留；
- 模型仍可以利用缺失模式，不能把它解释为完全无 table 输入；
- 用于检查缺失模式 shortcut 是否足以解释结果。

### 3.13 `cmrp_table_only`

构造模型时设置 `use_image=False`：

- MRI image 不进入 image backbone；
- MRI raw embedding 置为零；
- table encoder、specific projection、fusion 和 classifier 仍训练；
- source/target relation 被关闭；
- 用于检验 age/sex/education 单独能否提供接近 MRI+table 的诊断性能。

实现上 MRI projection 的 bias 仍存在，因此该变体没有 image-dependent signal，但不是把整个 MRI 分支模块从图中物理删除。

## 4. 训练入口的统一行为

`experiments/train_journal.py::_train_cmrp_variant` 对所有变体统一执行：

1. 按 seed 初始化 CMRP 模型和 AdamW；
2. 每个 source batch 循环取一个 target adaptation batch，target loader 较短时循环复用；
3. source validation 运行 `_selection_key`，不读取 target label；
4. 保存 `best_checkpoint.pt` 和 `last_checkpoint.pt`；
5. 加载 best checkpoint 后评估 source validation、source test 和 target test；
6. 输出 prediction CSV、`journal_metrics.json`、subject-level metrics 和 bootstrap CI。

当前实际 selector 需要特别记录：CMRP variant 没有进入 DualShift/METADATA 的 composite selector 集合，因而通用 `_selection_key` 对它默认按 source-validation AUROC 排序；YAML 中的 `primary_metric: balanced_accuracy` 不会自动改变这个行为。正式复现前必须把 selector 明确写入协议，不能只看配置字段名。

## 5. 代码和实验报告的对应关系

报告中的 13 行正好对应当前 YAML 的 13 个变体。报告给出的 `cmrp_uda` 与 `cmrp_source_only` 差值应解释为：

```text
CMRP-UDA target AUROC  - source-only AUROC = -0.0051
CMRP-UDA target BA     - source-only BA    = -0.0248
```

因此当前结果不支持 CMRP-UDA 的 target 性能提升假设。`cmrp_table_specific` 虽然 AUROC 最高，但 BA 仍低于 source-only，不能单凭一个指标升级为主方法。

## 6. 后续重跑优先级

1. 先固定 CMRP selector 为 source-val BA、AUROC 或预注册复合分数之一；
2. 修正 shuffled-table 的 permutation/clean-anchor 语义并重新生成该负控；
3. 将 CMRP 代码、YAML、target split、日志、checkpoint hash、predictions 和 metrics 放入同一 revision；
4. 在同一 selector 和 split 下补齐 image-only UDA 与普通 multimodal UDA；
5. 重新计算 paired seed difference 与 bootstrap CI，再决定是否继续 CMRP 方向。
