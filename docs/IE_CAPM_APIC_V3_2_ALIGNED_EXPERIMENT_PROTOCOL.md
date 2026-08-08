# IE-CAPM 与 APIC v3_2 对齐实验协议

> 历史协议声明：本文件记录配对 subject-wide 排除的旧 revision，供既有结果复现。
> 新的 scan-filtered 实验不得使用本文件第 3--4 节；应遵循
> `docs/IE_CAPM_APIC_V3_2_SCAN_FILTERED_1P5T_PROTOCOL.md` 和
> `review/plans/34_scan_filtered_capm_execution_plan_2026-08-08.md`。

## 1. 协议定位

本协议用于评估 Image-Evidence Calibrated CAPM（IE-CAPM）是否能够在
CAPM 的基础上提高影像与人口学信息融合的可靠性，并与 APIC v3_2 的既有
实验结果建立可追溯的对照关系。

本协议不支持以下表述：

- 扫描参数的因果校正；
- 1.5T/3T 或厂商效应已经被模型识别并消除；
- IE-CAPM 已经证明具有稳定域泛化增益；
- APIC v3_2 的 prototype/NO-GO 结果可以直接作为正式期刊基线。

APIC v3_2 当前仍是机制原型。IE-CAPM 的对齐主要发生在任务、数据边界、
subject 划分、方向、seed 和评估层面。

## 2. 任务与输入

### 2.1 任务

- 任务：`MCI_vs_AD`
- 标签：`MCI=0`，`AD=1`
- 方向：`ADNI_to_NACC`、`NACC_to_ADNI`
- 预处理：`skullstrip+n4+mni+crop+normalize`

### 2.2 表格变量

正式 IE-CAPM 实验固定使用 3 个变量，顺序不可改变：

```text
[age, sex, education]
```

所有表格模型均使用完全相同的三列、缺失值处理、归一化和列顺序：

- CAPM；
- IE-CAPM；
- shared image-gate ablation；
- table-conditioned gate negative control。

不使用 MMSE、CDRSB、ADAS11、FAQ、APOE4、site、scan、vendor、protocol 或
field-strength 作为模型输入。Image-only baseline 不接收任何表格输入。

实例化接口：

```python
from Model.backbone import resnet18_ie_capm

var_specs = [
    {"name": "age", "type": "continuous", "min_val": 55, "max_val": 95, "n_bases": 6},
    {"name": "sex", "type": "categorical", "n_cats": 2, "n_bases": 2},
    {"name": "education", "type": "continuous", "min_val": 0, "max_val": 22, "n_bases": 4},
]

model = resnet18_ie_capm(
    txt_dim=3,
    num_classes=2,
    var_specs=var_specs,
)
```

## 3. 3T/1.5T 配对 subject 排除

### 3.1 排除文件

使用：

```text
data/claim/paired_holdout_subjects.json
```

使用其中的：

```text
subjects_all_paired
```

该集合包含 73 个同时具有 1.5T 和 3T 记录的 ADNI subject。

不能替换成：

- `subjects_le_30d`：33 个 subject；
- `subjects_le_7d`：25 个 subject。

### 3.2 执行位置与范围

排除必须发生在任何 train/validation/test split 之前，并且是 subject-wide：

1. 读取完整 ADNI metadata/scan manifest；
2. 读取 `subjects_all_paired`；
3. 删除这些 subject 的全部扫描、全部访视和全部相关表格记录；
4. 对剩余 subject 执行后续划分。

当 ADNI 是 source 或 target 时都执行该排除。当 NACC 是 source 或 target
时，使用全部合格 NACC subject；不将不存在的 ADNI 3T/1.5T 配对规则强加
到 NACC。

此排除只用于避免同一人的 1.5T/3T 配对造成泄漏或过于容易的配对分析，
不是把 field strength 作为模型特征。

## 4. Subject-level 数据划分

### 4.1 两个方向

| direction | source | source split | external target |
| --- | --- | --- | --- |
| `ADNI_to_NACC` | 排除 73 个配对 subject 后的 ADNI | subject-level 6/2/2 | 全部合格 NACC |
| `NACC_to_ADNI` | 全部合格 NACC | subject-level 6/2/2 | 排除 73 个配对 subject 后的 ADNI |

### 4.2 Split 规则

- `split_seed=42`；
- source 按 subject 划分为 train 60%、validation 20%、test 20%；
- 同一 subject 的所有 scan/visit 必须进入同一 partition；
- 不允许 subject 在 source train、validation、test 间重复；
- target 不参与训练、early stopping、正则化权重选择、gate calibration 或
  checkpoint selection；
- IE-CAPM 不从 target 额外 carve mechanism subset，target 保持完整的最终
  外部评估集。

这里的 6/2/2 是 subject-level 比例，不是 scan 数量比例。

## 5. 运行配置

与 APIC v3_2 revision 4 对齐的基础运行配置：

| 项目 | 固定值 |
| --- | --- |
| train batch size | 4 |
| evaluation batch size | 2 |
| epochs | 50 |
| learning rate | `1e-4` |
| weight decay | `1e-4` |
| classification loss | class-weighted cross-entropy |
| metadata matching window | 90 days |
| diagnosis mismatch | exclude |
| seeds | `42`, `43` |
| split seed | `42` |

IE-CAPM 和 CAPM 对照必须保持相同的 backbone depth、classifier head、训练
预算、augmentation、checkpoint 选择规则和随机种子。

## 6. 比较矩阵

### 6.1 IE-CAPM 内部主比较

| ID | model | purpose |
| --- | --- | --- |
| B0 | image-only ResNet | 测量纯影像基线 |
| B1 | CAPM with age/sex/education | 主要表格融合基线 |
| B2 | IE-CAPM with `force_capm=True` | 同一 IE-CAPM 参数下的严格无门控控制 |
| P1 | IE-CAPM | 主要候选方法 |

B1 与 B2 必须先进行数值和指标一致性检查。若 B2 不能复现无门控 CAPM
路径，则 P1 不得进入性能解释。

### 6.2 扩展消融

只有 B1/P1 完成后才运行：

- A1：CAPM + 一个共享 image gate，用于检验逐变量门控是否必要；
- A2：由 table 直接生成 gate，用作 table-shortcut negative control。

### 6.3 与 APIC v3_2 的关系

APIC v3_2 已有结果可以作为历史协议参考，但不能直接当作严格公平基线，
原因是：

1. APIC v3_2 的 `*_x` 主要是 image-only 体系，而 IE-CAPM 使用三个人口学变量；
2. 两者 backbone capacity 和模型结构不同；
3. APIC v3_2 原型存在 target carve、prototype bank 和机制阶段，IE-CAPM 不需要
   target mechanism fitting；
4. APIC v3_2 当前 `formal_run_allowed=false`，且 M0 结论为 NO-GO。

因此，若要声称“IE-CAPM 优于 APIC v3_2”，必须使用同一份排除后 manifest、
同一 split、同一 target 边界、同一 checkpoint 规则重新运行 APIC v3_2 对照。

## 7. 评估与统计

### 7.1 主终点

主终点固定为：

```text
external target + subject_mean aggregation + balanced accuracy
```

配置约定：

```text
aggregate=subject_mean
cluster_by_subject=true
label_conflict=earliest_visit
bootstrap_samples=200
```

重复 scan 不作为独立 subject。subject-level bootstrap 也不能把同一 subject
的 scan 分开重采样。

### 7.2 辅助指标

- AUROC；
- sensitivity / specificity；
- calibration；
- demographic subgroup gaps；
- 每个 subject 的预测和 paired prediction change；
- IE-CAPM 每一 stage 的 gate、raw field、effective field 和 modulation summary。

## 8. Checkpoint 与停止规则

### 8.1 选择规则

- checkpoint 只由 source validation 选择；
- 所有模型使用完全相同的选择规则；
- 不允许使用 external target 选择最优 epoch；
- 不允许因为 target 指标更好而回溯调整训练配置。

### 8.2 Screening gate

P1 只有在以下条件都满足时才进入下一轮：

1. B2 与 B1 的无门控结果一致；
2. P1 在两个 direction、两个 seed 下方向一致；
3. P1 在固定三变量条件下没有明显性能损失；
4. gate 统计在 seed 间具有可复现性；
5. A2 不能单独解释 P1 的增益。

这只是两 seed 的协议筛选，不等价于五 seed 的期刊稳定性结论。

如果 B2 失败、P1 不满足 paired criterion、模型发生类别塌缩或 gate 机制在
seed 间不稳定，则停止扩展实验并报告 negative result。

## 9. 必须归档的实验资产

每个 `task × direction × seed × model` 单元必须保存：

- resolved config；
- split manifest 及其 hash；
- `paired_holdout_subjects.json` 的 hash；
- 实际排除的 subject ID 列表及数量；
- git commit；
- validation checkpoint 选择记录；
- subject-level predictions；
- metrics 与 200 次 bootstrap 结果；
- IE-CAPM stage-level gate summaries。

不得将 raw patient data 或未经批准的个体影像导出到 Git 仓库。
