# APIC-V3_2-P0：模型修复与分阶段实验计划

## 基本信息

- 日期：2026-08-04
- 负责人：cyh / 待认领实现与远程实验协作者
- 状态：计划中；机制原型已实现并完成首轮审阅，正式 revision 4 实验尚未获准
- Git commit：`2fe07a2`（APIC v3 诊断归档基线；本计划创建时读取 `origin/main`）
- 关联文档：`Model/APIC_V3_2_MODEL_DESIGN_DRAFT.md`、`docs/SCAN_AWARE_DATA_REALITY_AND_CLAIM_BOUNDARY.md`、`review/operations/24_apic_v3_failure_diagnostics_2026-08-04.md`
- 关联数据文件：`review/records/apic_v3/3090/apic_v3_failure_diagnostics_2026-08-04/`、`review/records/apic_v3/5090/apic_v3_failure_diagnostics_2026-08-04/`

### 实现审阅状态

当前 `v3_2_balanced_style_memory` 是用于关闭工程接口和机制单测的 prototype。正式 E0--E4
矩阵受 `review/analysis/27_apic_v3_2_implementation_review_2026-08-04.md` 阻断。当前允许：

- 静态检查、单元测试、合成数据机制开发；
- `run_apic_v3_2_screening.py --fingerprint-only`；
- 显式传入 `--allow-prototype-run` 的非主张开发运行。

当前禁止把 prototype 产生的指标登记为 revision 4 正式结果。只有审阅报告中 P0/P1 条目全部
关闭、重新验证并形成新 commit 后，才能把两个配置的 `formal_run_allowed` 改为 `true`。

## 1. 实验指导与依据

### 1.1 研究问题

在不使用 acquisition metadata、target 标签或 target 统计量的前提下，APIC v3_2 是否能先
消除 APIC v3 的 prototype collapse 与近恒等干预，并在相同数据、模态、backbone 和训练预算下，
获得比 clean CE 与 MixStyle 更稳定的 ADNI/NACC 双向泛化？

### 1.2 假设与基线

核心假设分为两个不可互相替代的层次：

- 机制假设：固定 style teacher、平衡 source prototype、support-aware gate 和 supported-only
  RMS band 能产生非零、有界、可回退的浅层干预；
- 性能假设：只有在机制假设成立后，该训练增强才可能改善 external target 的 subject-level BA。

公平基线为 `ce_x` 与 `mixstyle_x`。APIC v3 原版 `apic_v3_x` 只作为失败机制对照，不作为
v3_2 通过性能 Gate 的替代基线。

### 1.3 实施依据

3090 CN 与 5090 MCI 共八个 APIC v3 checkpoint 的诊断显示：

- gate 约 `0.03`，实际 layer RMS、JS、embedding distance 和 flip 接近 0；
- 7/8 target job 的 shifted BA 与 clean BA 完全一致；
- prototype assignment 普遍有效槽位不足或单槽支配；
- 原版目标中的一致性项与 intervention shrinkage 共同允许零干预捷径；
- clean 与 shifted 均存在明显的 train/validation 或 train/target 过拟合。

因此本计划先做 source-only 机制证伪，再进入外部 target 性能筛选，不允许直接扩大 seed 或启动
X+D。

### 1.4 实验范围

```text
tasks      = CN_vs_AD, MCI_vs_AD
directions = ADNI_to_NACC, NACC_to_ADNI
modalities = X only
split_seed = 42
train_seed = 42, 43 (仅在 Gate M0 后)
baselines  = ce_x, mixstyle_x
candidate  = apic_v3_2_x
```

标签、subject-level split、73 名 ADNI 配对 holdout 排除、预处理和 target 隔离规则继续沿用
APIC v3 screening revision 3。任何必要变更必须生成新 manifest 和 hash，并让三个方法共同重跑。
revision 4 明确令通用图像 augmentation 关闭；MixStyle 与 APIC 分别只启用其方法自身的训练期
变换，CE 不启用额外变换。若未来改变通用 augmentation，必须升高 protocol revision 并让三种
方法全部重跑，不能只对 APIC 改动。

### 1.5 冻结命名

```text
method_family    = APIC_v3_2
code_variant     = v3_2_balanced_style_memory
experiment_name  = apic_v3_2_x
protocol_revision = 4
output_root      = outputs/journal/apic_v3_2_screening_<task>/r4
```

旧 `APIC_v3` 代码、配置、checkpoint 和结果目录只读，不原地修改或覆盖。

## 2. 分阶段实验

### 2.1 E0：实现与数值回归

目的：证明新增分支没有破坏 clean path，且机制接口可审计。

测试项：

1. `apis_active=false` 时，新旧 clean logits 与 embedding 在容差内一致；
2. checkpoint reload 后 clean probability 与导出文件在 `1e-5` 内一致；
3. unsupported mask 为 0 时，layer1/layer2 shifted feature 与 clean feature 完全一致；
4. APIC residual 为 0 时，最终 shifted embedding/logits 与 clean 值在固定容差内一致；
5. shifted branch 逐层复用同一次 clean forward 的 BN batch moments，且不增加 running-stat
   update counter；专门测试“clean 用 batch moments、shifted 用 running moments”的错误实现会失败；
6. dropout/随机层复用同一 mask 或在双路径同时禁用；改变 DataLoader worker 数不改变目标原型选择；
7. 所有损失在 batch 无支持样本时退化为 clean CE，且无 NaN；supported/unsupported 每个样本的
   分类权重和均为 1；辅助损失按完整 batch 大小而非 supported count 归一化；
8. prototype/PCA 只由 source-train 的 `mechanism_fit` 构建，calibration/validation/test/target
   调用不会更新状态；
9. 多扫描受试者使用 subject-balanced 权重建立 prototype，零 IQR、低秩 PCA 和校准簇支持不足
   均有确定性处理或显式失败状态；
10. checkpoint selector 对 CE、MixStyle、v3_2 的 performance 部分产生相同语义；APIC mechanism
    guard 失败不会触发重选 checkpoint。

E0 任一关键项失败即停止，不进入 GPU 真实数据 smoke。

### 2.2 E1：合成样式机制实验

使用 source-train 的 `mechanism_fit` 图像构造已知、标签不变的 style pair：gamma、平滑 bias field、轻度 blur 和
对比度变化。另构造极端 OOD 与无合法替代原型两类 null control。

主要观察：

- 变换前后 style descriptor 距离能区分已知 style pair；
- target-stat intervention 使 shallow statistics 向指定 prototype 靠近；
- supported 样本的 layer RMS 落入冻结 band，而不是接近 0；
- OOD/null control 严格 clean fallback；
- deep feature 和预测保持在预注册容差内。

E1 只使用 source 数据，不读取 source-validation 分类指标或 external target 指标。用于冻结
`g_min/g_max`、`alpha_l` 和 support radius 的统计只能来自预先划定且 subject-disjoint 的
`mechanism_calibration`；RMS band 已在 E1 前固定，冻结后不再根据 E2 分类结果调整。

E1 启动前必须固定 layer1 `[0.001, 0.03]`、layer2 `[0.003, 0.05]` 的 RMS band，并提交、hash
一个有限的 `calibration_grid.yaml`，列出允许比较的 radius 分位数、`g_min/g_max`、`alpha_l` 与
ramp 候选；RMS band 不进入搜索。选择规则固定为：先淘汰 null fallback、BN 对照失败者以及
不满足 calibration-level effective slot、occupancy、support fraction 和 RMS band 基本条件的候选，
若无候选则 E1 失败；再要求浅层 target progress 达标，然后最大化两层中较低的 RMS band hit rate，最后
依次以较低 deep cosine drift、较低强度和配置字典序打破并列。选择过程不读取标签、
source-validation 分类指标或 external target；不得在查看 E1 输出后向网格追加候选。选出的唯一
配置写入 `r4_frozen_mechanism.yaml`，供 E2a/E2b/E3 原样复用。

### 2.3 E2a：source-only 单元消融

固定 `CN_vs_AD × ADNI_to_NACC × seed42`，只查看 source-train/source-validation：

| 变体 | 固定 teacher/bank | 显式统计目标 | 样本 support + 校准强度 | 配对 BN moments | 目的 |
|---|---|---|---|---|---|
| `v3_reference` | 否 | 否 | 否 | 否 | 复现近恒等诊断 |
| `v3_2a_no_sample_support` | 是 | 是 | 否 | 是 | 验证 support mask 的增量作用 |
| `v3_2a_full` | 是 | 是 | 是 | 是 | 验证完整主机制 |

`v3_2a_no_sample_support` 仅用于 source-only 消融：prototype 仍须满足支持量与原型间距离条件，但
不执行逐样本 `d(s_i,c_src) <= r_src` 检查；其 gate 固定为 `(g_min+g_max)/2`。该调试变体不得进入
E3，也不得用于选择主版本超参数。

此阶段不根据 external target 选择变体或超参数。`v3_2a_full` 未通过 Gate M0 时，停止并归档；
不通过提高 `alpha_max` 反复试验来追逐 target。

### 2.4 E2b：四单元机制预检

E2a 完成后，在不查看 external target 的条件下，对以下四个 source 条件分别执行
`B0 + seed42 mechanism smoke`：

```text
CN_vs_AD  x ADNI source
CN_vs_AD  x NACC source
MCI_vs_AD x ADNI source
MCI_vs_AD x NACC source
```

四个单元必须分别通过 prototype 初始化、校准支持量、occupancy、supported fraction、RMS、
strict fallback 和 BN 对照检查。阈值对四个单元完全相同；某个单元失败时不得针对该任务或方向
临时调整 K、radius、gate 或 band。E3 只有在四单元均通过 Gate M0 后才能启动。

### 2.5 E3：两 seed 双任务双向主筛选

Gate M0 通过后，重新运行公平矩阵：

```text
2 tasks x 2 directions x 2 seeds x 3 variants = 24 runs
variants = ce_x, mixstyle_x, apic_v3_2_x
```

由于 checkpoint 选择与 BN 语义已修订，CE 和 MixStyle 也必须在 revision 4 下重跑，不能直接
用 revision 3 数值替代主对照。三个变体共享完全相同的 frozen manifest、sampler、训练预算、
backbone、优化器和 source-validation performance selector。APIC v3_2 在 selector 选中的
checkpoint 上另行接受 mechanism guard，失败时直接判定该 run 失败，不得重选。三种方法共享
相同总优化 epoch；APIC 的无梯度 B0 描述符遍历只计计算开销，不计额外优化步。APIC v3_2 使用冻结教师原型间的相对统计位移，
不得把 warm-up 教师的绝对均值/标准差直接覆盖到持续更新的 student 特征上。

24 个 run 的 source-selected checkpoint、配置 hash 与 mechanism 判定全部锁定后，才允许一次性
启动 external target clean-path 推理。不得边完成单个 run 边查看 target，也不得根据部分 target
结果重新训练、改阈值或替换 checkpoint。任一 APIC run 的 selected checkpoint 未通过 mechanism
guard 时，Gate S1 自动失败；该单元不得作为缺失值从分母中删除。

### 2.6 E4：条件性稳定性扩展与预留配对评估

只有 E3 通过 Gate S1 后才允许：

- 增加 seeds 44--46，形成 5-seed 稳定性矩阵；
- 运行始终未进入模型、阈值或 Gate 选择的预留 ADNI 1.5T/3T 同人配对 Phase P；
- 按 common/unsupported scan support 做描述性分层；
- 研究 `v3_2-B` 的 adaptive balanced memory；
- 最后才考虑独立 X+D 扩展轴。

新增训练 seed 仍复用 E3 的 external target，因此只评估优化稳定性，不构成独立确认或显著优效
证据。Phase P 可作为预留的一致性评估，但不等同于跨队列诊断泛化确认。真正的确认性泛化结论
需要另一个从未用于 E3/S1 的 target subject 集、时间批次或独立队列。配对与扫描分层只能支持
一致性和关联性描述，不解释为扫描参数因果效应。

## 3. 预注册判定标准

### 3.1 Gate M0：机制进入条件

Gate 条目按其适用实验判定：E1 必须满足第 3--6、8 项；E2a 的 `v3_2a_full` 与四个 E2b
source-only 单元必须满足第 1--7、9 项。`v3_reference` 和消融分支不承担主候选的 M0 通过义务：

1. 在 `mechanism_calibration` subject 上，`effective_slots >= 3`，且 K=4 时最大 slot subject
   share 不超过 `0.70`；有效 slot 定义为校准支持量达到 `n_min` 的 slot；
2. `mechanism_calibration` 与 source-validation 的合法支持样本比例均位于 `[0.20, 0.90]`，
   避免全关或全开；
3. supported 样本 layer1/layer2 median relative RMS 均进入预注册 band；
4. supported 样本 RMS band hit rate 不低于 `0.80`；
5. unsupported/null control 的最终 embedding/logits 与 clean path 在注册容差内一致，非零 shift
   比例为 0；
6. shifted BN running-stat update 次数为 0，且逐层确认复用了配对 clean forward 的 batch moments；
7. source-validation clean/shifted 不出现单类预测、NaN 或任一 class recall `<0.15`；逐类
   supported fraction 差值不超过冻结的 `max_class_support_gap`，assignment-label NMI 不超过
   冻结的 `label_nmi_max`。NMI 使用 subject-majority hard assignment、每个 subject 一次计权和
   `normalized_mutual_info_score(average_method="arithmetic")`；这些标签指标只作拒绝门槛，不反馈
   到原型或超参数；
8. 合成 pair 的目标浅层统计距离相对干预前至少降低 `30%`；
9. source-validation 的 `shifted_CE - clean_CE <= 0.10`，两项均按同一 supported subject 集计算。

第 1 项衡量的是算法分配健康度，不代表原型一定对应真实扫描协议。低 NMI 也不能单独证明成功
解耦；高于冻结上界的 label NMI 则说明原型可能在编码疾病相关结构，主候选不得继续。

### 3.2 Gate S1：两 seed 性能条件

主要终点固定为 external target、`subject_mean`、balanced accuracy。复用 APIC v3 原筛选的
决策门槛用于纵向参照，但 revision 4 的 checkpoint/BN 协议已经改变，因此不把两版绝对数值视为
同一实验协议下的直接比较：

1. 四个 task×direction 中，v3_2 相对 CE 和 MixStyle 的两 seed 平均 delta BA 均为正的单元
   不少于 3 个；
2. 八个 seed×task×direction 单元中，v3_2 同时胜过两个基线的单元不少于 5 个；
3. 最差 task×direction 的两 seed平均退化不超过 `0.02 BA`；
4. sensitivity、specificity 和任一 class recall 均不低于 `0.15`；
5. performance selector 选出的八个 checkpoint 均继续满足 Gate M0 中适用于 checkpoint 的
   第 1--7、9 项；
6. MCI vs AD 必须单独可解释，不能由 CN vs AD 的改善抵消。

两 seed 仅作筛选，不进行显著优效主张。Gate S1 未通过时停止性能扩展，不筛选有利 seed、方向
或亚组继续。

### 3.3 Gate R1：五 seed 稳定性条件

本节随 revision 4 一并冻结，不在查看 E3 或 seeds 44--46 结果后修改。R1 只支持稳定性判断：

- 四个 task×direction 中，五 seed 平均 delta BA 同时高于两个基线的单元不少于 3 个；
- 最差 task×direction 的五 seed 平均退化不超过 `0.02 BA`；
- 上述正向单元中至少 3/5 seed 同时胜过两个基线；
- 逐 task×direction 报告五 seed 均值、标准差和全部胜负方向，不删除不利 seed；
- 对固定 seed 的逐 subject 配对预测做 subject-clustered bootstrap，再汇总 seed；不得把 seed
  当作独立受试者扩大样本量；置信区间只作不确定性描述，不作为独立确认性检验；
- BA、macro-F1、AUC、Brier、sensitivity、specificity；
- 机制健康量与性能改善之间的关联，但不作因果解释；
- paired holdout 与普通 target 结果分开报告。

## 4. 可复现记录

- 配置文件：待新增 `journal_dual_shift_apic_v3_2_screen_cn_ad.yaml` 与 `journal_dual_shift_apic_v3_2_screen_mci_ad.yaml`
- 数据与划分版本：复用或重新冻结 task-specific manifest，并在 source-train 内冻结
  subject-disjoint `mechanism_fit/mechanism_calibration`；必须记录全部 SHA-256
- 随机种子：`split_seed=42`；E2a/E2b 为 42；E3 为 42/43；E4 条件性增加 44--46
- 环境与硬件：3090 承担 CN，5090 承担 MCI；正式比较避免同一单元跨硬件混跑
- 启动命令：待实现 launcher 后冻结并写入独立 operations 文档
- 工作区状态：正式 run 必须使用 clean commit；当前工作区 dirty，不作为运行基线
- 产物位置：`outputs/journal/apic_v3_2_screening_<task>/r4/...`
- 结果归档：`review/records/apic_v3_2/<environment>/`，不得混入 `records/apic_v3/`

每个 run 必须归档 config、split、mechanism 子集、robust scaler、PCA/prototype、BN 对照摘要、
performance selector 输入与输出、checkpoint、环境和 Git 指纹；大型 checkpoint 可只保留稳定
外部路径与 SHA-256。E1 前必须冻结 calibration grid 与选择规则；E2a 前必须生成唯一的
`r4_frozen_mechanism.yaml`。E2a/E2b/E3 前必须冻结 warm-up epoch、mechanism 子集比例、K、
`eps_sigma/eps_scale/eps_distance/eps_rms`、PCA 最大维度与 whitening 开关、weighted k-means
`n_init/max_iter/seed`、occupancy、support radius 分位数、`n_min/delta_min/delta_max`、
`g_min/g_max`、`rho_mu/rho_sigma`、相对位移裁剪分位数、`r_base_max`、`alpha_l/ramp`、RMS band、
`max_class_support_gap/label_nmi_max`、所有损失权重、collapse guard、EMA 窗口、tie-break 顺序和
数值容差；缺少任一字段时 launcher 拒绝运行。

## 5. 分析与结果

### 5.1 结果

| 方法/条件 | 主要指标 | 辅助指标 | 判定 | 备注 |
|---|---:|---:|---|---|
| `v3_reference` | 待运行 | RMS / slots / gate | 待判定 | E2a 机制对照 |
| `apic_v3_2_x` | 待运行 | Gate M0 全量指标 | 待判定 | E1/E2a/E2b 候选 |
| `ce_x` | 待运行 | F1/AUC/Brier | 待判定 | E3 公平重跑 |
| `mixstyle_x` | 待运行 | F1/AUC/Brier | 待判定 | E3 公平重跑 |

### 5.2 分析

- 相对基线：待 E3 后逐 seed、task、direction 报告 v3_2 相对两基线的绝对 delta BA；
- 异常与局限：两 seed 不能支持显著性主张；prototype 平衡不等于扫描风格因果解耦；
- 结果结论：先独立判 Gate M0，再判 Gate S1；性能改善不能覆盖机制失败。

## 6. 建议下一步实验指导

- 建议动作：先实现 `APIC v3_2-A`，完成 E0 单元测试和 E1 合成机制实验；
- 建议依据：现有八个 checkpoint 已一致定位近恒等和 memory collapse，不需要继续扩展 v3 seed；
- 固定条件：image-only、source-only 原型、clean inference、数据划分、基线和主终点保持不变；
- 进入条件：E2b 四单元 Gate M0 全部满足后才启动 24-run E3，Gate S1 通过后才启动五 seed
  稳定性扩展与 v3_2-B；
- 禁止事项：不查看 target 调门控/RMS/K；不强制 unsupported 样本产生扰动；不在首轮加入 label adversary；不启动 X+D；不覆盖 APIC v3 历史结果。
