# Scan-aware 跨数据集泛化：低算力执行计划

日期：2026-08-03  
状态：冻结前执行草案  
数据根目录：`E:/2.causal`  
证据文件：`E:/2.causal/ADNI_NACC_descriptive_statistics.xlsx`  
适用任务：CN vs AD、MCI vs AD

## 1. 本次纠正与核心判断

此前配置中的 `F:/ADNI/scan_manifests` 是另一台机器的路径，本机不得继续把它当作数据来源。本机可用的权威扫描级文件为：

```text
E:/2.causal/ADNI_scan_site.csv
E:/2.causal/NACC_scan_site.csv
E:/2.causal/ADNI_csv_aligned_scan_site.csv
E:/2.causal/NACC_csv_aligned_scan_site.csv
E:/2.causal/scan_site_alignment_report.md
```

描述性统计显示 scan 参数不是轻微偏移，而是严重的队列支持集差异：

| 指标 | ADNI | NACC | 对实验的含义 |
|---|---:|---:|---|
| scans | 1,292 | 1,595 | 扫描数接近，但独立 subject 数不同 |
| subjects | 342 | 1,292 | ADNI 纵向重复更多，必须 subject 聚合/划分 |
| sites | 49 | 19 | site 不能直接当作可比类别 |
| 1.5T scans | 1,108 | 0 | NACC 完全没有 1.5T 支持 |
| 3T scans | 181 | 1,595 | 共同支持主要存在于 3T |
| 同时具有 1.5T/3T 的 subject | 73 | 0 | 仅 ADNI 可做同人场强一致性验证 |
| Siemens | 55.5% | 97.2% | manufacturer 几乎是队列标识 |
| GE | 42.3% | 0 | NACC 中缺乏 GE 支持 |

协议差异也很大。例如 ADNI/NACC 的平均 TE 约为 `3.616/2.977 ms`，平均层厚约为 `1.200/1.009 mm`，平均 slices 约为 `163.1/196.9`。因此，直接 concat scan 参数很容易学习“来自哪个数据集”，并不等于学到了可迁移的成像校正。

## 2. 必须区分的两个研究问题

### 2.1 共同支持上的泛化

问题：当 source 与 target 都有相近的 3T 协议时，scan-aware 调制是否优于 ce_only、MixStyle 和 APIS v2？

这是可检验、可解释的主问题。优先使用：

- ADNI 3T -> NACC 3T；
- NACC 3T -> ADNI 3T；
- manufacturer/TE/TI/voxel geometry 的 common-support 子集或加权结果。

### 2.2 未见协议外推

问题：NACC 3T source-only 模型能否泛化到 ADNI 1.5T？

当前 source 没有任何 1.5T，观测协议原型无法从 NACC 学到 3T->1.5T 残差。该单元只能作为严格 OOD stress test，不能与共同支持结果混为一谈。若要使用未标注 ADNI 数据建立 1.5T 原型，研究范式必须明确改称 UDA/test-time adaptation，而不是严格 domain generalization。

## 3. 算力分层

本机不承担完整 3D CNN 多 seed 训练，只承担低成本审计、探针和结果汇总。

| 层级 | 硬件 | 任务 |
|---|---|---|
| Local-0 | 本机 CPU | manifest 构建、缺失率、支持集和泄漏审计 |
| Local-1 | 本机 CPU/小显存 | scan-only probe、propensity overlap、统计检验 |
| Local-2 | 本机 | 使用远程导出的 frozen CNN embedding 训练轻量 head/FiLM |
| Remote-S | 3090/5090 | seed42/43 小矩阵；只训练通过 Local gate 的模型 |
| Remote-C | 3090/5090 | 冻结模型后的 seed42--46 确认性矩阵 |

原则：本机结果只能用于筛选和排除无效方案，最终跨数据集主张必须来自远程完整 3D 训练。

## 4. Phase 0：在本机生成冻结 manifest

### 4.1 输入

- ADNI：`E:/2.causal/ADNI_scan_site.csv`，仅保留 `pre_folder` 非空的 1,292 行；
- NACC：`E:/2.causal/NACC_scan_site.csv`，仅保留可对应 `NACC_pre` 的 1,595 行；
- 图像根：`E:/2.causal/ADNI_pre`、`E:/2.causal/NACC_pre`；
- 临床匹配：沿用同 subject、最近访视、绝对日期差不超过 90 天的规则。

### 4.2 冻结产物

```text
data/local_manifests/ADNI_scan_manifest.csv
data/local_manifests/NACC_scan_manifest.csv
data/local_manifests/build_meta.json
data/local_manifests/scan_parameter_audit/*.csv
```

manifest 不提交含隐私或绝对图像路径的完整数据到 Git；只提交 schema、hash、行数、审计摘要和生成命令。

### 4.3 Gate D0

- 行数必须复核为 ADNI 1,292、NACC 1,595；
- `pre_folder` 唯一且与图像目录对应；
- `pre_label == csv_label` 的已匹配样本必须为 100%；
- field strength 只允许 1.5T、3T、unknown/other；
- 单位固定为 ms、mm、degrees；
- subject 级 train/val/test 与 hold-out 不重叠。

## 5. Phase 1：本机无 3D 训练探针

统一使用 subject-level split；同一 subject 的所有 scans 必须留在同一折。连续字段只用 source-train 统计标准化。

### 5.1 Domain probe

输入 scan 参数及 missingness mask，预测 ADNI/NACC。报告 subject-level AUC、BA 和校准。由于差异巨大，预期 AUC 很高；该结果只证明参数携带 domain 信息。

### 5.2 Label probe

在每个 source 队列内部，用 scan 参数预测 CN/AD 或 MCI/AD。若 AUC/BA 明显高于随机，说明 scan 参数与诊断标签混杂；禁止把 concat 结果解释为成像机制改善。

### 5.3 Missingness probe

只用字段是否缺失预测 domain 和 diagnosis。若性能高，模型必须保留负对照，并限制缺失掩码对诊断 head 的直接贡献。

### 5.4 Common-support/propensity 审计

用 scan 参数估计每个 subject 属于 ADNI 的 propensity。分别报告：

- overlap 区间内的 subject 数；
- 极端 propensity 比例；
- 3T-only 子集；
- manufacturer×field-strength×sequence-family 单元的双边覆盖。

### 5.5 Gate D1

只有以下条件同时满足才进入 scan-aware 训练：

1. scan 参数可以预测 domain；
2. 至少存在足够的 common-support 3T subject；
3. label probe 没有显示无法控制的诊断泄漏，或计划中明确使用 shuffle/scan-only 负对照；
4. 所有结论按 subject 计算，不使用 scan 伪增样本量。

## 6. Phase 2：低成本机制验证

### 6.1 ADNI 同人 1.5T/3T 一致性

使用 73 名同时拥有两种场强的 subject。固定 CNN，只提取 embedding；比较同一 subject 的 1.5T/3T 表征距离和预测差异：

```text
ce_only vs mixstyle vs apis_v2 vs film_scan vs apis_scan
```

成功条件：scan-aware 模型降低同人跨场强 embedding distance 和预测不一致，同时不降低诊断可分性。该实验支持“对场强敏感性降低”，不能单独证明跨队列 BA 提升。

### 6.2 Leave-one-protocol-out

在 ADNI 内按 manufacturer×field-strength×sequence-family 构造协议簇，留一协议簇作伪 target。先在 frozen embedding 上筛选调制 head，仅保留能在多个协议簇上稳定提高 BA 的候选。

### 6.3 Gate M

- `apis_scan` 必须优于 `apis_v2_shuffle`；
- 至少两个不同协议 hold-out 上 ΔBA 为正；
- 同人场强一致性改善不能来自全部预测收缩到同一类别；
- 不满足时不启动远程完整训练。

## 7. Phase 3：远程 seed42/43 筛选

远程只运行以下固定矩阵：

```text
variants = ce_only, mixstyle, apis_v2, apis_v2_shuffle, film_scan, apis_scan
tasks    = CN_vs_AD, MCI_vs_AD
seeds    = 42, 43
```

每个方向同时输出两个 target 口径：

1. `target_full`：完整外部队列；
2. `target_common_support_3T`：共同支持的 3T 子集。

NACC->ADNI 另行报告 `target_ADNI_1.5T`，但标为 unsupported-protocol stress test。

### Gate S

候选进入 5-seed 确认的最低条件：

- common-support 主表中，两个 seed 的平均 ΔBA 相对 ce_only 和 MixStyle 均为正；
- `apis_scan - apis_v2_shuffle` 为正；
- sensitivity/specificity 均不低于 0.15；
- target_full 没有超过 0.02 的实质性 BA 退化；
- MCI vs AD 若未通过，不得用 CN vs AD 结果替代。

## 8. Phase 4：远程确认性实验

冻结唯一候选结构、配置和阈值后，运行 seed42--46。主终点仍为 target subject-level BA，并进行同 subject、同 seed 的 paired bootstrap。

“显著提高”要求某个 task×direction 中，候选相对 ce_only 和 MixStyle 的 paired ΔBA 95% CI 均不含 0。

“普遍提高”要求：

- 四个 task×direction 单元中至少三个显著为正；
- 第四个单元的 CI 不支持大于 0.02 的实质性负向退化；
- common-support 与 target_full 结论方向一致；
- acquisition shuffle 不能复现主要收益。

若只在 3T common support 上通过，论文主张必须限定为“共同扫描协议支持下的跨队列泛化”，不能写成对未见协议的普遍泛化。

## 9. 任务分配与节省算力措施

### 本机

1. 构建本地 frozen manifest；
2. 运行 scan audit、domain/label/missingness probes；
3. 生成 common-support 清单；
4. 接收远程 frozen embedding 后训练轻量调制 head；
5. 统一生成规范实验表和 paired bootstrap 报告。

### 3090/5090

1. 每个 backbone/seed/split 只导出一次 frozen embedding，后续轻量试验复用；
2. 先只跑 seed42/43，不提前运行 seed44--46；
3. CN vs AD 先行，MCI vs AD 在 Gate S 后运行；
4. 只有一个候选进入 5-seed 确认；
5. 不同时启动完整 ce_only/MixStyle/APIS 重复作业，优先复用协议 r2 已有有效 checkpoint。

## 10. 近期执行清单

- [ ] 将配置中的 manifest 根迁移到本地生成的 `data/local_manifests`，但不修改已有 r2 结果配置；新建独立 screening 配置。
- [ ] 运行 Phase 0 审计并提交不含隐私的摘要。
- [ ] 完成 domain、label、missingness、propensity 四类探针。
- [ ] 构建 ADNI 73 名配对场强 subject 清单。
- [ ] 远程导出 seed42/43 frozen embedding。
- [ ] 本机完成机制筛选并确定唯一远程候选。
- [ ] 通过 Gate S 后再启动完整 5-seed 矩阵。

