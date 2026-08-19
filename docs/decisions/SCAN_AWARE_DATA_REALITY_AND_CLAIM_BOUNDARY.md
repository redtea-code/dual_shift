# Scan-aware 数据现实、可识别性与主张边界

版本：2.0  
日期：2026-08-03  
状态：当前权威认知  
数据根目录：`E:/2.causal`  
证据：`E:/2.causal/ADNI_NACC_descriptive_statistics.xlsx`

本文取代以下已删除的中间认知文档：

- `review/plans/17_apis_v2_scan_parameter_generalization_plan_2026-08-03.md`
- `review/plans/18_scan_aware_low_compute_execution_plan_2026-08-03.md`

早期 `review/plans/01--16` 和 `review/records/` 是预注册与实验审计历史，不代表当前方法主张，不应删除或覆盖。它们位于 Git ref `archive/pre-plan34-cleanup-20260809`，不是当前 `main` 的工作文档。

## 1. 数据事实

### 1.1 数据规模

| Dataset | Scans | Subjects | Sites | Scans/subject mean |
|---|---:|---:|---:|---:|
| ADNI | 1,292 | 342 | 49 | 3.78 |
| NACC | 1,595 | 1,292 | 19 | 1.23 |

所有划分、bootstrap 和指标必须以 subject 为单位。ADNI 的纵向重复明显更多，scan-level 样本数不能当作独立样本量。

### 1.2 场强支持集

| Dataset | 1.5T scans | 3T scans | Unknown/other | 同时有 1.5T/3T 的 subjects |
|---|---:|---:|---:|---:|
| ADNI | 1,108 | 181 | 3 | 73 |
| NACC | 0 | 1,595 | 0 | 0 |

NACC 完全没有 1.5T。NACC->ADNI 时，ADNI 1.5T 属于严格 source-only 设置下的未见协议，不能用“从 source 学到的场强响应”解释结果。

### 1.3 零方差与近零方差参数

NACC 有效记录中：

- field strength 全部为 3T；
- TI 全部为 900 ms；
- flip angle 全部为 9 degrees；
- matrix rows 全部为 256；
- sequence family 全部为 MPRAGE；
- TE、pixel spacing、slice thickness 只有很小变化。

此外，NACC 97.2% 为 Siemens，而 ADNI 同时包含 Siemens、GE 和 Philips。多个 scan 参数几乎等价于 cohort 标识。

### 1.4 数据来源

本机权威输入为：

```text
E:/2.causal/ADNI_scan_site.csv
E:/2.causal/NACC_scan_site.csv
E:/2.causal/ADNI_csv_aligned_scan_site.csv
E:/2.causal/NACC_csv_aligned_scan_site.csv
E:/2.causal/scan_site_alignment_report.md
```

配置中的 `F:/ADNI/scan_manifests` 是另一台机器的路径，不是本机数据源。新实验必须从上述文件生成新的 frozen manifest，不能修改旧 r2 结果的配置或指纹。

## 2. 可识别性结论

### 2.1 当前数据可以支持

- 检验普通图像模型能否跨 ADNI/NACC 泛化；
- 检验 ADNI 内部协议多样性是否有助于 ADNI->NACC；
- 在 3T common support 上比较 scan-aware 方法；
- 利用 ADNI 同人 1.5T/3T 扫描验证预测与表征一致性；
- 检查 scan 参数、缺失模式和 cohort/diagnosis 的混杂；
- 把 NACC 3T->ADNI 1.5T 作为 unsupported-protocol stress test。

### 2.2 当前数据不能支持

- 从 NACC 估计连续 field-strength 响应；
- 将 field strength 与 manufacturer/site/sequence 的作用分离为因果效应；
- 证明 scan 参数控制对任意未见协议普遍有效；
- 仅凭跨队列 BA 提升证明模型学到了 acquisition mechanism；
- 把 direct concat、FiLM 或 metadata baseline 的结果解释为 scan 参数因果贡献。

### 2.3 方向不对称

ADNI->NACC 与 NACC->ADNI 不是对称实验：

- ADNI source 包含 1.5T/3T、多个厂家和协议，可学习有限的 source-observed invariance；
- NACC source 几乎是单一 3T/MPRAGE/Siemens 支持，无法学习 1.5T 或 GE 响应；
- 因此，双向平均不能掩盖 NACC->ADNI 的支持集外推问题。

## 3. 旧方法认识为何需要调整

随机 MixStyle 不使用 scan 参数，也没有在 MCI vs AD 上稳定优于 ce_only。直接 metadata concat 容易把 scan 参数当作 cohort shortcut。受约束 FiLM 虽能限制幅度，但在零方差 source 上仍无法识别参数效应。

当前 APIS v2 的 source-observed prototype residual 边界是正确的：没有 source prototype 时应返回 clean path，而不是外推。但仅靠 prototype residual 仍不足以证明 scan 参数带来了普遍性能改善。

因此，新方法不再以“scan 参数直接控制诊断特征”为中心，而以以下目标为中心：

> 在观测协议共同支持范围内，利用 scan 参数定义协议关系，通过同人配对一致性、协议簇稳健性和支持感知门控降低 acquisition-induced prediction instability。

## 4. 新方法方向：Support-aware Paired Protocol Invariance

### 4.1 Paired protocol consistency

在训练集之外保留 ADNI 同人 1.5T/3T 配对集合。对同一 subject 的两次扫描报告并约束：

```text
embedding distance
|p_1.5T - p_3T|
prediction agreement
diagnostic separability
```

一致性改善必须同时保持诊断判别力，防止模型通过输出常数获得虚假一致性。

### 4.2 Protocol-cluster robustness

用 manufacturer×field-strength×sequence-family 定义 source-observed 协议簇，执行 leave-one-protocol-cluster-out。scan 参数主要用于定义环境、距离和一致性约束，不直接作为无约束诊断输入。

### 4.3 Support-aware gate

模型必须估计输入协议是否落在 source 支持范围：

- supported：允许有界 residual/FiLM 调制；
- unsupported：回退到 image-only clean path；
- 始终输出支持状态和不确定性；
- 不得把 unsupported 输入映射到任意“最接近”原型后宣称完成校正。

### 4.4 Negative controls

每个方法结论必须同时包含：

- scan-only；
- missingness-only；
- acquisition shuffle；
- image+scan concat；
- bounded FiLM；
- clean image-only。

如果 shuffle 能复现收益，或 scan-only 已能预测 diagnosis，scan-aware 性能差异不能解释为 acquisition mechanism。

## 5. 重新定义实验终点

### 5.1 性能主终点

`target + subject_mean + balanced_accuracy`，按 task 和 direction 分别报告，不先合并方向。

### 5.2 机制终点

- ADNI 同人 1.5T/3T 的预测概率绝对差；
- paired embedding distance；
- protocol-cluster hold-out BA；
- supported/unsupported 分层性能；
- scan/domain probe 与 diagnosis probe。

### 5.3 结果口径

必须分别输出：

1. `target_full`；
2. `target_common_support_3T`；
3. `target_unsupported_protocol`；
4. paired field-strength consistency。

共同支持结果不能替代 full target；unsupported stress test 不能计入“显著提高”的主要成功单元。

## 6. 主张门槛

### 支持的表述

> 在观测协议共同支持范围内，scan-parameter-defined consistency 与支持感知调制降低了协议相关预测不稳定性，并改善了外部队列 subject-level balanced accuracy。

### 当前禁止的表述

- “模型学习了 MRI 扫描参数的因果响应”；
- “模型可以从 NACC 学到 1.5T 校正”；
- “scan 参数显著或普遍提高任意跨数据集泛化”；
- “双向平均提高证明方法对所有协议有效”。

只有新增具有多协议变化且双边重叠的数据集，或改变为明确的 multi-source/UDA 设置后，才允许重新评估更强主张。

## 7. 文档优先级

发生冲突时按以下优先级解释：

1. 本文的数据现实与主张边界；
2. `review/plans/19_support_aware_paired_protocol_execution_plan_2026-08-03.md`；
3. 实验表规范和当前代码设计文档；
4. `review/plans/01--16` 仅作为历史预注册/审计记录。

