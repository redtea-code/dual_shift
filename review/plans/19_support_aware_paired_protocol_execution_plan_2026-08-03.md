# Support-aware Paired Protocol 执行计划

日期：2026-08-03  
状态：新方法筛选计划；尚未进入确认性阶段  
权威认知：`docs/SCAN_AWARE_DATA_REALITY_AND_CLAIM_BOUNDARY.md`

## 1. 目标

在本机算力有限的条件下，先验证 scan 参数是否能定义可复用的协议关系，再决定是否值得在 3090/5090 上进行完整 3D CNN 训练。实验不再以“连续 scan 参数控制 CNN”作为默认成立的前提。

## 2. Phase D：数据与支持集审计（本机）

### D0：生成独立 screening manifest

从以下文件生成新 manifest，不修改已有 claim r2 配置：

```text
E:/2.causal/ADNI_scan_site.csv
E:/2.causal/NACC_scan_site.csv
```

只保留 `pre_folder` 非空且与当前预处理图像匹配的 ADNI 1,292 行、NACC 1,595 行。保存 manifest SHA-256、字段 schema、匹配规则和行数，不把隐私数据提交到 Git。

### D1：四类轻量探针

全部使用 subject-level split，连续字段只在 source-train 标准化：

| Probe | 输入 | 目标 | 用途 |
|---|---|---|---|
| domain | scan 参数 | ADNI/NACC | 测量 cohort shortcut |
| diagnosis | scan 参数 | CN/AD 或 MCI/AD | 测量标签混杂 |
| missingness | 缺失掩码 | domain/diagnosis | 测量缺失模式泄漏 |
| support | scan 参数 | propensity/邻域距离 | 划分 supported/unsupported |

### Gate D

- manifest 行数、subject 数、场强数量与描述性统计一致；
- 单位和异常值经审计；
- common-support 3T 子集有明确 subject 数；
- diagnosis probe 和 missingness probe 被纳入后续负对照；
- 未通过时停止模型扩展。

## 3. Phase P：配对与协议簇机制实验（本机/缓存特征）

### P1：ADNI 同人场强一致性

使用 73 名同时有 1.5T/3T 的受试者；主分析采用预先冻结的最近日期配对，另报告时间间隔敏感性分析。配对 subject 不进入对应训练集。

比较：

```text
ce_only
mixstyle
apis_v2
apis_v2_shuffle
bounded_film_scan
support_aware_paired
```

报告 `|delta probability|`、embedding cosine/distance、agreement、分类性能和 collapse 指标。

### P2：Leave-one-protocol-cluster-out

在 ADNI 内以 manufacturer×field-strength×sequence-family 定义协议簇。协议簇 subject-disjoint；小于预注册最小 subject 数的簇只做描述，不进入主比较。

### Gate P

- 候选必须在至少两个可用协议 hold-out 上相对 ce_only/MixStyle 改善；
- acquisition shuffle 不能复现收益；
- paired consistency 改善时 sensitivity/specificity 均不得塌缩；
- 只允许一个候选进入远程筛选。

## 4. Phase S：远程两 seed 筛选

### 4.1 计算矩阵

```text
tasks      = CN_vs_AD, MCI_vs_AD
directions = ADNI_to_NACC, NACC_to_ADNI
seeds      = 42, 43
variants   = ce_only, mixstyle, apis_v2, apis_v2_shuffle,
             bounded_film_scan, support_aware_paired
```

复用已有有效 r2 checkpoint；同一 backbone/seed/split 只导出一次 frozen embedding。本机完成轻量 head、统计和制表。

### 4.2 分层输出

- `target_full`；
- `target_common_support_3T`；
- `target_unsupported_protocol`；
- supported/unsupported 样本数；
- paired consistency。

### Gate S

- common-support 中相对 ce_only 和 MixStyle 的两 seed 平均 ΔBA 为正；
- 相对 acquisition shuffle 为正；
- target_full BA 退化不超过 0.02；
- MCI vs AD 独立判断，不用 CN vs AD 替代；
- 未通过时不运行 seed44--46。

## 5. Phase C：确认性实验

冻结唯一候选、配置、split、阈值和统计脚本后运行 seed42--46。

某一 task×direction 的显著提高要求：

- 相对 ce_only 与 MixStyle 的 subject-level paired ΔBA 均为正；
- paired bootstrap 95% CI 均不含 0；
- acquisition shuffle 不产生同等收益；
- calibration 和类别 recall 未发生预注册的灾难性退化。

“普遍提高”只有在四个 task×direction 单元至少三个显著为正，且第四个不支持大于 0.02 的实质性负向退化时才可使用。unsupported-protocol 单元不计入该门槛。

## 6. 算力分配

| 环境 | 工作 |
|---|---|
| 本机 | manifest、探针、support、配对分析、缓存特征 head、统计制表 |
| 3090/5090 | frozen embedding、seed42/43 筛选、通过后唯一候选 5-seed 训练 |

禁止在本机重复完整 3D CNN 矩阵；禁止在 Gate P/S 之前启动 seed44--46。

## 7. 近期任务清单

- [ ] 从 `E:/2.causal/*_scan_site.csv` 生成 screening manifest；
- [ ] 修正 audit 工具对 `tr_ms`/`tr_raw` 和 matrix/slices 字段的映射；
- [ ] 运行 domain/diagnosis/missingness/support probes；
- [ ] 冻结 73 名 ADNI 场强配对集合及排除列表；
- [ ] 实现 support-aware gate 和 paired consistency loss；
- [ ] 导出 seed42/43 frozen embedding；
- [ ] 通过 Gate S 后再创建确认性配置。

