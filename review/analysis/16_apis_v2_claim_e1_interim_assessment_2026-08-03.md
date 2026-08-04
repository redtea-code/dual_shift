# APIS v2 Claim E1 中期综合分析

**日期：** 2026-08-03  
**审阅范围：** `f8df40b` 与 `ecd17e9`  
**结论状态：** 中期审阅；不构成性能 claim、显著性结论或最终论文表。

## 结论摘要

当前最稳妥的判断是：**APIS v2 在相对 MixStyle 的外部 balanced accuracy 上出现了值得继续完成的正向信号，但证据尚不稳定；相对 `metadata_xda` 的大幅优势目前没有解释价值，因为该基线在目标域系统性塌缩。**

因此：

1. 不宣布 E1 通过，也不写“APIS v2 显著优于 MixStyle”。
2. 继续完成 r2 的 seed45/seed46 队列，但保持单进程并完整留存日志。
3. 将 metadata/metadata_xda 轴标记为**诊断阻断项**：先用 source-only 审计确认是训练实现、checkpoint 选择还是方法本身无效；在诊断闭合前，不把 APIS 对该轴的差值计作机制胜利。
4. 完整矩阵落盘后，必须由正式汇总脚本重新生成唯一主表与配对 bootstrap CI；中期手工表不可替代该门控。

## 1. 两个提交分别改变了什么

`f8df40b` 不是结果提交，而是确认性实验的协议加固：固定 `split_seed`、按 cohort 保存 E3 配对索引、使各变体重建相同 seed 的 sampler、对齐 metadata 与 dual-shift 的 checkpoint 选择规则，并增加产物身份检查。它解决了此前“不同变体看见不同采样序列”和“不同 seed 改变 subject split”等可比性问题。

`ecd17e9` 给出上述 r2 协议下的中期记录：38/50 个变体产物已落盘，完整双向 seed 为 42--44，另有 seed45 的 ADNI->NACC 已完成。它本身没有执行正式 report gate，也没有提供可复算的原始 r2 输出文件；本报告的数值判断因此以该记录为准，并将原始产物审计列为完成前提。

## 2. 对主终点的定量判断

主终点是 target subject-level balanced accuracy，预注册成功条件要求在**两个方向的全部 5 个 seeds**中，APIS v2 相对 MixStyle 和 `metadata_xda` 的差值为正，并满足 subject-level 配对 bootstrap CI 条件。

| 方向 | 已完成 seeds | APIS - MixStyle | 中期解释 |
|---|---:|---:|---|
| ADNI -> NACC | 4 (42--45) | mean +0.030，3/4 正 | 正向但有 seed43 的 -0.026 反例；估计 SD 约 0.043。 |
| NACC -> ADNI | 3 (42--44) | mean +0.051，2/3 正 | 正向但 seed42 基本为平局；估计 SD 约 0.045。 |

两个方向都没有出现“所有已完成 seed 均正”的模式。特别是 ADNI->NACC 的 seed43 说明 APIS 增益并非稳定地随初始化而出现。此时将均值写成方法优势，会忽略 seed 间方差与尚未计算的配对 CI。

APIS 的次要指标在已完成子集中表现健康：AUC 约 0.877--0.930、macro-F1 约 0.749--0.855，SEN 与 SPE 未显示单一方向的系统性塌缩。这支持“当前 APIS checkpoint 仍具有诊断判别力”，但并不改变主终点尚未通过的结论。

## 3. metadata 塌缩是当前最重要的解释风险

中期表中，ADNI->NACC 的 `metadata` 与 `metadata_xda` 在 4 个 completed seeds 均为 BA=0.50；NACC->ADNI 也约为 0.47--0.51。记录同时指出存在单类预测。这造成两个后果：

1. `APIS - metadata_xda` 的约 +0.30 差值主要反映基线失败，不能用来说明 APIS 的协议干预机制更好。
2. 虽然 `metadata_xda` 在输入变量上已经公平（X+D+A），它在训练动态、选模或实现上的失效会使确认性第二基线失去区分能力。

这不是“基线较弱”可以一笔带过的情况。若保持该结果，论文中最多可以如实报告“直接 concat 基线在该协议下失败”，而不能将其作为 APIS 优越性的核心证据。更合适的主叙事暂时只能围绕 APIS vs MixStyle，并且仍须等待完整预注册矩阵。

## 4. 协议与数据隔离判断

根据 r2 中期记录，以下工程性前提目前通过：

- `split_seed=42` 与 training seed 分离；报告的 ADNI->NACC seeds 42--45 训练 subject 集合一致。
- NACC->ADNI 的 E1 target 已排除配对 hold-out；记录为 E1 target n=229、E3 ADNI paired n=28。
- r1 与 smoke 产物没有并入中期表；已落盘 metrics 的 `claim_protocol_revision` 均为 2。
- `f8df40b` 已使主变体采用重建 sampler 与统一选模规则。

这些结果说明当前表具备继续执行的协议基础，但仍需在完整矩阵后由 `report_apis_v2_claim_e1.py` 对每个 run 的 split/config/target-subject 身份再次门控。中期文档中的手工汇总不能替代该检查。

## 5. 运行可靠性与可解释性边界

双 worker 并行时，NACC->ADNI 的 `ce_only` 两次约在 epoch 32 异常退出；单 worker 重开 seed44 后完整完成。现有证据更像资源、GPU 会话或并发问题，不能证明训练逻辑无误。因此剩余矩阵应使用 `max_workers=1`，并记录每个 job 的 stdout/stderr、GPU 状态、退出码和环境指纹。

此外，当前提交未包含完整 r2 原始 outputs。本报告无法独立复算表中的 BA、检查每个预测 CSV 是否与 split manifest 匹配，或验证 metadata 的预测类别分布。最终 gate 前应将下列 r2 产物归档到可审阅位置：每个 variant 的 `journal_metrics.json`、`target_predictions.csv`、run 级 `split_manifest.json`，以及 `env_fingerprint.json`。

## 6. 建议的下一步与停止规则

### 6.1 完成当前确认性矩阵

继续执行 seed45 NACC->ADNI 的剩余两个变体和 seed46 双向五变体。不要因当前 target 结果改变 APIS、MixStyle、学习率、epoch、损失权重或主终点。完成后运行：

```text
python experiments/report_apis_v2_claim_e1.py --seeds 42,43,44,45,46
```

只有正式报告同时通过完整性、split/config 身份和预注册 CI 门控，才进入 E1 Go 讨论。

### 6.2 对 metadata 进行 source-only 诊断

在不查看或优化 target 指标的条件下，对每个 metadata 变体输出并审计：

- source train/validation 的 BA、AUC、类别预测计数和混淆矩阵；
- 每 epoch 的 train/validation loss、选中 checkpoint、collapse-guard 状态；
- acquisition encoder 的 fitted 状态、OOV 比例、embedding 范数与融合前后 feature/logit 范数；
- 与 APIS 相同 backbone 上的 X-only、X+D、X+A、X+D+A source-validation 对照。

若 metadata_xda 在 source validation 同样接近单类预测，则这是实现或优化问题，应在 source-only 规则下修复并以**新的协议 revision**完整重跑该基线轴。若 source validation 正常而 target collapse，则将其报告为该直接条件化基线的外部泛化失败，不为获得更好 target 结果而调参。

### 6.3 结果解释的预注册边界

- 全部 5 seeds 未完成或任一主 CI 不满足：E1 为 No-Go，不宣称稳健性能优势。
- APIS 仅优于 metadata_xda、但不优于 MixStyle：不支持 APIS 特异机制优越性。
- APIS 优于 MixStyle、但 metadata_xda 未完成有效诊断：可报告相对 MixStyle 的结果，但不能把 metadata 比较作为强主张支撑。
- E1 通过后，E3 配对场强一致性、描述子 shuffle 负对照和 E2 协议簇 hold-out 才用于支持机制解释；它们不能补救 E1 性能门控失败。

## 7. 当前建议的内部表述

可使用：

> 在协议 r2 的不完整中期矩阵中，APIS v2 在两个迁移方向上相对 MixStyle 呈现小幅正平均 balanced-accuracy 差值，但该差异存在 seed 间不稳定性，尚未完成预注册的配对 CI 门控。直接 acquisition-metadata concat 基线发生系统性塌缩，其比较暂不用于方法机制结论。

不可使用：

> APIS v2 已显著优于 MixStyle 或公平 metadata 基线。

## 参考

- `review/records/apis_v2/3090/15_apis_v2_claim_e1_interim_2026-08-03.md`
- `journal_dual_shift_apis_v2_claim.yaml`
- `experiments/run_apis_v2_claim_e1.py`
- `experiments/report_apis_v2_claim_e1.py`
