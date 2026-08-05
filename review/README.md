# Review 文档索引

`review/` 按文档的主要用途分类。所有新文档先确定类别，再按
`{index}_{content}_{YYYY-MM-DD}.md` 命名；`index` 在整个 `review/` 范围内递增。

## 分类规则

| 目录 | 用途 | 典型内容 |
|---|---|---|
| `plans/` | 实验开始前的决策与预注册计划 | 假设、范围、冻结条件、Gate、运行安排 |
| `records/` | 实验执行过程与产物登记；版本化实验按方法再分层 | 状态、任务认领、环境、上传清单、失败记录 |
| `analysis/` | 基于结果数据的分析与结论 | 指标汇总、Gate 判定、环境对比、综合结论 |
| `operations/` | 不直接承载实验结论的协作材料 | 交接、任务分发、远程操作说明 |

文档同时包含多类内容时，以主要目的归档，并通过“关联文档”链接到其他类别。
计划中的结果区填写“待运行”，完成后应另建 `records/` 或 `analysis/` 文档，不在计划文件中覆盖原预注册条件。

## 当前索引

### Plans

| 文档 | 作用 |
|---|---|
| [`01_dual_shift_next_step_decision_2026-07-29.md`](plans/01_dual_shift_next_step_decision_2026-07-29.md) | 下一步实验与论文决策边界 |
| [`02_dual_shift_experiment_schedule_2026-07-29.md`](plans/02_dual_shift_experiment_schedule_2026-07-29.md) | postfix 后续实验执行安排 |
| [`09_apis_3seed_samehost_repro_plan_2026-07-30.md`](plans/09_apis_3seed_samehost_repro_plan_2026-07-30.md) | APIS 同机三种子复现预注册计划 |
| [`10_apis_v2_claim_validation_plan_2026-07-30.md`](plans/10_apis_v2_claim_validation_plan_2026-07-30.md) | APIS v2 主张边界与验证实验方案 |
| [`11_apis_v2_local_experiment_schedule_2026-07-30.md`](plans/11_apis_v2_local_experiment_schedule_2026-07-30.md) | APIS v2 本地 smoke 安排 |
| [`14_apis_v2_claim_execution_plan_2026-07-31.md`](plans/14_apis_v2_claim_execution_plan_2026-07-31.md) | APIS v2 主张验证执行计划 |
| [`16_apis_v2_claim_e1_mci_ad_remote_plan_2026-08-02.md`](plans/16_apis_v2_claim_e1_mci_ad_remote_plan_2026-08-02.md) | APIS v2 MCI vs AD 远程执行计划 |
| [`19_support_aware_paired_protocol_execution_plan_2026-08-03.md`](plans/19_support_aware_paired_protocol_execution_plan_2026-08-03.md) | Scan-aware 支持集与配对协议筛选计划 |
| [`20_apic_v3_image_only_comparison_plan_2026-08-03.md`](plans/20_apic_v3_image_only_comparison_plan_2026-08-03.md) | APIC v3 严格 X / 扩展 X+D 多任务双向两种子对比计划 |
| [`22_apic_v3_s1_3090_cn_ad_execution_plan_2026-08-03.md`](plans/22_apic_v3_s1_3090_cn_ad_execution_plan_2026-08-03.md) | 本机 3090：仅 CN_vs_AD × 三 X 变体执行计划 |
| [`26_apic_v3_2_model_and_experiment_plan_2026-08-04.md`](plans/26_apic_v3_2_model_and_experiment_plan_2026-08-04.md) | APIC v3_2 机制修复、分阶段验证与性能筛选计划 |

### Records

版本化实验采用“方法优先、运行环境次级”的目录结构：
`records/<method>/<environment>/`。当前冻结的方法目录为 `apis_v2/`、`apic_v3/` 与 `apic_v3_2/`；
早于该约定的 Dual-Shift/APIS 历史记录保留在 `records/` 根目录。

| 文档 | 作用 |
|---|---|
| [`03_dual_shift_apis_3seed_status_2026-07-29.md`](records/03_dual_shift_apis_3seed_status_2026-07-29.md) | APIS 队列执行状态 |
| [`05_dual_shift_remote_claim_2026-07-29.md`](records/05_dual_shift_remote_claim_2026-07-29.md) | 远程任务认领与启动记录 |
| [`06_dual_shift_apis_3seed_remote_results_2026-07-30.md`](records/06_dual_shift_apis_3seed_remote_results_2026-07-30.md) | Linux 结果上传登记 |
| [`apis_v2/`](records/apis_v2/README.md) | APIS v2 smoke、claim E1 记录与机读表 |
| [`apis_v2/common/12_apis_v2_smoke_adni_to_nacc_2026-07-30.md`](records/apis_v2/common/12_apis_v2_smoke_adni_to_nacc_2026-07-30.md) | APIS v2 ADNI→NACC smoke |
| [`apis_v2/common/13_apis_v2_smoke_bidirectional_2026-07-30.md`](records/apis_v2/common/13_apis_v2_smoke_bidirectional_2026-07-30.md) | APIS v2 双向 smoke |
| [`apis_v2/3090/15_apis_v2_claim_e1_interim_2026-08-03.md`](records/apis_v2/3090/15_apis_v2_claim_e1_interim_2026-08-03.md) | 3090：APIS v2 CN claim E1 中期判断 |
| [`apis_v2/3090/16_apis_v2_claim_e1_metrics_main_table_2026-08-03.md`](records/apis_v2/3090/16_apis_v2_claim_e1_metrics_main_table_2026-08-03.md) | 3090：APIS v2 CN claim E1 全指标主表 |
| [`apis_v2/5090/17_apis_v2_claim_mci_ad_e1_interim_2026-08-03.md`](records/apis_v2/5090/17_apis_v2_claim_mci_ad_e1_interim_2026-08-03.md) | 5090：APIS v2 MCI vs AD claim E1 中期判断 |
| [`apis_v2/5090/18_apis_v2_claim_mci_ad_e1_metrics_main_2026-08-03.md`](records/apis_v2/5090/18_apis_v2_claim_mci_ad_e1_metrics_main_2026-08-03.md) | 5090：APIS v2 MCI vs AD claim E1 全指标主表 |
| [`apic_v3/`](records/apic_v3/README.md) | APIC v3 image-only screening 记录、机读表与日志 |
| [`apic_v3/3090/23_apic_v3_cn_ad_s1_metrics_2026-08-04.md`](records/apic_v3/3090/23_apic_v3_cn_ad_s1_metrics_2026-08-04.md) | 3090：APIC v3 CN_vs_AD primary 12-run 主表 |
| [`apic_v3/3090/apic_v3_screening_cn_ad_primary_logs/`](records/apic_v3/3090/apic_v3_screening_cn_ad_primary_logs/) | 3090：APIC v3 CN_vs_AD primary 训练日志 |
| [`apic_v3/3090/25_apic_v3_cn_failure_diagnosis_2026-08-04.md`](records/apic_v3/3090/25_apic_v3_cn_failure_diagnosis_2026-08-04.md) | 3090：APIC v3 CN 失败机制诊断（ops 24） |
| [`apic_v3/3090/apic_v3_failure_diagnostics_2026-08-04/`](records/apic_v3/3090/apic_v3_failure_diagnostics_2026-08-04/) | 3090：ops 24 诊断机读产物（CN 四 job 完整 layer-2） |
| [`apic_v3/3090/apic_v3_failure_diagnostics/`](records/apic_v3/3090/apic_v3_failure_diagnostics/) | 3090：ops 24 早期 seed43-only 过渡副本（以 dated 目录为准） |
| [`apic_v3/5090/19_apic_v3_screening_mci_ad_primary_2026-08-04.md`](records/apic_v3/5090/19_apic_v3_screening_mci_ad_primary_2026-08-04.md) | 5090：APIC v3 MCI_vs_AD primary 结果表 |
| [`apic_v3/5090/apic_v3_screening_mci_ad_primary_logs/`](records/apic_v3/5090/apic_v3_screening_mci_ad_primary_logs/) | 5090：APIC v3 MCI_vs_AD primary 训练日志 |
| [`apic_v3/5090/25_apic_v3_failure_diagnostics_mci_ad_2026-08-04.md`](records/apic_v3/5090/25_apic_v3_failure_diagnostics_mci_ad_2026-08-04.md) | 5090：APIC v3 MCI vs AD 失败机制诊断（ops 24） |
| [`apic_v3/5090/apic_v3_failure_diagnostics_2026-08-04/`](records/apic_v3/5090/apic_v3_failure_diagnostics_2026-08-04/) | 5090：ops 24 诊断机读产物 |
| [`apic_v3_2/`](records/apic_v3_2/README.md) | APIC v3_2（revision 4）原型实验记录 |
| [`apic_v3_2/3090/31_apic_v3_2_cn_ad_r4_prototype_metrics_2026-08-05.md`](records/apic_v3_2/3090/31_apic_v3_2_cn_ad_r4_prototype_metrics_2026-08-05.md) | 3090：APIC v3_2 CN prototype 12-run 主表 |
| [`apic_v3_2/5090/29_apic_v3_2_mci_ad_r4_primary_2026-08-05.md`](records/apic_v3_2/5090/29_apic_v3_2_mci_ad_r4_primary_2026-08-05.md) | 5090：APIC v3_2 MCI vs AD 原型主表（12/12） |
| [`apic_v3_2/5090/30_apic_v3_2_mci_ad_r4_primary_report_2026-08-05.md`](records/apic_v3_2/5090/30_apic_v3_2_mci_ad_r4_primary_report_2026-08-05.md) | 5090：APIC v3_2 MCI vs AD 执行报告 |
| [`apic_v3_2/5090/apic_v3_2_mci_ad_r4_primary_logs/`](records/apic_v3_2/5090/apic_v3_2_mci_ad_r4_primary_logs/) | 5090：APIC v3_2 MCI 训练日志 |

### Analysis

| 文档 | 作用 |
|---|---|
| [`dual_shift_postfix_gate_results_2026-07-25.md`](analysis/dual_shift_postfix_gate_results_2026-07-25.md) | postfix Gate 结果摘要 |
| [`05_dual_shift_apis_3seed_gate_a_report_2026-07-30.md`](analysis/05_dual_shift_apis_3seed_gate_a_report_2026-07-30.md) | Windows 三种子 Gate A 报告 |
| [`07_apis_3seed_windows_vs_remote_compare_2026-07-30.md`](analysis/07_apis_3seed_windows_vs_remote_compare_2026-07-30.md) | Windows/Linux 结果对比 |
| [`08_dual_shift_apis_3seed_analysis_2026-07-30.md`](analysis/08_dual_shift_apis_3seed_analysis_2026-07-30.md) | APIS 三种子综合分析与下一步指导 |
| [`27_apic_v3_2_implementation_review_2026-08-04.md`](analysis/27_apic_v3_2_implementation_review_2026-08-04.md) | APIC v3_2 原型与研究计划一致性审阅；正式 revision 4 阻断项 |
| [`32_apic_v3_2_cn_ad_r4_prototype_run_report_2026-08-05.md`](analysis/32_apic_v3_2_cn_ad_r4_prototype_run_report_2026-08-05.md) | 3090：APIC v3_2 CN prototype 运行报告（12/12） |

### Operations

| 文档 | 作用 |
|---|---|
| [`04_dual_shift_remote_handoff_2026-07-29.md`](operations/04_dual_shift_remote_handoff_2026-07-29.md) | 远程实验交接与任务分发 |
| [`21_apic_v3_remote_screening_handoff_2026-08-03.md`](operations/21_apic_v3_remote_screening_handoff_2026-08-03.md) | APIC v3 两 seed 多任务远程筛选命令与回传规范 |
| [`24_apic_v3_failure_diagnostics_2026-08-04.md`](operations/24_apic_v3_failure_diagnostics_2026-08-04.md) | APIC v3 训练轨迹、style memory与反事实特征干预诊断脚本说明 |

## 新文档要求

1. 使用 [`docs/EXPERIMENT_RECORD_TEMPLATE.md`](../docs/EXPERIMENT_RECORD_TEMPLATE.md) 的固定字段。
2. “关联文档”和命令中的文档路径必须使用整理后的仓库相对路径。
3. 实验数据继续放在 `outputs/`；文档目录只保存 Markdown，不复制结果文件。
4. 移动或新增文档后必须同步更新本索引并检查失效引用。

`docs/` 中的规范/模板和 `Model/` 中的模型设计文档保留在各自代码所有权目录，
不归入实验生命周期分类；项目级入口见仓库根目录 `README.md`。
