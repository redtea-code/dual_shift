# Review 文档索引

`review/` 按文档的主要用途分类。所有新文档先确定类别，再按
`{index}_{content}_{YYYY-MM-DD}.md` 命名；`index` 在整个 `review/` 范围内递增。

## 分类规则

| 目录 | 用途 | 典型内容 |
|---|---|---|
| `plans/` | 实验开始前的决策与预注册计划 | 假设、范围、冻结条件、Gate、运行安排 |
| `records/` | 实验执行过程与产物登记 | 状态、任务认领、环境、上传清单、失败记录 |
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

### Records

| 文档 | 作用 |
|---|---|
| [`03_dual_shift_apis_3seed_status_2026-07-29.md`](records/03_dual_shift_apis_3seed_status_2026-07-29.md) | APIS 队列执行状态 |
| [`05_dual_shift_remote_claim_2026-07-29.md`](records/05_dual_shift_remote_claim_2026-07-29.md) | 远程任务认领与启动记录 |
| [`06_dual_shift_apis_3seed_remote_results_2026-07-30.md`](records/06_dual_shift_apis_3seed_remote_results_2026-07-30.md) | Linux 结果上传登记 |

### Analysis

| 文档 | 作用 |
|---|---|
| [`dual_shift_postfix_gate_results_2026-07-25.md`](analysis/dual_shift_postfix_gate_results_2026-07-25.md) | postfix Gate 结果摘要 |
| [`05_dual_shift_apis_3seed_gate_a_report_2026-07-30.md`](analysis/05_dual_shift_apis_3seed_gate_a_report_2026-07-30.md) | Windows 三种子 Gate A 报告 |
| [`07_apis_3seed_windows_vs_remote_compare_2026-07-30.md`](analysis/07_apis_3seed_windows_vs_remote_compare_2026-07-30.md) | Windows/Linux 结果对比 |
| [`08_dual_shift_apis_3seed_analysis_2026-07-30.md`](analysis/08_dual_shift_apis_3seed_analysis_2026-07-30.md) | APIS 三种子综合分析与下一步指导 |

### Operations

| 文档 | 作用 |
|---|---|
| [`04_dual_shift_remote_handoff_2026-07-29.md`](operations/04_dual_shift_remote_handoff_2026-07-29.md) | 远程实验交接与任务分发 |

## 新文档要求

1. 使用 [`docs/EXPERIMENT_RECORD_TEMPLATE.md`](../docs/EXPERIMENT_RECORD_TEMPLATE.md) 的固定字段。
2. “关联文档”和命令中的文档路径必须使用整理后的仓库相对路径。
3. 实验数据继续放在 `outputs/`；文档目录只保存 Markdown，不复制结果文件。
4. 移动或新增文档后必须同步更新本索引并检查失效引用。

`docs/` 中的规范/模板和 `Model/` 中的模型设计文档保留在各自代码所有权目录，
不归入实验生命周期分类；项目级入口见仓库根目录 `README.md`。
