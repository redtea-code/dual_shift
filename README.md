# dual_shift

Dual-Shift 实验代码、结构化结果与研究记录仓库。

## 项目导航

| 路径 | 内容 |
|---|---|
| `Model/` | 模型实现与模型级设计文档 |
| `data/` | 数据加载和预处理源码 |
| `experiments/` | 实验与结果导出脚本 |
| `utils/` | 公共工具代码 |
| `outputs/` | 已纳入版本控制的结构化指标、预测和审计结果 |
| [`review/`](review/README.md) | 实验计划、执行记录、结果分析和协作交接 |
| `docs/` | 团队协作规范与固定文档模板 |

## 文档入口

- 实验文档分类与完整索引：[`review/README.md`](review/README.md)
- 实验分析与记录固定模板：[`docs/EXPERIMENT_RECORD_TEMPLATE.md`](docs/EXPERIMENT_RECORD_TEMPLATE.md)
- Git 协作与实验记录指南：[`docs/GIT_PROJECT_GUIDE_from_local.md`](docs/GIT_PROJECT_GUIDE_from_local.md)
- Scan-aware 数据现实与当前主张边界：[`docs/SCAN_AWARE_DATA_REALITY_AND_CLAIM_BOUNDARY.md`](docs/SCAN_AWARE_DATA_REALITY_AND_CLAIM_BOUNDARY.md)
- Support-aware paired protocol 执行计划：[`review/plans/19_support_aware_paired_protocol_execution_plan_2026-08-03.md`](review/plans/19_support_aware_paired_protocol_execution_plan_2026-08-03.md)
- APIC v3 image-only 对比实验计划：[`review/plans/20_apic_v3_image_only_comparison_plan_2026-08-03.md`](review/plans/20_apic_v3_image_only_comparison_plan_2026-08-03.md)
- APIC v3 失败机制诊断：[`review/operations/24_apic_v3_failure_diagnostics_2026-08-04.md`](review/operations/24_apic_v3_failure_diagnostics_2026-08-04.md)
- Dual-Shift 模型初步设计：[`Model/DUAL_SHIFT_MODEL_PRELIMINARY_DESIGN.md`](Model/DUAL_SHIFT_MODEL_PRELIMINARY_DESIGN.md)

新增实验文档必须先按主要用途放入 `review/plans/`、`review/records/`、
`review/analysis/` 或 `review/operations/`，并同步更新 `review/README.md`。
