# APIC v3_2 M0 尺度修复验收归档（5090 · 2026-08-06）

**结论：NO-GO。** `formal_run_allowed=false`；非正式 E3；不扩 seed、不启 X+D。

## 本目录内容

| 路径 | 说明 |
|---|---|
| `GO_NO_GO.md` | 验收裁决与 M0 scorecard |
| `m0_metrics_summary.json` | 机读 scorecard（含 A2N / N2A epoch7 / N2A epoch50） |
| `ARTIFACT_PATHS.json` | 5090 上大文件（`.pt` / 完整预测）绝对相对路径索引 |
| `configs/` | 远程验收配置副本 |
| `metrics/` | E2b 训练 summary / slim journal metrics |
| `layer1/` | 训练历史汇总 |
| `layer2/{a2n,n2a_epoch7,n2a_epoch50}/` | 诊断 summary + sample CSV |
| `logs/` | layer1/2 驱动与 stdout 摘录 |
| `unit_tests/` | `pytest tests/test_apis_v3_2.py` 结果 |

## 关联

- 验收标准：`review/analysis/36_apic_v3_2_m0_scale_repair_acceptance_2026-08-05.md`
- 实验报告：`review/analysis/37_apic_v3_2_m0_scale_repair_experiment_report_2026-08-06.md`
- 前序失败诊断：`../apic_v3_2_mci_failure_diagnostics_2026-08-05/`
