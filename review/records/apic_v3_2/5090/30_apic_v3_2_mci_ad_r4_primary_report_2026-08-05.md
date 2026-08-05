# APIC v3_2 MCI vs AD 原型主矩阵报告（5090 · 2026-08-05）

## 1. 执行状态

| 项 | 状态 |
| --- | --- |
| 训练进程 | 已结束（GPU 空闲） |
| launcher | 4/4 `[apic-v3_2] OK` |
| metrics | 12/12 `journal_metrics.json` |
| 任务 | MCI vs AD（不变） |
| 协议 | revision 4 · image-only X |
| 主张 | **prototype / 非正式 Gate** |

## 2. 主终点（target subject-level BA）

| seed | direction | ce_x | mixstyle_x | apic_v3_2_x | Δ(v3_2−ce) | Δ(v3_2−mix) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | ADNI_to_NACC | 0.672500 | 0.699359 | 0.659830 | -0.012670 | -0.039529 |
| 42 | NACC_to_ADNI | 0.675651 | 0.639907 | 0.679526 | +0.003875 | +0.039618 |
| 43 | ADNI_to_NACC | 0.657679 | 0.679472 | 0.670690 | +0.013011 | -0.008782 |
| 43 | NACC_to_ADNI | 0.659398 | 0.664488 | 0.668768 | +0.009370 | +0.004280 |
| mean | — | 0.666307 | 0.670807 | 0.669703 | +0.003396 | -0.001103 |

**均值**：ce_x **0.6663** · mixstyle_x **0.6708** · apic_v3_2_x **0.6697**

相对基线胜场：vs ce **3/4** · vs mix **2/4** · 双胜 **2/4**。

## 3. 机制侧快照（非 layer-2 全量诊断）

- 正式推理路径均为 `clean`（`apic_v3_audit.inference_path`）。
- memory：`valid_slots=4`；分配未再出现 v3 那种 >99% 单槽独占（见结果表 D），但仍有不均衡。
- 训练日志中 `apis_l2` 多记为 0（v3_2 审计字段语义与 v3 不同）；完整反事实 RMS/flip 需另跑 ops/24 风格 layer-2（`apic_v3_2_x`）。

## 4. 解读（受主张边界约束）

1. 本跑验证了 **5090 上 MCI 全矩阵可跑通**（含 `_apply`→`_apply_feature_shift` 修复后）。
2. 性能上，`apic_v3_2_x` **未能稳定超过 MixStyle**；对 CE 仅部分 cell 更好。
3. 因 `formal_run_allowed=false`，结果只用于工程/机制反馈，**不能**替代计划 26 的 Gate M0→E3 正式路径，也不得扩 seed / 启 X+D。

## 5. 产物索引

- 主表：`29_apic_v3_2_mci_ad_r4_primary_2026-08-05.md`
- CSV：`metrics_table_apic_v3_2_mci_ad_r4_primary_2026-08-05.csv`
- 场强：`metrics_table_apic_v3_2_mci_ad_r4_primary_by_field_strength_2026-08-05.csv`
- 日志：`apic_v3_2_mci_ad_r4_primary_logs/`
- 原始输出：`outputs/journal/apic_v3_2_screening_mci_ad/r4/`

## 6. 建议下一步

- 按计划 26 补齐 E0/E1/E2 Gate M0 机制证伪后再谈正式 E3。
- 对本跑 1–2 个代表 cell 做 `export_apic_v3_checkpoint_diagnostics.py --variant apic_v3_2_x`，核对干预是否仍近恒等。
