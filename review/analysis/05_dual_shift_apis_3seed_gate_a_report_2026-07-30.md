# Dual-Shift APIS 3-seed Gate A 实验报告

**日期：** 2026-07-30  
**文档序号：** `05`  
**继承：** `01` 决策、`02` 安排、`03` 状态  
**证据目录：** `outputs/apis_3seed/`、`outputs/postfix/`  

---

## 1. 结论（Gate A）

**Gate A = No-Go**（不满足「两方向 × 全部 3 seeds 均过 APIS gate」）。

因此：**不扩展 MCI / 三分类；不恢复 Joint；不调 APIS/CDT 超参。**

论文口径收缩为：protocol-aware / 方向依赖的 APIS 探索 + 机制边界（配对 1.5T/3T），而非双向稳定模块通关。

---

## 2. 实验范围

| 项 | 设定 |
|---|---|
| 任务 | CN vs AD |
| 方向 | ADNI→NACC、NACC→ADNI |
| Seeds | 42（postfix）、43、44 |
| 变体 | `ce_only`、`mixstyle`、`apis_only` |
| 配置 | 冻结 `journal_dual_shift_postfix.yaml` |
| 禁止 | CDT/Joint 多 seed；target-driven 调参 |

---

## 3. APIS gate（相对 `ce_only`）

判定：AUC 非劣（下降 ≤0.01）∧ F1 提升 ∧ 无 SEN/SPE 塌缩。

| 方向 | seed42 | seed43 | seed44 |
|---|---|---|---|
| ADNI→NACC | **fail**（F1↓） | pass | pass |
| NACC→ADNI | pass | **fail**（AUC 非劣失败 + F1↓） | pass |

逐 seed 细节见 `outputs/apis_3seed/gate_report_3seed.json`；全表见 `metrics_table_3seed.csv`。

---

## 4. 并行机制证据（摘要）

### 4.1 CDT source 审计（seed42）

`outputs/apis_3seed/cdt_source_audit_seed42.json`：两方向 **非 pathological**，不触发一次稳定化修复。CDT 仍不作性能扩展。

### 4.2 同受试者 1.5T/3T 配对（≤90d，可评估 21/34）

`outputs/apis_3seed/paired_field_strength_seed42.json`：

- `apis_only` mean\|Δp\| **低于** CE 与 MixStyle（约 −0.0018 / −0.0097）
- 三变体 flip rate = 0  
- **未见系统性反证**；但不单独推翻 Gate A No-Go

---

## 5. 产物清单

```text
review/
  plans/01_dual_shift_next_step_decision_2026-07-29.md
  plans/02_dual_shift_experiment_schedule_2026-07-29.md
  records/03_dual_shift_apis_3seed_status_2026-07-29.md
  analysis/05_dual_shift_apis_3seed_gate_a_report_2026-07-30.md   ← 本报告
  analysis/dual_shift_postfix_gate_results_2026-07-25.md
outputs/
  apis_3seed/{gate_report_3seed.json,metrics_table_3seed.csv,
              cdt_source_audit_seed42.json,paired_field_strength_seed42.json}
  postfix/{gate_report.json,postfix_metrics_table.csv}
```

未上传：训练 checkpoint / 原始队列日志（体积大；可按需另行归档）。
