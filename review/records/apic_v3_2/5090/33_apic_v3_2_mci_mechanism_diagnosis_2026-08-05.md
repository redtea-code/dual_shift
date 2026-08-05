# APIC-V3_2-MCI-MECHDIAG-20260805：成功/失败单元机制诊断

## 基本信息

- 日期：2026-08-05
- 负责人：本机 5090
- 状态：已完成；**建议不进入正式 E3**
- Git commit（诊断时基线）：`9acb30f`；诊断脚本含本地修补（见 §2）
- 关联文档：`review/plans/26_apic_v3_2_model_and_experiment_plan_2026-08-04.md`、`review/records/apic_v3_2/5090/29_apic_v3_2_mci_ad_r4_primary_2026-08-05.md`、`review/analysis/35_apic_v3_2_mci_mechanism_diagnosis_report_2026-08-05.md`
- 关联数据：`review/records/apic_v3_2/5090/apic_v3_2_mci_failure_diagnostics_2026-08-05/`、`outputs/journal/apic_v3_2_failure_diagnostics/`
- 对照：CN 同构结论见 `review/analysis/34_apic_v3_2_cn_mechanism_diagnosis_report_2026-08-05.md`（3090 分支）

## 1. 实验指导与依据

- 研究问题：在 MCI 唯一双胜单元与最差单元上，APIC v3_2 是否产生可测的 clean/shift 干预（RMS、flip、JS），以及 style bank 是否健康（slot / NMI）。
- 假设与基线：若机制近恒等或支持率为 0，则性能表不能支持进入正式 E3。
- 实施依据：plan 26 Gate M0；ops 24 诊断流程适配 `apic_v3_2_x`。
- 实验范围：

| 角色 | seed | direction | target BA (apic) | vs CE | vs Mix |
|---|---:|---|---:|---:|---:|
| 成功（唯一双胜） | 42 | NACC→ADNI | 0.6795 | +0.0039 | +0.0396 |
| 失败（最差 −mix） | 42 | ADNI→NACC | 0.6598 | −0.0127 | −0.0395 |

- 判定标准：对照 Gate M0 中与 checkpoint 相关的条目（有效干预、RMS band、support fraction、slot 健康）；本轮不调参。

## 2. 可复现记录

- 配置：`journal_dual_shift_apic_v3_2_screen_mci_ad_remote.yaml`
- Layer-1：

```text
python experiments/summarize_apic_v3_diagnostics.py \
  --roots outputs/journal/apic_v3_2_screening_mci_ad/r4 \
  --output-dir outputs/journal/apic_v3_2_failure_diagnostics/history
```

- Layer-2（先 smoke `--max-samples 32`，再全量；GPU 5）：

```text
python experiments/export_apic_v3_checkpoint_diagnostics.py \
  --config journal_dual_shift_apic_v3_2_screen_mci_ad_remote.yaml \
  --job-dir outputs/journal/apic_v3_2_screening_mci_ad/r4/seed42/<slug> \
  --variant apic_v3_2_x --device cuda --batch-size 2
```

- 诊断脚本修补（相对 `9acb30f`）：
  1. 加载 `acquisition_encoder_extra` / bank / CDT 后再 `load_state_dict`；
  2. 从 `style_valid` 恢复 `_finalized`（Python 标志，不在 state_dict）；
  3. `prepare_style_condition(..., sample_ids=subject_id)`。
- 参考预测一致性：两单元 source_val/test/target 均 `matches_within_1e_5=true`。
- Layer-2 样本数：失败单元 1322；成功单元 1322（四 split 合计）。

## 3. 分析与结果

### 3.1 第一层：训练轨迹（两单元 apic_v3_2_x）

| unit | best ep | early L2 | late L2 | late feat strength | valid_intervention (train log max) | memory_valid_slots |
|---|---:|---:|---:|---:|---:|---:|
| seed42 N→A（成功） | 46 | 0 | 0 | 0 | 0 | 4 |
| seed42 A→N（失败） | 50 | 0 | 0 | 0 | 0 | 4 |

训练全阶段 `condition_gate` / `apis_feature_strength` / `apis_coefficient_l2` / `valid_intervention_frac` 最大值均为 **0**。

### 3.2 第二层：反事实（全四 split）

共性（成功与失败单元相同）：

| 量 | 观察 |
|---|---|
| `valid_intervention` 比例 | **0.0**（1322/1322 False） |
| gate / layer1–2 RMS / JS / flip | 全为 **0**；clean BA ≡ shifted BA |
| style bank | **4/4** valid slots；counts 非单槽垄断 |
| assignment–label NMI | ≈0.03（低）；未出现 v3 式单槽塌缩 |

明细：

| unit | split | n | flip | gate mean | L1/L2 RMS | JS | clean BA | shifted BA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seed42 N→A | source_train | 389 | 0 | 0 | 0 / 0 | 0 | 1.000 | 1.000 |
| seed42 N→A | source_val | 133 | 0 | 0 | 0 / 0 | 0 | 0.716 | 0.716 |
| seed42 N→A | source_test | 132 | 0 | 0 | 0 / 0 | 0 | 0.663 | 0.663 |
| seed42 N→A | target | 668 | 0 | 0 | 0 / 0 | 0 | 0.649 | 0.649 |
| seed42 A→N | source_train | 407 | 0 | 0 | 0 / 0 | 0 | 0.987 | 0.987 |
| seed42 A→N | source_val | 109 | 0 | 0 | 0 / 0 | 0 | 0.521 | 0.521 |
| seed42 A→N | source_test | 152 | 0 | 0 | 0 / 0 | 0 | 0.613 | 0.613 |
| seed42 A→N | target | 654 | 0 | 0 | 0 / 0 | 0 | 0.650 | 0.650 |

（注：上表 clean BA 为 scan-level 诊断聚合；主表 subject-level target BA 见 §1。）

### 3.3 根因

配置 `delta_min=0.02`、`delta_max=0.50`，但 PCA 空间原型间距约为 **18–52**：

| unit | 原型间距范围 | radii | counts |
|---|---|---|---|
| seed42 A→N | 17.7–39.5 | 18.5–24.5 | 39 / 2 / 18 / 31 |
| seed42 N→A | 20.3–52.0 | 16.8–32.1 | 75 / 5 / 67 / 25 |

`prepare_style_condition` 要求替代原型距离落入 `[delta_min, delta_max]`，导致 **choices 恒为空 → target=src → separation=0 → valid=False**。真正阻断的是 **delta band 尺度错误**，不是 slot 塌缩。

因此：性能表上的“双胜/失败”均发生在 **APIC 干预从未生效** 的条件下，等价于带无效 APIC 脚手架的 clean-path 波动。与 CN 3090 机制诊断结论同构。

### 3.4 Gate M0 / E3 判定

| 问题 | 答案 |
|---|---|
| 是否近恒等？ | 是（支持掩码全关导致的恒等） |
| 是否有害大扰动？ | 否（无扰动） |
| Gate M0？ | 不通过（support / RMS） |
| 正式 E3？ | **否** |

## 4. 下一步

1. 重新定义并冻结相对位移 / delta band（与 PCA 度量一致），写入新机制配置。
2. 用 E1 合成与 E2 source-only 证明 supported 比例与 RMS band。
3. 通过 E2b 四单元 Gate M0 后，再考虑正式 revision-4 E3（CE/MixStyle 需同协议重跑）。

在此之前，保持 `formal_run_allowed: false`，不扩 seed、不启 X+D。
