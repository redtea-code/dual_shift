# APIC-V3_2-CN-MECHDIAG-20260805：成功/失败单元机制诊断

## 基本信息

- 日期：2026-08-05
- 负责人：本机 3090
- 状态：已完成；**建议不进入正式 E3**
- Git commit（诊断时）：`9acb30f`（记录基线）；诊断脚本含本地修补（见 §2）
- 关联文档：`review/plans/26_apic_v3_2_model_and_experiment_plan_2026-08-04.md`、`review/records/apic_v3_2/3090/31_apic_v3_2_cn_ad_r4_prototype_metrics_2026-08-05.md`、`review/analysis/34_apic_v3_2_cn_mechanism_diagnosis_report_2026-08-05.md`
- 关联数据：`review/records/apic_v3_2/3090/apic_v3_2_cn_failure_diagnostics_2026-08-05/`、`outputs/journal/apic_v3_2_failure_diagnostics/`

## 1. 实验指导与依据

- 研究问题：在 CN 唯一双胜单元与最差单元上，APIC v3_2 是否产生可测的 clean/shift 干预（RMS、flip、JS），以及 style bank 是否健康（slot / NMI）。
- 假设与基线：若机制近恒等或支持率为 0，则性能表不能支持进入正式 E3。
- 实施依据：plan 26 Gate M0；ops 24 诊断流程适配 `apic_v3_2_x`。
- 实验范围：

| 角色 | seed | direction | target BA (apic) | vs CE | vs Mix |
|---|---:|---|---:|---:|---:|
| 成功（唯一双胜） | 42 | adni_to_nacc | 0.8162 | +0.0358 | +0.0224 |
| 失败（最差） | 43 | nacc_to_adni | 0.7353 | −0.0778 | −0.0478 |

- 判定标准：对照 Gate M0 中与 checkpoint 相关的条目（有效干预、RMS band、support fraction、slot 健康）；本轮不调参。

## 2. 可复现记录

- 配置：`journal_dual_shift_apic_v3_2_screen_cn_ad.yaml`
- Layer-1：

```text
python experiments/summarize_apic_v3_diagnostics.py \
  --roots outputs/journal/apic_v3_2_screening_cn_ad/r4 \
  --output-dir outputs/journal/apic_v3_2_failure_diagnostics/history
```

- Layer-2（先 smoke `--max-samples 32`，再全量）：

```text
python experiments/export_apic_v3_checkpoint_diagnostics.py \
  --config journal_dual_shift_apic_v3_2_screen_cn_ad.yaml \
  --job-dir outputs/journal/apic_v3_2_screening_cn_ad/r4/seed<SEED>/<slug> \
  --variant apic_v3_2_x --device cuda --batch-size 2 \
  --output-dir outputs/journal/apic_v3_2_failure_diagnostics/checkpoint/seed<SEED>_<slug>
```

- 诊断脚本本地修补（相对 `9acb30f`）：
  1. 加载 `acquisition_encoder_extra` / bank 后恢复 `_finalized`；
  2. `prepare_style_condition(..., sample_ids=subject_id)`。
- 参考预测一致性：两单元 source_val/test/target 均 `matches_within_1e_5=true`。

## 3. 分析与结果

### 3.1 第一层：训练轨迹（两单元 apic_v3_2_x）

| unit | best ep | early L2 | late L2 | late feat strength | valid_intervention (train log max) | memory_valid_slots |
|---|---:|---:|---:|---:|---:|---:|
| seed42 A→N | 35 | 0 | 0 | 0 | 0 | 4 |
| seed43 N→A | 29 | 0 | 0 | 0 | 0 | 4 |

训练全阶段 `condition_gate` / `apis_feature_strength` / `apis_coefficient_l2` / `valid_intervention_frac` 最大值均为 **0**。

### 3.2 第二层：反事实（全四 split）

共性（成功与失败单元相同）：

| 量 | 观察 |
|---|---|
| `valid_intervention` 比例 | **0.0**（含 source_train） |
| gate / layer1–2 RMS / JS / flip | 全为 **0**；clean BA ≡ shifted BA |
| style bank | **4/4** valid slots；counts 非单槽垄断 |
| assignment–label NMI | 约 0.08–0.11（低）；未出现 v3 式单槽塌缩 |

明细：

| unit | split | n | flip | gate mean | L1/L2 RMS p50 | JS p50 | clean BA | shifted BA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seed42 A→N | source_train | 334 | 0 | 0 | 0 / 0 | 0 | 1.000 | 1.000 |
| seed42 A→N | source_val | 106 | 0 | 0 | 0 / 0 | 0 | 0.809 | 0.809 |
| seed42 A→N | source_test | 107 | 0 | 0 | 0 / 0 | 0 | 0.832 | 0.832 |
| seed42 A→N | target | 1181 | 0 | 0 | 0 / 0 | 0 | 0.811 | 0.811 |
| seed43 N→A | source_train | 716 | 0 | 0 | 0 / 0 | 0 | 1.000 | 1.000 |
| seed43 N→A | source_val | 227 | 0 | 0 | 0 / 0 | 0 | 0.785 | 0.785 |
| seed43 N→A | source_test | 238 | 0 | 0 | 0 / 0 | 0 | 0.841 | 0.841 |
| seed43 N→A | target | 547 | 0 | 0 | 0 / 0 | 0 | 0.730 | 0.730 |

### 3.3 根因

配置 `delta_min=0.02`、`delta_max=0.50`，但 PCA 空间原型间距约为 **15–54**：

| unit | 原型间距范围 | radii |
|---|---|---|
| seed42 A→N | 16.4–43.5 | 15.3–21.9 |
| seed43 N→A | 15.6–54.2 | 14.5–25.2 |

`prepare_style_condition` 要求替代原型距离落入 `[delta_min, delta_max]`，导致 **choices 恒为空 → target=src → separation=0 → valid=False**。支持半径本身较大，真正阻断的是 **delta band 尺度错误**，不是 slot 塌缩。

因此：性能表上的“双胜/失败”均发生在 **APIC 干预从未生效** 的条件下，等价于带无效 APIC 脚手架的 clean CE 变体波动。

### 3.4 Gate M0 / E3 判定

| Gate M0 相关项 | 本诊断 |
|---|---|
| effective slots ≥3、非单槽垄断 | 槽位表面通过（4 slots） |
| support fraction ∈ [0.20, 0.90] | **失败（0.0）** |
| supported RMS 进入 band / hit≥0.80 | **失败（无 supported）** |
| unsupported 严格 clean | 形式上全样本 clean（因全 unsupported） |
| 机制可检验非零干预 | **失败** |

**结论：不得进入正式 E3。** 须先修正 delta / 距离尺度（或重新定义相对位移约束），冻结新机制配置并完成 E1/E2a/E2b Gate M0 后，再谈正式矩阵。

## 4. 建议下一步实验指导

- 建议动作：修复原型替代距离约束与 PCA 尺度一致性；合成/source-only 验证 supported 比例与 RMS band；再跑 Gate M0。
- 建议依据：成功与失败单元同构的零干预诊断 + 训练日志全程 gate=0。
- 固定条件：image-only、CN/MCI 划分、seeds、`formal_run_allowed=false`。
- 进入条件：E2b 四 source 单元 Gate M0 全过。
- 禁止事项：不把当前 12/12 性能表解释为 APIC 机制收益；不扩 seed；不启 X+D；不按 target 调参。
