# APIC v3_2 CN_vs_AD revision-4 prototype 运行报告（3090）

## 结论

CN_vs_AD prototype 矩阵 **12/12 完成**（含 `_apply`→`_apply_feature_shift` 修复后的候选补跑）。相对基线：**仅 1/4** seed×direction 双胜（seed42 ADNI→NACC）；两方向两 seed 平均 Δ 均未对 CE+MixStyle 同时为正。本轮仍是 **prototype / 非正式**，**不能**主张 Gate S1 或 revision-4 通过。

## 1. 进程与修复

| 项 | 状态 |
|---|---|
| 首轮 launcher | 基线 8/8 OK；候选 0/4 在 `.to(cuda)` 崩溃 |
| 根因 | `FixedStyleBankAPICV32._apply` 覆盖 `Module._apply` |
| 修复 | upstream `0a46d98`：`_apply_feature_shift` + `model.to(device)` 回归 |
| 补跑 | 4× `apic_v3_2_x` 串行完成，`ALL_OK`，exit 0 |

## 2. 主终点（target · subject-mean BA）

| seed | direction | ce_x | mixstyle_x | apic_v3_2_x | Δce | Δmix | win_both |
|---:|---|---:|---:|---:|---:|---:|---|
| 42 | A→N | 0.7804 | 0.7937 | **0.8162** | +0.0358 | +0.0224 | Y |
| 43 | A→N | 0.8427 | 0.7922 | 0.7974 | −0.0453 | +0.0052 |  |
| 42 | N→A | 0.8349 | 0.7845 | 0.8254 | −0.0095 | +0.0409 |  |
| 43 | N→A | 0.8131 | 0.7832 | 0.7353 | −0.0778 | −0.0478 |  |

机读与详表：`review/records/apic_v3_2/3090/31_apic_v3_2_cn_ad_r4_prototype_metrics_2026-08-05.md` 与同目录 CSV。

## 3. 判读

1. **工程**：候选路径已可完整训练与导出，首轮阻塞已关闭。
2. **性能（描述性）**：收益不稳定；A→N 有局部双胜，N→A seed43 明显落后 CE/MixStyle。
3. **Gate S1（CN 份额对照）**：不满足“双胜数量 / 双向均值 Δ 同时为正 / 最差退化≤0.02”。
4. **协议**：`formal_run_allowed: false`；缺少 Gate M0 机制审计产物，故不得升级为正式 E3 结论。
5. **与 5090 MCI**：见 `review/records/apic_v3_2/5090/29_…` / `30_…`；两侧均为 prototype，合并 Gate 前须各自机制审计。

## 4. 建议下一步

1. 按 plan 26 补 E1/E2a/E2b 与 Gate M0，再决定是否正式重跑 E3。
2. 机制验证前不扩 seed、不启 X+D、不把本表登记为正式 revision-4 结果。

## 5. 关联

- 计划：`review/plans/26_apic_v3_2_model_and_experiment_plan_2026-08-04.md`
- 记录表：`review/records/apic_v3_2/3090/31_apic_v3_2_cn_ad_r4_prototype_metrics_2026-08-05.md`
- 代码修复：`0a46d98`（`Model/dual_shift/apis_v3_2.py`）
- 启动基线：`bedb638`
