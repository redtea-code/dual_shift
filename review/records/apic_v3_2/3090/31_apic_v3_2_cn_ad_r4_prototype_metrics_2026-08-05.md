# APIC-V3_2-CN-R4-PROTO-20260805：3090 CN_vs_AD prototype 矩阵（含修复补跑）

## 基本信息

- 日期：2026-08-05
- 负责人：本机 3090 节点
- 状态：**已完成（prototype 12/12）**；正式 revision-4 / Gate S1 仍未获准
- Git commit：启动基线 `bedb638`；`_apply` 修复已在 `0a46d98`（`_apply_feature_shift`）；本推送登记 3090 CN 12/12 记录
- 关联文档：`review/plans/26_apic_v3_2_model_and_experiment_plan_2026-08-04.md`、`review/analysis/32_apic_v3_2_cn_ad_r4_prototype_run_report_2026-08-05.md`
- 关联数据文件：`outputs/journal/apic_v3_2_screening_cn_ad/r4/`、`review/records/apic_v3_2/3090/apic_v3_2_cn_ad_r4_prototype_metrics_main_table.csv`

## 1. 实验指导与依据

- 研究问题：修复 `.to(device)` 崩溃后，prototype `apic_v3_2_x` 在 CN_vs_AD 双向 × seeds {42,43} 上相对 `ce_x`/`mixstyle_x` 的 target BA 表现如何。
- 假设与基线：非主张开发运行（`--allow-prototype-run` / 直接 `run_v2` 仅补候选）；正式 Gate 仍关闭。
- 实施依据：首轮 8/8 基线成功、4/4 候选因 `_apply` 命名冲突失败；修复后补跑候选。
- 实验范围：CN_vs_AD × 2 directions × seeds {42,43} × 3 variants = **12/12**。
- 判定标准：按 plan 26 描述 Gate S1 口径做 **对照判读**，但不把本轮登记为正式通过/失败决议（`formal_run_allowed: false`）。

## 2. 可复现记录

- 配置文件：`journal_dual_shift_apic_v3_2_screen_cn_ad.yaml`
- 数据与划分：配置内 ADNI/NACC 路径；`split_seed=42`；paired holdout 排除
- 随机种子：42, 43
- 环境：Windows · RTX 3090 · Python 3.8.19 · PyTorch 1.13.0+cu117
- 首轮启动：

```text
python experiments/run_apic_v3_2_screening.py \
  --configs journal_dual_shift_apic_v3_2_screen_cn_ad.yaml \
  --seeds 42,43 --gpu-ids 0 --max-workers 1 --allow-prototype-run
```

- 候选补跑（修复后，仅 `apic_v3_2_x`）：

```text
python run_v2.py --exp journal \
  --config_path journal_dual_shift_apic_v3_2_screen_cn_ad.yaml \
  --device cuda --direction <DIR> --seed <SEED> \
  --variants apic_v3_2_x \
  --output-dir outputs/journal/apic_v3_2_screening_cn_ad/r4/seed<SEED>/<slug>
```

- 修复要点：`0a46d98` 将 `FixedStyleBankAPICV32._apply` 重命名为 `_apply_feature_shift`（避免覆盖 `nn.Module._apply`）；本机补跑在该修复语义下完成
- 产物：`outputs/journal/apic_v3_2_screening_cn_ad/r4/`
- 补跑日志：`…/logs/*_apic_v3_2_x_rerun.log`；汇总 `…/logs/rerun_apic_v3_2_x_stdout.txt`（含 `ALL_OK`）

## 3. 分析与结果

### 3.1 完成率

| 单元 | ce_x | mixstyle_x | apic_v3_2_x |
|---|---|---|---|
| seed42 adni_to_nacc | OK | OK | OK（补跑） |
| seed42 nacc_to_adni | OK | OK | OK（补跑） |
| seed43 adni_to_nacc | OK | OK | OK（补跑） |
| seed43 nacc_to_adni | OK | OK | OK（补跑） |

### 3.2 Target · subject-mean 速览（balanced_accuracy）

| seed | direction | ce_x | mixstyle_x | apic_v3_2_x | Δ vs ce | Δ vs mix | win_both |
|---:|---|---:|---:|---:|---:|---:|---|
| 42 | adni_to_nacc | 0.7804 | 0.7937 | 0.8162 | +0.0358 | +0.0224 | Y |
| 43 | adni_to_nacc | 0.8427 | 0.7922 | 0.7974 | −0.0453 | +0.0052 |  |
| 42 | nacc_to_adni | 0.8349 | 0.7845 | 0.8254 | −0.0095 | +0.0409 |  |
| 43 | nacc_to_adni | 0.8131 | 0.7832 | 0.7353 | −0.0778 | −0.0478 |  |
| **mean** | adni_to_nacc | 0.8115 | 0.7930 | 0.8068 | −0.0048 | +0.0138 |  |
| **mean** | nacc_to_adni | 0.8240 | 0.7838 | 0.7804 | −0.0437 | −0.0035 |  |

- seed×direction 同时胜两基线：**1/4**
- direction 两 seed 平均对两基线均为正：**0/2**（仅 CN 份额）

### 3.3 `adni_to_nacc` 全指标

| seed | variant | BA | AUC | macro-F1 | SEN | SPE | ACC | Brier | ECE | n | BA_CI_lo | BA_CI_hi | best_ep |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | ce_x | 0.7804 | 0.9065 | 0.7868 | 0.6344 | 0.9264 | 0.8698 | 0.1010 | 0.0713 | 960 | 0.7478 | 0.8120 | 15 |
| 42 | mixstyle_x | 0.7937 | 0.9103 | 0.8004 | 0.6559 | 0.9315 | 0.8781 | 0.1006 | 0.0817 | 960 | 0.7635 | 0.8272 | 31 |
| 42 | apic_v3_2_x | 0.8162 | 0.8971 | 0.7906 | 0.7473 | 0.8850 | 0.8583 | 0.1221 | 0.1028 | 960 | 0.7816 | 0.8470 | 35 |
| 43 | ce_x | 0.8427 | 0.9196 | 0.7995 | 0.8172 | 0.8682 | 0.8583 | 0.1152 | 0.0934 | 960 | 0.8152 | 0.8751 | 38 |
| 43 | mixstyle_x | 0.7922 | 0.9077 | 0.7852 | 0.6774 | 0.9070 | 0.8625 | 0.1088 | 0.0837 | 960 | 0.7599 | 0.8268 | 26 |
| 43 | apic_v3_2_x | 0.7974 | 0.8995 | 0.7950 | 0.6774 | 0.9173 | 0.8708 | 0.1002 | 0.0656 | 960 | 0.7656 | 0.8319 | 22 |

### 3.4 `nacc_to_adni` 全指标

| seed | variant | BA | AUC | macro-F1 | SEN | SPE | ACC | Brier | ECE | n | BA_CI_lo | BA_CI_hi | best_ep |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | ce_x | 0.8349 | 0.9209 | 0.8323 | 0.8448 | 0.8250 | 0.8367 | 0.1225 | 0.0826 | 196 | 0.7820 | 0.8834 | 49 |
| 42 | mixstyle_x | 0.7845 | 0.8577 | 0.7839 | 0.8190 | 0.7500 | 0.7908 | 0.1663 | 0.1197 | 196 | 0.7269 | 0.8412 | 36 |
| 42 | apic_v3_2_x | 0.8254 | 0.9138 | 0.8144 | 0.7759 | 0.8750 | 0.8163 | 0.1412 | 0.0955 | 196 | 0.7643 | 0.8725 | 32 |
| 43 | ce_x | 0.8131 | 0.8876 | 0.8204 | 0.9138 | 0.7125 | 0.8316 | 0.1350 | 0.0854 | 196 | 0.7584 | 0.8636 | 31 |
| 43 | mixstyle_x | 0.7832 | 0.8898 | 0.7731 | 0.7414 | 0.8250 | 0.7755 | 0.1679 | 0.1393 | 196 | 0.7256 | 0.8321 | 29 |
| 43 | apic_v3_2_x | 0.7353 | 0.8564 | 0.7142 | 0.6207 | 0.8500 | 0.7143 | 0.2069 | 0.1611 | 196 | 0.6781 | 0.7839 | 29 |

### 3.5 Gate S1 对照（仅 CN；非正式）

按 plan 26 口径在本机 CN 份额上：

| 条款 | CN 观察 | 备注 |
|---|---|---|
| 两 seed 平均 Δ 对两基线均为正的 direction ≥3/4（全任务） | CN 为 **0/2** | 完整 Gate 还需 MCI |
| seed×direction 双胜 ≥5/8 | CN **1/4** | 仅 seed42 A→N |
| 最差 direction 两 seed 平均退化 ≤0.02 | N→A mean Δ vs CE ≈ **−0.044** | 超出 |
| SEN/SPE/class recall ≥0.15 | 表内 SEN/SPE 均满足 | 未单独拆 class recall |

**非正式结论：CN 份额不足以支持 Gate S1；且本轮仍是 prototype。**

## 4. 建议下一步实验指导

- 建议动作：按 plan 26 回到 E1/E2 Gate M0，而不是据此扩 seed。
- 建议依据：性能矩阵已可跑通，但双胜与均值 Δ 未达 S1；机制健康尚未按 M0 审计。
- 固定条件：image-only、CN/MCI 划分、seeds、基线、`formal_run_allowed=false` 直至复审关闭。
- 进入条件：机制 Gate M0（含 RMS band / support / BN）通过后再谈正式 E3。
- 禁止事项：不把本表登记为正式 revision-4 结果；不扩 44–46；不启 X+D；不按 target 调门控。
