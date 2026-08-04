# APIS-V2-SMOKE-BIDIR-20260730：双向 seed42 短程 smoke 汇总

## 基本信息

- 日期：2026-07-30
- 负责人：本地 Windows（`merge/apis-v2-local`）
- 状态：已完成
- Git commit：工作区含 W1/W2/W3 改动（未单独提交）
- 关联文档：`review/plans/11_apis_v2_local_experiment_schedule_2026-07-30.md`、`review/records/apis_v2/common/12_apis_v2_smoke_adni_to_nacc_2026-07-30.md`
- 关联数据文件：`outputs/journal/dual_shift_apis_v2/smoke/seed42/`、`outputs/journal/dual_shift_apis_v2/paired_holdout/`

## 1. 实验指导与依据

- 研究问题：APIS v2 短程 smoke 在 NACC→ADNI 方向是否同样端到端可跑，并与既有 ADNI→NACC smoke 形成双向通路证据。
- 假设与基线：仅验证通路与审计；短程数值不作确认性比较。
- 实施依据：计划 `11` 的 W3（先补反向 smoke）与 W4（配对 held-out 准备）。
- 实验范围：CN vs AD；双向；seed 42；`ce_only`/`mixstyle`/`apis_only`；6 epochs。
- 判定标准：反向三变体完成；`apis_only` 有非零 `apis_l2` 与 `valid_frac=1`；配对 ≤30d / ≤7d 名单落盘。

## 2. 可复现记录

- 配置：`config/journal_dual_shift_apis_v2_smoke.yaml`
- 启动（反向）：

```bat
C:\Anaconda3\envs\pytorch\python.exe run_v2.py --exp journal --direction NACC_to_ADNI ^
  --variants ce_only mixstyle apis_only --seed 42 --device cuda ^
  --output-dir outputs/journal/dual_shift_apis_v2/smoke/seed42/nacc_to_adni ^
  --config_path config/journal_dual_shift_apis_v2_smoke.yaml --force-variants
```

- 配对 held-out：复用 `F:/ADNI/scan_manifests/paired_field_strength_manifest.csv`，写出  
  `outputs/journal/dual_shift_apis_v2/paired_holdout/{paired_holdout_subjects.json,paired_le_30d.csv,paired_le_7d.csv}`

## 3. 分析与结论

### 3.1 结果（target subject-mean；短程）

| 方向 | 变体 | AUC | macro-F1 | SEN | SPE | 备注 |
|---|---|---:|---:|---:|---:|---|
| ADNI→NACC | ce_only | 0.896 | 0.763 | 0.796 | 0.835 | 见 `12` |
| ADNI→NACC | mixstyle | 0.913 | 0.775 | 0.812 | 0.842 | |
| ADNI→NACC | apis_only | 0.920 | 0.815 | 0.737 | 0.915 | `valid_frac=1` |
| NACC→ADNI | ce_only | 0.860 | 0.296 | 0.000 | 1.000 | **短程塌缩** |
| NACC→ADNI | mixstyle | 0.879 | 0.791 | 0.792 | 0.796 | |
| NACC→ADNI | apis_only | 0.901 | 0.803 | 0.899 | 0.694 | `apis_l2≈0.023`，`valid_frac=1` |

反向 `apis_only` 相位：`clean_warmup → apis_warmup → joint`；epoch3 起干预生效。

配对 held-out：全部 73 对；≤30d **33**；≤7d **25**（与设计文档一致）。

### 3.2 分析

- 通路：双向 smoke **均完成**；APIS v2 审计在两侧均出现有效干预。
- 异常：反向 `ce_only` 在 6 epoch 下出现 SEN=0 塌缩，属短程/选模不稳定信号，**不得**外推到完整 postfix 预算；完整 E1 必须用 50 epoch 与 collapse guard 重新判定。
- 结论：双向短程 smoke **Go**；可进入完整 epoch 的 E1（建议先排除/冻结 ≤30d 配对受试者），或先做 W5 shuffle 负对照最小子集。

## 4. 建议下一步实验指导

- 建议动作：启动完整预算 E1（`journal_dual_shift_postfix.yaml` 的 epochs/warmups，输出到 `outputs/journal/dual_shift_apis_v2/e1/`），训练前排除 `paired_holdout_subjects.json` 中 ≤30d 受试者。
- 固定条件：source-only checkpoint；subject-mean；不与 legacy 3-seed 混表。
- 禁止事项：把本短程表写成确认性通关；不为短程 CE 塌缩做 target 调参。
