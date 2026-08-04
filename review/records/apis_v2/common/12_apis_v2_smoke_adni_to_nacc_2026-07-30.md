# APIS-V2-SMOKE-20260730：本地 ADNI→NACC seed42 短程 smoke 记录

## 基本信息

- 日期：2026-07-30
- 负责人：本地 Windows（`merge/apis-v2-local`）
- 状态：已完成
- Git commit：工作区含 W1 接线后未单独提交；运行时基于合并基线 `1c50fd6` + 本地 W1/W2 改动
- 关联文档：`review/plans/11_apis_v2_local_experiment_schedule_2026-07-30.md`、`review/plans/10_apis_v2_claim_validation_plan_2026-07-30.md`
- 关联数据文件：`outputs/journal/dual_shift_apis_v2/smoke/`

## 1. 实验指导与依据

- 研究问题：W1 就绪后，APIS v2 训练回路能否在真实 ADNI→NACC 数据上无 NaN 跑通，并写出 checkpoint / metrics / 干预审计。
- 假设与基线：短程 smoke 仅验证端到端通路；**不**作为确认性性能证据。
- 实施依据：计划 `11` 的 W2 / R6。
- 实验范围：CN vs AD；ADNI→NACC；seed 42；`ce_only` / `mixstyle` / `apis_only`；6 epochs（warm_clean=1, warm_apis=2）。
- 判定标准：训练完成；三变体均有 `journal_metrics.json` 与 `best_checkpoint.pt`；`apis_only` 出现非零 `apis_l2` 与 `valid_frac>0`；无 Traceback / NaN。

## 2. 可复现记录

- 配置文件：`config/journal_dual_shift_apis_v2_smoke.yaml`
- 数据与划分：`F:/ADNI/scan_manifests/{ADNI,NACC}_scan_manifest.csv`（指纹见 `env_fingerprint.json`）
- 随机种子：42
- 环境与硬件：Windows；RTX 3090；`C:\Anaconda3\envs\pytorch`（PyTorch 1.13.0+cu117）
- 启动命令：

```bat
set PYTHONPATH=%CD%
set PYTHONUNBUFFERED=1
C:\Anaconda3\envs\pytorch\python.exe run_v2.py --exp journal --direction ADNI_to_NACC ^
  --variants ce_only mixstyle apis_only --seed 42 --device cuda ^
  --output-dir outputs/journal/dual_shift_apis_v2/smoke/seed42/adni_to_nacc ^
  --config_path config/journal_dual_shift_apis_v2_smoke.yaml --force-variants
```

- 工作区状态：dirty（含 W1 代码与依赖修复）
- 产物位置：`outputs/journal/dual_shift_apis_v2/smoke/seed42/adni_to_nacc/`

## 3. 分析与结论

### 3.1 结果

| 方法 | target AUC | macro-F1 | Brier | ECE | 备注 |
|---|---:|---:|---:|---:|---|
| ce_only | 0.8963 | 0.7634 | 0.1288 | 0.0573 | 短程 |
| mixstyle | 0.9130 | 0.7752 | 0.1180 | 0.0432 | 短程 |
| apis_only | 0.9202 | 0.8146 | 0.0890 | 0.0227 | 短程；审计见下 |

`apis_only` 相位审计（训练日志）：

- epoch1 `clean_warmup`：`apis_l2=0`, `valid_frac=0`
- epoch2–3 `apis_warmup`：epoch3 起 `apis_l2≈0.0036`, `valid_frac=1.0`
- epoch4–6 `joint`：`apis_l2≈0.002`, `valid_frac=1.0`

### 3.2 分析

- 相对基线：短程数值对 APIS 有利，但 epoch 远少于 postfix（50），**不得**写入确认性 Gate / 论文主表。
- 通路验证：W1 的 penalty 与 per-sample mask 已在真实数据路径生效（`valid_frac=1.0` 且系数进入日志）。
- 异常与局限：依赖环境曾缺 `nibabel`/`torchmetrics`，已装入 `pytorch` env；`Model.comparison` 改为 opportunistic 以避免无关几何包阻断 journal。
- 结果结论：**R6 smoke Go**；可进入 W3（反向 smoke 或加长 E1），仍禁止 MCI / Joint / target 调参。

## 4. 建议下一步实验指导

- 建议动作：启动 W3——优先 NACC→ADNI seed42 短程 smoke，或直接用冻结 postfix 预算开 E1（双向 × 3 seeds）。
- 建议依据：本记录证明端到端可跑；确认性仍需完整 epoch 与多 seed。
- 固定条件：source-only checkpoint；subject-mean；不混用 legacy Windows/Linux 表。
- 进入条件：保留本 smoke 产物与 `env_fingerprint.json`。
- 禁止事项：把 6-epoch smoke 写成 Gate A / E1 通关。
