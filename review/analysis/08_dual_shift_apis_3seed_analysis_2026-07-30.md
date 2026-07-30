# APIS-3SEED-20260730：双向三随机种子 Gate A 分析

## 基本信息

- 日期：2026-07-30
- 负责人：cyh（汇总分析）；redtea-code / Qi Zhang（远程实验与结果提交）
- 状态：已完成（正式 Gate A = No-Go）
- Git commit：分析基线 `535eedf`；Windows 结果提交 `a44194e`；Linux 结果提交 `cc45471`
- 关联文档：`review/plans/01_dual_shift_next_step_decision_2026-07-29.md`、`review/plans/02_dual_shift_experiment_schedule_2026-07-29.md`、`review/records/03_dual_shift_apis_3seed_status_2026-07-29.md`、`review/operations/04_dual_shift_remote_handoff_2026-07-29.md`、`review/analysis/05_dual_shift_apis_3seed_gate_a_report_2026-07-30.md`、`review/records/05_dual_shift_remote_claim_2026-07-29.md`、`review/records/06_dual_shift_apis_3seed_remote_results_2026-07-30.md`、`review/analysis/07_apis_3seed_windows_vs_remote_compare_2026-07-30.md`
- 关联数据文件：`outputs/apis_3seed/gate_report_3seed.json`、`outputs/apis_3seed/metrics_table_3seed.csv`、`outputs/apis_3seed/cdt_source_audit_seed42.json`、`outputs/apis_3seed/paired_field_strength_seed42.json`、`outputs/journal/dual_shift_apis_3seed/gate_report_3seed.json`、`outputs/journal/dual_shift_apis_3seed/metrics_table_3seed.csv`、`outputs/journal/dual_shift_apis_3seed/seed{43,44}/`

## 1. 实验指导与依据

- 研究问题：冻结 postfix 协议后，APIS 相对 `ce_only` 是否能在 ADNI→NACC 与 NACC→ADNI 两个方向、全部三个随机种子上稳定通过 Gate A。
- 假设与基线：`apis_only` 相对 `ce_only` 应同时满足 AUC 非劣、macro-F1 提升且敏感度/特异度不塌缩；`mixstyle` 作为辅助参照，不参与 APIS gate 的基线替换。
- 实施依据：`review/plans/01_dual_shift_next_step_decision_2026-07-29.md` 决定冻结 postfix 并先补齐 APIS 三种子；`review/plans/02_dual_shift_experiment_schedule_2026-07-29.md` 预注册实验范围、命令和 Gate A；`review/operations/04_dual_shift_remote_handoff_2026-07-29.md` 规定远程节点不得修改超参数或按目标域结果调参。
- 实验范围：CN vs AD；ADNI→NACC、NACC→ADNI；seeds 42、43、44；变体为 `ce_only`、`mixstyle`、`apis_only`。Windows 汇总包含三个 seeds；Linux 独立结果仅包含 seeds 43、44，不与 Windows 数值混合汇总。
- 判定标准：每个 seed×direction 均须满足 `APIS AUC >= CE AUC - 0.01`、`APIS macro-F1 > CE macro-F1`、无 SEN/SPE collapse。双方向全部 3 seeds 通过才判 Gate A = Go；任一失败即 No-Go。

## 2. 可复现记录

- 配置文件：`journal_dual_shift_postfix.yaml`；Linux 使用仅作路径映射的 `journal_dual_shift_postfix_remote.yaml`，该远程副本未随结果上传。
- 数据与划分版本：远程 `split_manifest.json` 显示 ADNI→NACC 为 source 791 scans/257 subjects、target 1181 scans/960 subjects；反向对调。seed 43/44 的人数一致，具体清单见各方向的 `split_manifest.json`。
- 随机种子：42（既有 postfix）、43、44；Linux 独立复现仅 43、44。
- 环境与硬件：Windows 本机环境与远程 Linux `an5bi4acenfa1-0`；远程使用 3 张 GPU 并行。提交产物未完整记录 GPU 型号、CUDA/PyTorch 版本，这是当前复现限制。
- 启动命令：`python scripts/run_journal_queue.py --stages apis_3seed --device cuda --max-workers 1 --force`；单任务可使用 `python run_v2.py --exp journal --direction <DIRECTION> --variants ce_only mixstyle apis_only --seed <SEED> --device cuda --output-dir <OUTPUT> --config_path <CONFIG>`。
- 工作区状态：分析基线的已跟踪文件为 clean；存在一个与本实验无关的历史未跟踪协作指南副本，未参与实验、分析或本次提交。原训练时工作区状态未包含在上传产物中。
- 产物位置：结构化指标与预测文件位于上述关联数据路径；训练 checkpoint 和原始队列日志未上传，因而没有可核验的外部稳定路径、大小或 SHA-256。

## 3. 分析与结果

### 3.1 结果

Windows 三种子正式 gate：

| 方向 | seed 42 | seed 43 | seed 44 | 方向汇总 |
|---|---|---|---|---|
| ADNI→NACC | Fail（F1 下降） | Pass | Pass | 2/3，Fail |
| NACC→ADNI | Pass | Fail（AUC 非劣失败且 F1 下降） | Pass | 2/3，Fail |

Windows 三种子均值：

| 方向 | 方法 | AUC | macro-F1 | Brier | ECE |
|---|---|---:|---:|---:|---:|
| ADNI→NACC | `ce_only` | 0.8751 | 0.7317 | 0.1176 | 0.0690 |
| ADNI→NACC | `apis_only` | 0.9036 | 0.7722 | 0.1271 | 0.0835 |
| NACC→ADNI | `ce_only` | 0.9062 | 0.7576 | 0.1798 | 0.1416 |
| NACC→ADNI | `apis_only` | 0.9270 | 0.7546 | 0.1918 | 0.1442 |

Windows 与 Linux 最大分歧（seed 43，NACC→ADNI，`apis_only - ce_only`）：

| 环境 | ΔAUC | Δmacro-F1 | ΔBrier | ΔECE | ΔSensitivity | Gate |
|---|---:|---:|---:|---:|---:|---|
| Windows | -0.0130 | -0.1063 | +0.1021 | +0.0983 | -0.2617 | Fail |
| Linux | +0.0177 | +0.1815 | -0.1259 | -0.1400 | +0.3893 | Pass |

### 3.2 分析

- 相对基线：Windows 中 APIS 的平均 AUC 在两个方向均高于 CE，但反向平均 F1 略降，且两个方向的平均 Brier/ECE 均变差。排序能力提升没有稳定转化为阈值分类和校准收益。
- 跨 seed 稳定性：Windows 六个 seed×direction gate 仅通过四个，seed 42 正向和 seed 43 反向失败，因此正式 Gate A 为 No-Go。Linux seeds 43/44 的四个 gate 全部通过，但缺少同机 seed 42，不能替代正式三种子判定。
- 机制证据：seed 42 的同受试者 1.5T/3T 配对中，APIS mean|Δp| 相对 CE 和 MixStyle 分别约降低 0.0018、0.0097，三变体 flip rate 均为 0；该结果未形成系统性反证，但不足以推翻性能 gate。
- 数据核验：45 个结果 JSON 均可解析；远程 12 份 target prediction 文件按 subject mean 重新计算 AUC、macro-F1、Brier、敏感度、特异度和准确率，与 `journal_metrics.json` 完全一致。
- 异常与局限：Windows/Linux 在关键 seed 43 反向任务上出现方向相反的大幅差异；远程配置副本、完整依赖版本、checkpoint、队列日志和训练代码指纹未完整上传，暂时不能把差异归因于操作系统、并行方式、依赖或非确定性算子中的某一项。
- 结果结论：当前证据不支持“APIS 双向稳定通关”。可保留“APIS 具有方向和协议依赖的潜在跨域收益”这一探索性结论，但不得扩展为稳健性声明。

## 4. 建议下一步实验指导

- 建议动作：在一个预先指定的参考主机上，用同一代码、同一环境和同一数据清单重跑 seeds 42/43/44 × 双方向 × `{ce_only, mixstyle, apis_only}`；优先复核 seed 43 NACC→ADNI，并重新生成单一三种子 Gate A 报告。
- 建议依据：当前唯一改变正式决策可能性的证据冲突集中在运行环境；同一 seed/方向的 ΔF1 从 Windows -0.1063 变为 Linux +0.1815，已超过一般随机波动可直接忽略的范围。
- 固定条件：冻结 `journal_dual_shift_postfix.yaml` 的全部超参数、split manifest、subject-mean 聚合、CE 基线、seeds 和 Gate A 阈值；归档代码 commit、配置/manifest SHA-256、Python/PyTorch/CUDA/cuDNN 版本、GPU 型号、确定性设置、启动命令及日志。
- 进入条件：代码、配置、数据和环境指纹全部留档后才能启动；只有同机六个 seed×direction gate 全部通过，才允许讨论下一阶段机制验证或任务扩展，否则维持 No-Go 并停止性能扩展。
- 禁止事项：在复现实验完成前，不扩展 MCI/三分类，不恢复 Joint，不补 CDT 多 seed，不根据 target 指标修改 APIS/CDT 超参数，不混合 Windows 与 Linux 的 seed 结果形成一个三种子均值。
