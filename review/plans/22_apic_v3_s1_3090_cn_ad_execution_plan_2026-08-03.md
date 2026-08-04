# APIC-V3-S1-3090-CN：本机 CN_vs_AD image-only 主矩阵执行计划

## 基本信息

- 日期：2026-08-03
- 负责人：cyh / 本机 RTX 3090
- 状态：P3 完成（CN 12/12 已落盘；见 `review/records/3090/23_apic_v3_cn_ad_s1_metrics_2026-08-04.md`）
- Git commit：`40c7371`（分支 `run/apic-v3-s1-cn-3090`）
- 关联文档：
  - `review/plans/20_apic_v3_image_only_comparison_plan_2026-08-03.md`（科学预注册）
  - `review/operations/21_apic_v3_remote_screening_handoff_2026-08-03.md`（执行交接）
  - `docs/SCAN_AWARE_DATA_REALITY_AND_CLAIM_BOUNDARY.md`（主张边界）
- 关联数据文件：运行后写入 `outputs/journal/apic_v3_screening_cn_ad/`（当前无）

## 1. 实验指导与依据

- 研究问题：在 **CN_vs_AD**、不向模型提供 acquisition metadata（`A`）的前提下，APIC v3（`apic_v3_x`）是否相对同模态 `ce_x` 与 `mixstyle_x` 获得更稳定的 ADNI/NACC 双向跨队列 subject-level balanced accuracy？
- 假设与基线：预期 `apic_v3_x` 在多数 seed×direction 上不低于两基线；主比较仅在 **严格 image-only `X`** 轴内进行。本机不做 `X+D`。
- 实施依据：plan 20 冻结科学矩阵与 Gate S1；ops 21 冻结命令与回传；本文件将完整筛选中的 **CN_vs_AD 份额** 固定到本机 3090。
- 实验范围：
  - Task：**仅 CN_vs_AD**（标签 `1→0` CN，`3→1` AD）
  - Directions：`ADNI_to_NACC`、`NACC_to_ADNI`
  - Seeds：`42`、`43`（`split_seed=42` 固定划分）
  - Variants：`ce_x`、`mixstyle_x`、`apic_v3_x`
  - 规模：`2 directions × 2 seeds × 3 variants = 12 runs`
  - Hold-out：`data/claim/paired_holdout_subjects.json` 的 `subjects_all_paired`（73 人）在划分前排除
  - **本机明确不做：** `MCI_vs_AD`、secondary `X+D`、seeds 44–46、plan 19 support-aware 队列、APIS v2 claim 续跑
- 判定标准：
  - 本机交付标准：12/12 正式 run 落盘；smoke 与 fingerprint 通过；产物满足 ops 21 回传字段（CN 子集）
  - **完整 APIC-V3-S1 Gate：** 需要 CN+MCI 共四个 task×direction；本机 **不得** 在仅有 CN 时宣称 `pass=true`，**不得** 启动 secondary
  - 本机允许产出 **CN_vs_AD 描述性 ΔBA 表**（逐 seed、逐 direction），不作显著性优效声明
  - 停止条件：fingerprint/smoke 失败；路径或协议身份不匹配；系统性 NaN/单类预测且无法在不改协议下修复；试图覆盖 APIS v2 r2 目录

## 2. 可复现记录

- 配置文件：`journal_dual_shift_apic_v3_screen_cn_ad.yaml`
- 数据与划分版本：YAML 内 `cohorts` / `scan_manifest`；开训后记录各 job `split_manifest.json` 的 SHA-256；本机路径映射（只改路径、不改协议字段）：
  - `F:/ADNI/ADNI_dataset`、`F:/ADNI/NACC_dataset`、`F:/ADNI/scan_manifests`
  - `F:/NACC/MRI_mulclass3.csv`、`D:/ADNI/merge.csv`
- 随机种子：`split_seed=42`；`training_seeds={42,43}`
- 环境与硬件：Windows；Conda `pytorch`；**NVIDIA RTX 3090 24GB**；每卡最多 1 worker
- 启动命令：

```bash
# P0：clean checkout 到含 screening 的 main commit 后记录 HEAD

# P1：配置与环境指纹（仅 CN）
python experiments/run_apic_v3_screening.py \
  --phase primary \
  --configs journal_dual_shift_apic_v3_screen_cn_ad.yaml \
  --fingerprint-only

# P2：one-epoch smoke（独立输出树；不得计入正式矩阵）
# CN × ADNI→NACC × seed42 × {ce_x,mixstyle_x,apic_v3_x}
# 记录：checkpoint reload、target clean inference、memory valid slots、
# valid_intervention_frac、NaN、单类预测检查

# P3：正式 12-run 主矩阵
python experiments/run_apic_v3_screening.py \
  --phase primary \
  --configs journal_dual_shift_apic_v3_screen_cn_ad.yaml \
  --gpu-ids 0 \
  --max-workers 1
```

- 工作区状态：开训前必须 **clean**；dirty 时不得启动正式矩阵
- 产物位置：
  - `outputs/journal/apic_v3_screening_cn_ad/protocol_freeze/`
  - `outputs/journal/apic_v3_screening_cn_ad/s1/seed{42|43}/{adni_to_nacc|nacc_to_adni}/{ce_x|mixstyle_x|apic_v3_x}/`
  - 日志：`outputs/journal/apic_v3_screening_cn_ad/s1/primary/seed*_*.log`
  - **禁止写入：** `outputs/journal/dual_shift_apis_v2/`

### 阶段门控

| 阶段 | 动作 | 进入下一阶段条件 |
|---|---|---|
| P0 | 检出代码；核对路径与 73 人 hold-out | 数据可读；工作区 clean |
| P1 | `--fingerprint-only` | freeze 文件写出；config hash 稳定 |
| P2 | CN×A→N×seed42 三变体 one-epoch smoke | reload/clean-path/memory/无 NaN/无系统单类 |
| P3 | 正式 primary 12 runs | 12/12 `journal_metrics.json` 且协议身份匹配 |
| P4 | 本机 CN 描述性汇总 | 不调用完整 Gate 冒充通过 |
| P5 | 回传 / `review/records/3090/` 登记 | ops 21 §6 的 CN 子集齐全 |

不完整 job：按 launcher 语义 **整 job 三变体重跑**；不得用 smoke epoch 预算或 smoke 目录冒充正式结果。

## 3. 分析与结果

### 3.1 结果

待运行。正式数值必须由 `journal_metrics.json` / predictions 生成，不在本计划中手工覆盖预注册条件。

| 方法/条件 | 主要指标（target BA） | 辅助指标 | 判定 | 备注 |
|---|---:|---:|---|---|
| `ce_x` | 待运行 | AUC / F1 / SEN / SPE / Brier / ECE | 待判定 | X 基线 |
| `mixstyle_x` | 待运行 | 同上 | 待判定 | X 主对照 |
| `apic_v3_x` | 待运行 | 同上 + APIC memory/gate 审计 | 待判定 | 提出方法 |

### 3.2 分析

- 相对基线：待运行后报告每个 direction×seed 的 `ΔBA(apic−ce)`、`ΔBA(apic−mixstyle)` 及胜负计数
- 异常与局限：两 seed 不能宣称显著优效；仅 CN 不能代替完整 Gate S1；NACC 无 1.5T，跨队列差异不可解释为单一场强因果效应
- 结果结论：仅允许“本机 CN 主矩阵是否按协议完成 + 描述性趋势”；完整 Go/No-Go 留给 CN+MCI 齐套后的 Gate S1

## 4. 建议下一步实验指导

- 建议动作：P0→P3 跑完并登记后，在另一机或本机后续窗口补齐 **MCI_vs_AD** 同等 12-run，再运行 `experiments/report_apic_v3_screening.py`
- 建议依据：Gate S1 与 ops 21 secondary 均要求双任务齐套
- 固定条件：模态 `X`、三变体、split_seed、标签映射、训练预算、主终点、禁止输入 `A` 保持不变
- 进入条件：
  - 启动 P3：P1+P2 通过
  - 启动完整 Gate S1：CN+MCI primary 均齐全且 config hash 一致
  - 启动 `X+D` secondary：Gate JSON `pass=true` 且 hash 匹配
- 禁止事项：看 target 调参；用 `X+D` 对比 `X` 归因 APIC；两 seed 写显著优效；覆盖 APIS v2 r2；在仅 CN 完成时宣称 Gate S1 通过或启动 secondary；本计划执行期间插入 plan 19 完整 3D 矩阵

## 5. 回传清单（CN 子集，对应 ops 21 §6）

- [ ] `protocol_freeze/`（CN）
- [ ] 每 job `split_manifest.json`、`metadata_match_audit.json`、日志
- [ ] 三变体 `journal_metrics.json`、逐 subject predictions、checkpoint
- [ ] 若生成：job 级 `summary.json` / paired 比较文件
- [ ] 失败 job 的退出码、完整日志、GPU/驱动状态
- [ ] 本机记录登记至 `review/records/3090/`（执行后另建 records，不改写本计划判定区）
