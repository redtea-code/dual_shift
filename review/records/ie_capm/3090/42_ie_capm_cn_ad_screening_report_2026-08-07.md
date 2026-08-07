# IE-CAPM-CN-AD-3090-20260807：CN vs AD 筛选矩阵阶段报告

## 基本信息

- 日期：2026-08-07
- 负责人：本机 3090 节点
- 状态：**运行中（10/16 单元完成）**；ADNI→NACC 两 seed 全矩阵已齐，NACC→ADNI 仅 seed42 的 B0/B1 完成
- Git commit（启动配置登记）：`7b430db`；当前分支 `run/ie-capm-cn-ad-3090`
- 关联文档：
  - `docs/IE_CAPM_APIC_V3_2_ALIGNED_EXPERIMENT_PROTOCOL.md`
  - `review/plans/32_ie_capm_paired_experiment_plan_2026-08-07.md`
  - `review/records/ie_capm/3090/41_ie_capm_cn_ad_resume_2026-08-07.md`
  - `docs/EXPERIMENT_RECORD_TEMPLATE.md`
- 关联数据文件：
  - 原始产物：`outputs/journal/ie_capm_cn_ad/`
  - 机读汇总：`review/records/ie_capm/3090/ie_capm_cn_ad_2026-08-07/metrics_main_table.csv`
  - 配对差分：`review/records/ie_capm/3090/ie_capm_cn_ad_2026-08-07/paired_target_ba_deltas.csv`
  - gate 表：`review/records/ie_capm/3090/ie_capm_cn_ad_2026-08-07/gate_summary_table.csv`
  - holdout SHA-256：`f60ac915b6e2a5b40d46a4a6a3a8fad89f620104a059694e626eeaf027e12770`（`data/claim/paired_holdout_subjects.json`）

## 1. 实验指导与依据

- 研究问题：在 **CN vs AD**、固定三变量表 `[age, sex, education]` 下，IE-CAPM（P1）相对 CAPM 控制（B1）是否提高外部 target 的 subject-mean balanced accuracy，并满足协议筛选门。
- 假设与基线：P1 应不低于/优于 B1；B2（P1 权重 + `force_capm=True`）应与无门控 CAPM 路径可对照；B0 提供纯影像参照。
- 实施依据：
  - 协议卫生（holdout、batch、bootstrap、raw var_specs、不 carve target）对齐 `docs/IE_CAPM_APIC_V3_2_ALIGNED_EXPERIMENT_PROTOCOL.md`
  - **任务不照搬**：正式协议任务为 `MCI_vs_AD`；本机章程为 **CN vs AD**（见记录 41）
- 实验范围：CN vs AD × 方向 {ADNI→NACC, NACC→ADNI} × seeds {42, 43} × {B0, B1, P1}（+ B2 评估）
- 判定标准（运行前固定，摘自协议 §8.2，用于 CN 筛选口径）：
  1. B2 与 B1 无门控结果一致（再解释 P1）
  2. P1 在两方向、两 seed 上方向一致
  3. 固定三变量下无明显性能损失
  4. gate 统计跨 seed 可复现
  5. A2 未跑（本轮不判定）
  - 主终点：`external target + subject_mean + balanced_accuracy`
  - **禁止表述**：扫描因果校正、稳定域泛化已证明、可直接当作期刊主张

## 2. 可复现记录

- 配置文件：`journal_ie_capm_cn_ad.yaml`
- 数据与划分：`subjects_all_paired` 排除后 subject-level 6/2/2；`split_seed=42`；manifest hash 见 `ie_capm_cn_ad_2026-08-07/split_manifest_hashes.json`
- 随机种子：训练 seed `42, 43`
- 环境与硬件：Windows · RTX 3090 · Conda `pytorch` · CUDA
- 启动命令：

```text
python experiments/train_journal.py \
  --config_path journal_ie_capm_cn_ad.yaml \
  --study --directions ADNI_to_NACC NACC_to_ADNI \
  --seeds 42 43 \
  --variants ie_capm_img ie_capm_force ie_capm \
  --device cuda \
  --output-dir outputs/journal/ie_capm_cn_ad
```

（中途曾在 epoch 40 崩溃后无 `--force-variants` 重启；已完成单元保留。）

- 工作区状态：dirty（主要为 `__pycache__` / 运行产物）
- 产物位置：`outputs/journal/ie_capm_cn_ad/{direction}/seed_{seed}/{variant}/`

### 2.1 完成率

| 方向 | seed | B0 img | B1 force | P1 ie_capm | B2 force-eval |
|---|---:|---|---|---|---|
| ADNI→NACC | 42 | DONE | DONE | DONE | DONE |
| ADNI→NACC | 43 | DONE | DONE | DONE | DONE |
| NACC→ADNI | 42 | DONE | DONE | **运行中**（last ckpt ~epoch 17） | — |
| NACC→ADNI | 43 | 未开始 | 未开始 | 未开始 | — |

**完成：10/16 指标文件**（含 2 个 B2）。

## 3. 分析与结果

### 3.1 Target · subject-mean balanced accuracy（主终点）

| seed | direction | B0 img | B1 force | P1 ie_capm | B2 (P1+force) | P1−B1 | P1−B0 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | ADNI→NACC | 0.8464 | 0.8186 | **0.8508** | 0.8508 | +0.0322 | +0.0044 |
| 43 | ADNI→NACC | 0.8122 | 0.8322 | **0.8405** | 0.8405 | +0.0084 | +0.0283 |
| 42 | NACC→ADNI | 0.8276 | 0.7364 | — | — | — | — |
| 43 | NACC→ADNI | — | — | — | — | — | — |
| **mean** | ADNI→NACC | 0.8293 | 0.8254 | **0.8457** | 0.8457 | **+0.0203** | +0.0164 |

辅助指标（ADNI→NACC，target）：见 `metrics_main_table.csv`（AUC / SEN / SPE / Brier）。

### 3.2 协议筛选门（仅就已完成 ADNI 份额）

| 门控项 | 证据 | 当前判定 |
|---|---|---|
| 1. B2 与 B1 一致 | seed42：B1=0.8186 vs B2=0.8508（Δ=+0.032）；seed43：0.8322 vs 0.8405（Δ=+0.008）。B2 与 P1 几乎数值重合 | **未通过 / 不一致**（尤其 seed42） |
| 2. 两方向 × 两 seed 方向一致 | NACC→ADNI 的 P1 尚未完成 | **不能关闭** |
| 3. 无明显损失 | ADNI 上 P1≥B1、P1≥B0（两 seed） | ADNI 份额暂支持；全矩阵未齐 |
| 4. gate 跨 seed 可复现 | P1 gate_mean 各 stage ≈ 0.993–0.998（两 seed 均近 1） | **可复现，但近恒等门控** |
| 5. A2 | 未实现/未跑 | 不适用 |

**机制观察：** P1 的 stage gate 均值 ≈ 0.994–0.998，故 `force_capm` 评估（B2）与 P1 指标几乎相同；当前 ADNI 上“P1 相对 B1 的增益”主要来自 **P1 训练轨迹/字段**，而非可观测的大幅度门控衰减。这与“证据门控改变调制”的机制叙事不完全同构，报告时必须分开。

### 3.3 分析

- 相对基线（ADNI→NACC 仅）：P1 平均 target BA 高于 B1 约 +0.020，两 seed 符号一致为正；相对 B0 平均约 +0.016，但 seed42 相对 B0 仅 +0.004（接近持平）。
- B1 在 NACC→ADNI seed42 已出现明显弱于 B0（0.736 vs 0.828），提示表格融合控制在该方向不稳定；P1 结果尚未出，不得外推。
- 异常与局限：
  1. 矩阵未完成，**筛选门不能正式通过或失败归档**
  2. 任务为 CN，不是协议正文的 MCI 正式矩阵
  3. B1≠B2（seed42）阻碍“先确认无门控路径再解释 P1”
  4. gate≈1 使证据校准几乎不改变反事实输出
- 结果结论（阶段性）：**不支持**任何期刊级或正式筛选通过结论；ADNI 方向出现可记录的 P1≥B1 趋势，但因门控近恒等、B1/B2 不一致、反向未齐，**停止扩展主张，继续跑完矩阵**。

## 4. 建议下一步实验指导

- 建议动作：保持当前配置不变，跑完 NACC→ADNI × {42,43} × {B0,B1,P1(+B2)}；完成后刷新本报告终稿与 `metrics_main_table.csv`。
- 建议依据：协议 §8.2 要求两方向两 seed；当前仅 ADNI 齐。
- 固定条件：`journal_ie_capm_cn_ad.yaml`；seeds {42,43}；三变量 raw 表；subject BA 选点；不看 target 调参。
- 进入条件：16/16 指标文件齐备后，再做筛选门总判定；仅当门控 1–4 满足才考虑 A1/A2。
- 禁止事项：
  - 不得仅凭 ADNI 两 seed 宣称 IE-CAPM 稳定优于 CAPM
  - 不得与 APIC v3_2 数字直接比“谁更好”（协议 §6.3）
  - 不得把 gate≈1 写成“证据校准已发挥作用”
  - 不得在本机改跑 MCI 冒充本轮 CN 结果，或反之

## 提交前检查

- [x] 日期、Git commit、关联文档和关联数据文件均已填写
- [x] 判定标准在结果分析前已固定（协议筛选门）
- [x] 表格数值可由 `metrics_main_table.csv` / 各 `journal_metrics.json` 复核；缺失单元已标明
- [x] 结论区分事实（ADNI 完成）与推断（全矩阵筛选）
- [x] 下一步含固定条件、进入条件与禁止事项
