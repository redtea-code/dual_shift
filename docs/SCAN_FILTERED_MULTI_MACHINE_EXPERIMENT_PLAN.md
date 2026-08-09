# Scan-Filtered 多机器实验计划

版本：1.0
协议：`scan_filtered_v1_2026-08-08`
任务：`<TASK>`
发布分支：`codex/scan-filtered-loader`

冻结代码：不可变标签 `plan34-scan-filtered-v1`。每台机器必须 checkout 此标签的
detached HEAD，并将实际 SHA 写入 `git_commit.txt`；不得只记录分支名或在实验期间
执行 `git pull`。

## 1. 研究问题

验证在以下固定数据协议下，CAPM/APIC V3_2 系列是否能稳定完成跨队列分类：

```text
ADNI 1.5T scan-filtered -> NACC 3T
NACC 3T -> ADNI 1.5T scan-filtered
```

ADNI 中同时拥有 1.5T 和 3T 的 subject 保留其 1.5T scan；其 3T scan 不进入新 manifest。不得使用旧的 `subjects_all_paired` subject-wide 排除。

## 2. 机器分工

每台机器只执行一个明确角色，并把结果写入独立目录：

| 角色 | 工作内容 | 可读取 target 指标 |
|---|---|---|
| M0 | 环境、代码、manifest、场强与 subject 泄漏审计 | 否 |
| M1 | 一个指定任务的 `ADNI_to_NACC`，seeds 42/43 | E2 后统一汇总 |
| M2 | 同一任务的 `NACC_to_ADNI`，seeds 42/43 | E2 后统一汇总 |
| M3 | 另一指定任务或独立复跑 | E2 后统一汇总 |

机器之间不得共享 checkpoint、validation 结果或手工挑选的 epoch。每台机器必须从同一 commit、同一 manifest hash 和同一 YAML 启动。

## 3. E0 工程门

在任何训练前执行：

```text
git fetch origin --tags
git checkout --detach plan34-scan-filtered-v1
git rev-parse HEAD
<PYTHON> -m pytest -q tests/test_scan_filtered_loader.py tests/test_evidence_calibrated_capm.py tests/test_scale_table_transformer_ablation.py
<PYTHON> -m py_compile data/scan_filtered_loader.py data/journal_dataset.py experiments/train_journal.py
```

E0 必须满足：测试通过、Python 编译通过、工作区没有对协议代码的未提交修改。缓存和 outputs 不应提交。

## 4. Manifest 门

先由 M0 使用 `write_filtered_manifest` 生成：

```text
<MANIFEST_ROOT>/ADNI_1p5T_scan_filtered_manifest.csv
<MANIFEST_ROOT>/NACC_3T_scan_filtered_manifest.csv
<MANIFEST_ROOT>/filter_audit_ADNI.json
<MANIFEST_ROOT>/filter_audit_NACC.json
```

检查：

- ADNI manifest 不含 `3T`、`unknown` 或 `other`；
- NACC manifest 全部为 `3T`；
- ADNI 配对 subject 的 1.5T 行存在；
- source train/val/test subject 集合互不相交；
- target subject 不与 source subject 重叠；
- manifest 中包含 `protocol_version=scan_filtered_v1_2026-08-08`；
- `paired_origin` 仅用于审计，不作为模型输入。

M0 需保存 manifest SHA256、行数、subject 数、场强计数和 paired-origin 计数，形成 `e0_manifest_audit.json`。

## 5. E1 Smoke

每个任务、每个方向先执行 1 epoch、CPU 或单 GPU smoke。任务必须从下表选择：

| task | label mapping | config/output suffix |
|---|---|---|
| `MCI_vs_AD` | `{2: 0, 3: 1}` | `mci_ad` |
| `CN_vs_AD` | `{1: 0, 3: 1}` | `cn_ad` |

每个任务使用独立的 resolved YAML、manifest audit 和输出目录，不得混用 label mapping。

结构消融的 `MCI_vs_AD` 模板是 `journal_scale_table_scan_filtered_1p5t_mci_ad.yaml`；
`journal_dual_shift_scan_filtered_1p5t_mci_ad.yaml` 仅用于 APIC 接线 smoke。执行
`CN_vs_AD` 时复制相应模板为任务专用 YAML，并只修改以下字段：

```yaml
task:
  mode: binary
  label_mapping: {1: 0, 3: 1}
  class_names: {0: CN, 1: AD}
  positive_class: AD
claim:
  task_pair: CN_vs_AD
output_root: <OUTPUT_ROOT>/cn_ad
```

执行 `MCI_vs_AD` 时使用 `{2: 0, 3: 1}` 和 `{0: MCI, 1: AD}`。除任务标签、任务名和输出根目录外，不得修改 YAML。

E1 必须从任务 YAML 生成独立的 `resolved_smoke_<task>.yaml`，只额外将
`training.epochs` 改为 `1`、`evaluation.bootstrap_samples` 改为 `10`，并记录该文件的 hash；E2 不得使用 smoke YAML。

```text
<PYTHON> experiments/train_journal.py --config_path <SCALE_SMOKE_CONFIG>.yaml --direction ADNI_to_NACC --variants image_only capm transformer_cross --output-dir <OUTPUT_ROOT>/<task>/scale_smoke/adni_to_nacc --device cpu
```

结构消融使用 `journal_scale_table_scan_filtered_1p5t_mci_ad.yaml`。仅为验证保留的
APIC 接线时，使用 `journal_dual_shift_scan_filtered_1p5t_mci_ad.yaml` 并单独运行：

```text
<PYTHON> experiments/train_journal.py --config_path <APIC_SMOKE_CONFIG>.yaml --direction ADNI_to_NACC --variants ce_x mixstyle_x apic_v3_2_x --output-dir <OUTPUT_ROOT>/<task>/apic_smoke/adni_to_nacc --device cpu
```

NACC→ADNI 只替换 `--direction` 和输出目录。E1 只检查接线、样本数、shape、checkpoint、collapse guard 和 target 未读；不作性能主张。

## 6. E2 正式实验

对每个 `scale_table_ablation.preset`（`layer3_patch2`、`layer4_pixel`、
`layer5_pixel`）生成独立的 resolved YAML。每个尺度固定运行
`image_only capm conv_gate original_capm transformer_self transformer_cross`；
APIC smoke 不进入结构主表，也不得与该矩阵合并统计。

M1/M2 分别执行自己的方向：

```text
<PYTHON> experiments/train_journal.py --config_path <SCALE_TASK_CONFIG>.yaml --study --directions ADNI_to_NACC --seeds 42 43 --output-dir <OUTPUT_ROOT>/<task>/<preset>/adni_to_nacc
```

M2 将 `--directions` 改为 `NACC_to_ADNI`。训练预算、variants、checkpoint 选择、collapse guard、subject_mean 聚合和 200 次 subject-level bootstrap 均不可改变。

主要报告：external target balanced accuracy。辅助报告 AUROC、sensitivity、specificity、Brier、ECE、macro-F1、按场强审计指标及 subject-level prediction。

## 7. 停止规则

出现以下任一情况立即标记 `NO-GO`，停止该机器扩展：

- loader 发现旧 paired holdout 或混合场强 manifest；
- target 被用于 early stopping、checkpoint 或阈值选择；
- source split 出现 subject 重叠；
- ADNI target 中出现 3T/unknown；
- APIC/CAPM 发生类别塌缩且未通过 collapse guard；
- 代码、YAML、manifest hash 与 E0 记录不一致。

## 8. 结果交付格式

每台机器只回传以下内容，不回传原始 MRI：

```text
<machine_id>/e0_manifest_audit.json
<machine_id>/resolved_config.yaml
<machine_id>/git_commit.txt
<machine_id>/manifest_hashes.json
<machine_id>/<direction>/seed_42/summary.json
<machine_id>/<direction>/seed_43/summary.json
<machine_id>/<direction>/seed_*/split_manifest.json
<machine_id>/<direction>/seed_*/journal_metrics.json
<machine_id>/e2_status.md
```

`e2_status.md` 必须写明：机器、GPU、Python 环境、commit、manifest hash、方向、seed、是否通过 collapse guard、是否发生任何人工干预。

## 9. 分发给远程机器的提示词

将以下文本连同本文件和 YAML 一起发送：

```text
你负责执行 dual_shift 的 scan-filtered 期刊实验。严格遵守以下规则：

1. `git fetch origin --tags` 后执行 `git checkout --detach plan34-scan-filtered-v1`；确认 `git rev-parse HEAD` 与主控提供的 SHA 一致，不得执行 `git pull`。
2. 使用本机已验证的 Python 环境 `<PYTHON>`，不要升级依赖。
3. 只使用主控指定的 `<TASK_CONFIG>.yaml`，并确认任务是 `<TASK>`。
4. ADNI 只允许 1.5T scan；NACC 只允许 3T scan。ADNI 中拥有 3T 的 subject 仍保留其 1.5T，但 3T 行必须不存在于 manifest。
5. 不得使用 data/claim/paired_holdout_subjects.json，不得启用 claim.exclude_subjects_json。
6. 不得读取 target 指标来选 checkpoint、改超参数或决定是否重跑。
7. 先执行 E0 和 manifest audit，再从任务 YAML 生成 smoke resolved YAML（仅 epochs=1、bootstrap=10），执行 E1；E0/E1 不通过时停止并报告，不得自行修改正式协议。
8. 结构任务只能是 `MCI_vs_AD`（mapping `{2:0,3:1}`）或 `CN_vs_AD`（mapping `{1:0,3:1}`）；方向是 `<DIRECTION>`；seed 为 42、43。每个 preset 的 variants 固定为 `image_only capm conv_gate original_capm transformer_self transformer_cross`。APIC smoke 仅使用 `ce_x mixstyle_x apic_v3_2_x`，不进入结构主表。
9. 所有机器路径使用本机配置中的相对路径或占位符替换；不得把本机绝对路径写入共享文档或结果说明。
10. 输出只能放在 `<OUTPUT_ROOT>/<TASK>/<DIRECTION>/`，不要提交 outputs、pycache 或原始 MRI。
11. 完成后只回传 e0_manifest_audit.json、manifest/config/git hash、split_manifest、summary、metrics 和 e2_status.md。

报告格式：
- E0：pass/fail + pytest/py_compile 结果
- Manifest：ADNI/NACC 行数、subject 数、场强计数、paired-origin 计数、hash
- E1：每个 variant 的 source-val BA、collapse guard、是否读取 target
- E2：每个 seed 的 target BA/AUROC/ECE/Brier，以及失败原因
- 未通过时：停止，不要用 target 结果调参
```

## 10. 汇总审阅

主控机器只在所有方向和 seed 完成后合并结果。先检查 commit、YAML、manifest hash 和 split manifest，再比较性能。任何机器的协议不一致结果都单独归档，不进入主表。
