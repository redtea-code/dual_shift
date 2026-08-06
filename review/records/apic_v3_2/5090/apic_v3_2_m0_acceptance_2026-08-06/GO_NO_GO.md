# APIC v3_2 M0 scale-repair acceptance — 2026-08-06

## Verdict

**NO-GO.** `formal_run_allowed` stays **false**. This is **not** an E3 result and does **not** authorize expanded seeds or X+D.

Mechanism scale-repair is evidenced on A2N (success / healthy unit) and on N2A post-bank snapshots (failure / selector-collapse unit). N2A full retrain to epoch 50 with always-saved `last_checkpoint.pt` **does not** lift the post-bank BA collapse (~0.5).

## Scope / code baseline

| 项 | 值 |
|---|---|
| Machine | 5090 host；repo `/zjs/AD_Project/dual_shift_github` |
| Branch | `fix/apic-v3-2-m0` |
| Diagnostics git HEAD | `cee93138cc95b12690d6a7db32bd079d73f142bc`（相对 delta band 等已合入该 commit） |
| 本归档额外本地修复 | calibration floor；radii/device align；style_bank 全覆盖 DataLoader；conditional teacher reload；每 epoch 写 `last_checkpoint.pt`；对应单测 |
| Config | `journal_dual_shift_apic_v3_2_m0_mci_ad_remote.yaml`（本目录 `configs/` 有副本） |
| 范围 | **仅** MCI · ADNI↔NACC · **seed42**（CN 本轮不做） |
| 标准 | `review/analysis/36_apic_v3_2_m0_scale_repair_acceptance_2026-08-05.md` |

## Layer-2 paths（磁盘大文件见 `ARTIFACT_PATHS.json`）

| Unit | Role | Checkpoint | Archived diagnostics |
|---|---|---|---|
| ADNI→NACC seed42 | success | `.../adni_to_nacc/apic_v3_2_x/best_checkpoint.pt`（epoch **47**，bank+teacher） | `layer2/a2n/` |
| NACC→ADNI seed42 | failure / selector-collapse | snapshot `.../n2a_postbank_snapshots/last_checkpoint_epoch7.pt` | `layer2/n2a_epoch7/` |
| NACC→ADNI seed42 | failure persists | snapshot `.../final_last_checkpoint_epoch50.pt` | `layer2/n2a_epoch50/` |

Published N2A **selector clean-best**（无 bank）归档于：
`outputs/journal/apic_v3_2_m0_mci_ad/r4/seed42/nacc_to_adni/published_clean_best_2026-08-06/`（epoch **3**，target BA **0.7036**）。

Retrain 后 job-dir 发布的 best 为 epoch **4**（仍无 bank），target BA **0.6366**；机制审计以 `last_checkpoint` 为准。

## M0 condition scorecard（对照 doc 36）

| # | Condition | A2N (success) | N2A epoch7 last | N2A epoch50 last |
|---|---|---|---|---|
| 1 | K=4 slot fit+calibration | **PASS** 4/4；counts `[28,8,26,33]` | **PASS** 4/4；`[80,56,71,47]` | **PASS** 同左 |
| 2 | 每有效 source slot 有 in-band 替代 | **PASS**；supported 中 band hit=1.0 | **PASS** | **PASS** |
| 3 | supported / nonzero RMS 非零；报告 band hit | **PASS** supported∈[0.41,0.79] | **PASS** ∈[0.49,0.93] | **PASS** 同 epoch7 量级 |
| 4 | unsupported → clean fallback | **PASS** | **PASS** | **PASS** |
| 5 | reload clean 概率与导出 ≤1e-5 | **PASS**（max abs ~1e-7） | **N/A**（对照路径指向 snapshot） | **N/A**（job-dir CSV 存在；export 旁路未对齐） |
| 6 | Layer-2 覆盖成功+失败单元 | **PASS（pair）** | 见下：BA 塌缩 | 塌缩仍在 |

### Key A2N full（n=1322）

- Published target BA：`0.6265`
- Layer-2 target clean/shift BA：`0.6175` / `0.6178`；flip `0.0046`；JS mean `~1.6e-5`
- Assignment NMI：split `0.054`，label `0.022`，field_strength `0.062`，manufacturer `0.040`

### Key N2A epoch-7 post-bank（n=1322）

- Mechanism ON：source supported `~0.91–0.93`；target `~0.49`；band hit among supported `=1.0`
- Performance collapsed：clean=shift BA **0.5** 全 split；**flip=0**，**JS=0**；supported 预测全 class 0
- 历史 published clean-best（epoch 3）target BA **0.7036**（pre-bank；非机制主张）

### Key N2A epoch-50 post-bank（n=1322）

- Bank 仍 4/4；supported 比例与 epoch7 同量级
- BA 仍 **~0.48–0.52**；JS 现为极小非零（`~5e-5…2e-4`）；flip `~0.005–0.03`；预测多为 class 1
- **失败模式 BA 塌缩持续**

## 单测

`conda run -n cyh python -m pytest tests/test_apis_v3_2.py -q` → **10 passed**（含 calibration floor）。见 `unit_tests/pytest_apis_v3_2.txt`。

## Why still NO-GO

1. Doc 36 要求完整 E1/E2 清零 M0 后才能解除 `formal_run_allowed`；本轮仅为 seed42 双向机制证据。
2. N2A 仍为 selector 锁 clean-best + post-bank 性能塌缩；nonzero RMS ≠ 健康决策（JS/flip/BA）。
3. N2A 1e-5 预测一致性对 snapshot 导出未闭环（job-dir CSV 在，export 对照路径未指过去）。
4. 政策：**无 E3**，**`formal_run_allowed=false`**。

## Companion

- 机读：`m0_metrics_summary.json`
- 报告：`review/analysis/37_apic_v3_2_m0_scale_repair_experiment_report_2026-08-06.md`
