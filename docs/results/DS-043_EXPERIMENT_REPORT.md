# DS-043 实验报告：CAPM-GRL 多模态拼接与频域/残差对抗基线

> 实验编号：DS-043
> 主方向：ADNI → NACC
> 任务：scan-filtered MCI vs AD
> 报告状态：9 个变体、3 个正式 seed 已完成（42 / 43 / 44）
> 结果口径：target-test scan-level，均值 ± 标准差

## 1. 摘要

DS-043 在固定 `ADNI_to_NACC`、subject-disjoint target split 和 source-validation checkpoint 选择协议下，比较 CAPM 多模态 source-only 基线 `P0/P0-M` 与 7 个频域/GRL 变体。所有 target label 仅用于冻结模型后的最终评估。

本报告保留原有 AUROC、balanced accuracy（BA）和 accuracy（ACC），并补充五个二分类指标：Precision、Sensitivity/Recall、Specificity、F1-score 和 MCC。按 BA、F1 和 MCC 综合观察，`F1` 是本批次表现最稳定的强候选；但该结论是描述性跨 seed 比较，不构成统计显著性结论。

## 2. 实验协议

| 项目 | 设定 |
|---|---|
| 数据方向 | `ADNI_to_NACC` |
| 任务 | scan-filtered MCI vs AD，二分类 |
| 输入 | MRI + `age, sex, education` |
| target adaptation | 无标签 target MRI；按变体协议使用 target-style 数据 |
| target 评估 | 冻结 checkpoint 后的 target test |
| seeds | 42、43、44 |
| 变体 | P0、P0-M、F0–F3、R1–R3 |
| 主要终点 | AUROC、balanced accuracy |
| 补充指标 | ACC、Precision、Sensitivity/Recall、Specificity、F1、MCC |

严格 UDA 边界：target label、prediction、metric 和模型排名不进入 adaptation 或 selector。P0-M 是 source-only 多模态基线；其 target table 只在最终冻结推理时使用。

## 3. Target 结果

数值为 3 个 seed 的 `均值 ± 标准差`。Sensitivity 与 Recall 在该二分类任务中相同，仅展示一次。

| 变体 | AUROC | BA | ACC | Precision | Sensitivity / Recall | Specificity | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.7027 ± 0.0415 | 0.6097 ± 0.0559 | 0.6644 ± 0.0531 | 0.6047 ± 0.0904 | 0.3651 ± 0.1656 | 0.8543 ± 0.0871 | 0.4344 ± 0.1397 | 0.2525 ± 0.1075 |
| P0-M | 0.7255 ± 0.0278 | 0.5935 ± 0.0309 | 0.6663 ± 0.0396 | 0.6185 ± 0.0367 | 0.2941 ± 0.0720 | 0.8929 ± 0.0112 | 0.3954 ± 0.0707 | 0.2347 ± 0.0611 |
| F0 | 0.7053 ± 0.0232 | 0.6058 ± 0.0391 | 0.6672 ± 0.0243 | 0.6176 ± 0.0918 | 0.4007 ± 0.2778 | 0.8109 ± 0.2005 | 0.4407 ± 0.1593 | 0.2575 ± 0.0425 |
| F1 | 0.7288 ± 0.0290 | 0.6574 ± 0.0055 | 0.6920 ± 0.0260 | 0.6144 ± 0.0716 | 0.5315 ± 0.1204 | 0.7834 ± 0.1101 | 0.5597 ± 0.0516 | 0.3321 ± 0.0195 |
| F2 | 0.7173 ± 0.0509 | 0.6087 ± 0.0383 | 0.6726 ± 0.0541 | 0.6176 ± 0.0749 | 0.3696 ± 0.1024 | 0.8479 ± 0.1006 | 0.4544 ± 0.0843 | 0.2576 ± 0.0887 |
| F3 | 0.7363 ± 0.0239 | 0.6354 ± 0.0315 | 0.6675 ± 0.0829 | 0.5997 ± 0.0950 | 0.5539 ± 0.2254 | 0.7168 ± 0.2694 | 0.5462 ± 0.0718 | 0.3017 ± 0.0713 |
| R1 | 0.7192 ± 0.0231 | 0.6085 ± 0.0144 | 0.6703 ± 0.0434 | 0.6156 ± 0.0573 | 0.3810 ± 0.1334 | 0.8360 ± 0.1257 | 0.4565 ± 0.0785 | 0.2571 ± 0.0395 |
| R2 | 0.7155 ± 0.0247 | 0.6140 ± 0.0246 | 0.6425 ± 0.0371 | 0.5776 ± 0.1198 | 0.4781 ± 0.2443 | 0.7499 ± 0.1961 | 0.4794 ± 0.0981 | 0.2538 ± 0.0266 |
| R3 | 0.6946 ± 0.0231 | 0.6184 ± 0.0229 | 0.6365 ± 0.0813 | 0.5781 ± 0.1619 | 0.5237 ± 0.1898 | 0.7131 ± 0.2353 | 0.5126 ± 0.0283 | 0.2622 ± 0.0860 |

## 4. 结果解读

- `F1` 的 target BA 为 **0.6574 ± 0.0055**，F1-score 为 **0.5597 ± 0.0516**，MCC 为 **0.3321 ± 0.0195**，是综合分类指标最优的变体。
- `F3` 的 AUROC 最高，为 **0.7363 ± 0.0239**；但其 specificity 和 ACC 波动较大，因此不能仅凭 AUROC 宣称整体最优。
- `P0-M` 的 AUROC 高于 `P0`，但 BA、Sensitivity、F1 和 MCC 较低，当前三 seed 结果不支持 table concat 带来稳定分类收益。
- R 系列没有一致优于对应 full-feature GRL 的证据；残差 GRL 的作用应继续按对抗梯度范围解释，不能将 residual 直接称为纯 batch 表示。

## 5. 结论与边界

1. DS-043 的 27 个 seed-variant 运行均已完成并生成 `report.json` 与 `predictions.json`。
2. 在本协议下，`F1` 在 BA/F1/MCC 的综合观察中最具竞争力，但仍需预注册 paired-seed difference 或 bootstrap CI 才能进行显著性判断。
3. P0-M 未显示稳定的多模态 concat 增益，因此按计划不扩展 `F1-M` 至 `R3-M`。
4. 这些结果不能证明已经去除了 scanner/batch effect，也不能把 CAPM 或 residual 表示解释为纯 biological / 纯 batch 信息。

## 6. 结果文件

- 汇总指标：`dual_shift_ds043_run/outputs/DS-043_SUMMARY_METRICS.json`
- 每个运行：`dual_shift_ds043_run/outputs/ds043_capm_concat/` 和 `ds043_capm_grl/` 下的 `report.json`、`predictions.json`
- 实验计划：`dual_shift_ds043_run/docs/DS-043_CAPM_MULTIMODAL_CONCAT_EXPERIMENT_PLAN_2026-09-02.md`
