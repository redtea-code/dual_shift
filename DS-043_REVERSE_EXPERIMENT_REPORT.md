# DS-043 反向实验报告：NACC→ADNI CAPM-GRL 多模态拼接与对抗基线

> 实验编号：DS-043 Reverse  
> 实验方向：NACC → ADNI  
> 协议定位：unsupported-protocol stress test（压力测试）  
> 任务：scan-filtered MCI vs AD  
> 状态：9 个变体 × 3 个 seed，27/27 完成  
> 结果口径：target-test scan-level，均值 ± 标准差  

## 1. 摘要

本实验复用 DS-043 正向实验的 9 个变体（P0、P0-M、F0–F3、R1–R3）和 seeds 42/43/44，将数据方向反转为 NACC→ADNI。全部 27 个运行均成功生成报告、预测和完成记录。

本实验用于检验方向敏感性，不属于预注册主方向，不能与 ADNI→NACC 结果直接合并排名。结果保留 AUROC、balanced accuracy（BA）和 accuracy（ACC），并补充 Precision、Sensitivity/Recall、Specificity、F1-score 与 MCC。

## 2. 实验协议

| 项目 | 设定 |
|---|---|
| 方向 | `NACC_to_ADNI` |
| 协议定位 | unsupported-protocol stress test |
| 任务 | scan-filtered MCI vs AD，二分类 |
| 变体 | P0、P0-M、F0–F3、R1–R3 |
| seeds | 42、43、44 |
| target adaptation | 无标签 ADNI MRI/image-only view |
| target 评估 | 冻结 checkpoint 后的 ADNI target test |
| 主要终点 | AUROC、balanced accuracy |
| 补充指标 | ACC、Precision、Sensitivity/Recall、Specificity、F1、MCC |

target label、prediction、metric 和模型排名不进入 adaptation 或 selector；target label 仅在冻结模型后的最终评估阶段读取。

## 3. Target 结果

Sensitivity 与 Recall 在当前二分类任务中相同，仅展示一次。

| 变体 | AUROC | BA | ACC | Precision | Sensitivity / Recall | Specificity | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.6585 ± 0.0119 | 0.6206 ± 0.0049 | 0.6337 ± 0.0123 | 0.6533 ± 0.0561 | 0.4548 ± 0.1044 | 0.7863 ± 0.1013 | 0.5279 ± 0.0468 | 0.2619 ± 0.0185 |
| P0-M | 0.6379 ± 0.0224 | 0.5993 ± 0.0199 | 0.6158 ± 0.0239 | 0.6243 ± 0.0298 | 0.4061 ± 0.0230 | 0.7925 ± 0.0295 | 0.4920 ± 0.0248 | 0.2165 ± 0.0450 |
| F0 | 0.6296 ± 0.0251 | 0.5983 ± 0.0185 | 0.6117 ± 0.0208 | 0.6116 ± 0.0398 | 0.4285 ± 0.0495 | 0.7681 ± 0.0487 | 0.5018 ± 0.0306 | 0.2105 ± 0.0392 |
| F1 | 0.6051 ± 0.0271 | 0.5647 ± 0.0323 | 0.5734 ± 0.0300 | 0.5784 ± 0.1200 | 0.4212 ± 0.1794 | 0.7082 ± 0.1791 | 0.4600 ± 0.1012 | 0.1478 ± 0.0808 |
| F2 | 0.6386 ± 0.0241 | 0.5885 ± 0.0202 | 0.5949 ± 0.0206 | 0.5861 ± 0.0606 | 0.4772 ± 0.1698 | 0.6999 ± 0.1520 | 0.5068 ± 0.0964 | 0.1885 ± 0.0320 |
| F3 | 0.6624 ± 0.0397 | 0.6020 ± 0.0350 | 0.6165 ± 0.0335 | 0.6917 ± 0.1583 | 0.3906 ± 0.1832 | 0.8135 ± 0.1445 | 0.4623 ± 0.1447 | 0.2439 ± 0.0609 |
| R1 | 0.6418 ± 0.0165 | 0.6061 ± 0.0114 | 0.6136 ± 0.0161 | 0.6113 ± 0.0569 | 0.4890 ± 0.1590 | 0.7232 ± 0.1488 | 0.5281 ± 0.0699 | 0.2261 ± 0.0180 |
| R2 | 0.6327 ± 0.0112 | 0.5832 ± 0.0225 | 0.5990 ± 0.0170 | 0.6065 ± 0.0270 | 0.3783 ± 0.1435 | 0.7880 ± 0.0990 | 0.4536 ± 0.0917 | 0.1859 ± 0.0297 |
| R3 | 0.6251 ± 0.0429 | 0.5890 ± 0.0361 | 0.6016 ± 0.0369 | 0.6002 ± 0.0719 | 0.4212 ± 0.0782 | 0.7568 ± 0.0837 | 0.4897 ± 0.0539 | 0.1919 ± 0.0770 |

## 4. 结果解读

- P0 baseline 的 target BA 为 **0.6206 ± 0.0049**，作为反向方向的 CAPM source-only 参考。
- P0-M 的 target BA 为 **0.5993 ± 0.0199**、F1 为 **0.4920 ± 0.0248**；是否优于 P0 需结合 seed-level 配对差异，不能仅依据均值判断。
- 9 个变体中，按 BA 最高者为 **P0**（0.6206 ± 0.0049）。按 AUROC 最高者为 **F3**（0.6624 ± 0.0397）。
- 反向结果只能说明模型在 NACC→ADNI 方向上的压力测试表现，不能证明方法具有跨方向稳定性，也不能替代 ADNI→NACC 主结果。

## 5. 运行完整性

- `report.json`：27/27
- `predictions.json`：27/27
- exit records：27/27，全部 `exit_code=0`、`state=completed`
- 当前训练进程与 GPU 计算进程：均已结束

## 6. 结果文件

- 汇总：`dual_shift_ds043_run/outputs/DS-043_REVERSE_SUMMARY_METRICS.json`
- 原始结果：`dual_shift_ds043_run/outputs/ds043_reverse/{variant}/seed_{42,43,44}/`
- 反向配置：`dual_shift_ds043_run/journal_ds043_capm_concat_scan_filtered_nacc_to_adni.yaml`
- 实验计划：`dual_shift_ds043_run/docs/DS-043_REVERSE_NACC_TO_ADNI_EXPERIMENT_PLAN.md`
