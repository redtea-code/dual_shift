# DS-040: CAPM-conditioned frequency UDA with domain/intensity GRL

Status: **COMPLETED SCREENING / MECHANISM CLAIM INCONCLUSIVE**

Plan: [DS-040 plan](../../plans/DS-040_CAPM_CONDITIONED_FREQUENCY_GRL.md)

## 1. Executive summary

The registered DS-040 matrix is complete: 8 variants × 2 seeds × 2 directions = **32/32 real-data screening cells**. ADNI→NACC is the primary direction; NACC→ADNI is the prespecified extension. The matched model is `original_capm` with `layer4_pixel`, CAPM variables `[age, sex, education]`, GRL coefficient 1.0, projector rank 32, and source-validation-only selection.

The results demonstrate an executable screening matrix with strong seed and direction dependence. They do not establish residual GRL superiority or GRL complementarity: each cell used a one-epoch screening budget, and the audit artifact does not independently bind every epoch-level mechanism record to the selected checkpoint.

## 2. Protocol and variants

| Item | Value |
|---|---|
| Task | MCI vs AD, scan-filtered |
| Primary | ADNI→NACC |
| Extension | NACC→ADNI |
| Variants | P0, F0–F3, R1–R3 |
| Seeds | 42, 43 |
| Feature | `layer4_pixel` |
| CAPM variables | age, sex, education |
| Training budget | 1 epoch per screening cell |
| Selection | source-validation-only |
| Target evaluation | exploratory post-selection report |

P0 is source-only CAPM; F0 is CAPM plus raw-Fourier/attention control; F1/F2/F3 apply domain, intensity, or both GRL heads to complete CAPM features; R1/R2/R3 apply the corresponding heads to the task-residual branch.

## 3. Artifact and audit verification

Every cell contains `report.json`, `audit.json`, `predictions.json`, and `best.pt`. The audits record exact variant flags, model signature, GRL coefficient, code hash, `target_labels_used_for_adaptation=false`, `target_labels_used_for_selection=false`, and `selection_source_validation_only=true`. Target labels are used only for final exploratory target reporting.

Mechanism history is present in `report.json.variant_history` for F0–F3/R1–R3. P0 has no frequency-variant history. The lack of an independently checkpoint-bound longitudinal mechanism record is retained as a limitation.

## 4. ADNI→NACC primary target-test results

Values are mean ± sample SD across seeds 42/43; Δ values are means relative to F0.

| Variant | BA | AUROC | Macro-F1 | Sensitivity | Specificity | Δ BA vs F0 | Δ AUROC vs F0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.5263 ± 0.0347 | 0.6466 ± 0.0491 | 0.4568 ± 0.0882 | 0.0808 ± 0.1023 | 0.9717 ± 0.0330 | 0.0009 | -0.0436 |
| F0 | 0.5254 ± 0.0359 | 0.6901 ± 0.0164 | 0.4528 ± 0.0938 | 0.0766 ± 0.1083 | 0.9742 ± 0.0365 | 0.0000 | 0.0000 |
| F1 | 0.5283 ± 0.0281 | 0.6745 ± 0.0035 | 0.4623 ± 0.0815 | 0.0896 ± 0.1027 | 0.9671 ± 0.0465 | 0.0030 | -0.0156 |
| F2 | 0.5320 ± 0.0332 | 0.6675 ± 0.0084 | 0.4640 ± 0.0839 | 0.0851 ± 0.0963 | 0.9789 ± 0.0299 | 0.0066 | -0.0227 |
| R1 | 0.5321 ± 0.0454 | 0.6531 ± 0.0075 | 0.4633 ± 0.1086 | 0.0901 ± 0.1274 | 0.9742 ± 0.0365 | 0.0068 | -0.0370 |
| R2 | 0.5336 ± 0.0356 | 0.6851 ± 0.0057 | 0.4744 ± 0.0986 | 0.1166 ± 0.1409 | 0.9507 ± 0.0697 | 0.0083 | -0.0050 |
| R3 | 0.5352 ± 0.0413 | 0.6736 ± 0.0217 | 0.4704 ± 0.0948 | 0.0941 ± 0.1091 | 0.9764 ± 0.0264 | 0.0098 | -0.0165 |
| F3 | 0.5308 ± 0.0351 | 0.6583 ± 0.0021 | 0.4618 ± 0.0826 | 0.0805 ± 0.0899 | 0.9811 ± 0.0197 | 0.0054 | -0.0318 |

| Variant | Seed42 BA/AUROC | Seed43 BA/AUROC |
|---|---:|---:|
| P0 | 0.5508 / 0.6813 | 0.5017 / 0.6119 |
| F0 | 0.5508 / 0.6786 | 0.5000 / 0.7017 |
| F1 | 0.5482 / 0.6769 | 0.5085 / 0.6721 |
| F2 | 0.5554 / 0.6734 | 0.5085 / 0.6615 |
| R1 | 0.5643 / 0.6584 | 0.5000 / 0.6478 |
| R2 | 0.5588 / 0.6811 | 0.5085 / 0.6891 |
| R3 | 0.5645 / 0.6583 | 0.5060 / 0.6889 |
| F3 | 0.5556 / 0.6598 | 0.5060 / 0.6568 |

## 5. NACC→ADNI extension target-test results

Values are mean ± sample SD across seeds 42/43; Δ values are means relative to F0.

| Variant | BA | AUROC | Macro-F1 | Sensitivity | Specificity | Δ BA vs F0 | Δ AUROC vs F0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.6163 ± 0.0480 | 0.6645 ± 0.0501 | 0.5872 ± 0.0704 | 0.8064 ± 0.0526 | 0.4262 ± 0.1486 | 0.1149 | 0.0812 |
| F0 | 0.5014 ± 0.0020 | 0.5833 ± 0.0269 | 0.3581 ± 0.0036 | 0.0028 ± 0.0040 | 1.0000 ± 0.0000 | 0.0000 | 0.0000 |
| F1 | 0.5000 ± 0.0000 | 0.5956 ± 0.0312 | 0.3550 ± 0.0008 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0014 | 0.0123 |
| F2 | 0.5000 ± 0.0000 | 0.6404 ± 0.0234 | 0.3550 ± 0.0008 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0014 | 0.0571 |
| R1 | 0.5045 ± 0.0063 | 0.5904 ± 0.0192 | 0.3838 ± 0.0415 | 0.0357 ± 0.0505 | 0.9732 ± 0.0379 | 0.0031 | 0.0071 |
| R2 | 0.5000 ± 0.0000 | 0.5834 ± 0.0557 | 0.3550 ± 0.0008 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0014 | 0.0001 |
| R3 | 0.5000 ± 0.0000 | 0.6418 ± 0.0537 | 0.3550 ± 0.0008 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | -0.0014 | 0.0585 |
| F3 | 0.5252 ± 0.0318 | 0.6080 ± 0.0447 | 0.4396 ± 0.1104 | 0.1284 ± 0.1739 | 0.9220 ± 0.1103 | 0.0238 | 0.0247 |

| Variant | Seed42 BA/AUROC | Seed43 BA/AUROC |
|---|---:|---:|
| P0 | 0.6502 / 0.6999 | 0.5823 / 0.6291 |
| F0 | 0.5000 / 0.6024 | 0.5028 / 0.5643 |
| F1 | 0.5000 / 0.5735 | 0.5000 / 0.6177 |
| F2 | 0.5000 / 0.6569 | 0.5000 / 0.6239 |
| R1 | 0.5089 / 0.5768 | 0.5000 / 0.6040 |
| R2 | 0.5000 / 0.6229 | 0.5000 / 0.5440 |
| R3 | 0.5000 / 0.6798 | 0.5000 / 0.6038 |
| F3 | 0.5027 / 0.6396 | 0.5477 / 0.5764 |

## 6. Source-test preservation

| Direction | Variant | BA | AUROC | Macro-F1 | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|---:|
| ADNI→to→NACC | P0 | 0.5761 ± 0.0185 | 0.6489 ± 0.0080 | 0.5503 ± 0.0060 | 0.2941 ± 0.0416 | 0.8580 ± 0.0786 |
| ADNI→to→NACC | F0 | 0.5171 ± 0.0241 | 0.6544 ± 0.0178 | 0.4027 ± 0.0714 | 0.0588 ± 0.0832 | 0.9753 ± 0.0349 |
| ADNI→to→NACC | F1 | 0.5218 ± 0.0295 | 0.6691 ± 0.0022 | 0.4445 ± 0.0416 | 0.1176 ± 0.0416 | 0.9259 ± 0.0175 |
| ADNI→to→NACC | F2 | 0.5188 ± 0.0216 | 0.6539 ± 0.0148 | 0.4210 ± 0.0456 | 0.0809 ± 0.0520 | 0.9568 ± 0.0087 |
| ADNI→to→NACC | R1 | 0.5285 ± 0.0062 | 0.6539 ± 0.0320 | 0.4525 ± 0.0115 | 0.1250 ± 0.0312 | 0.9321 ± 0.0436 |
| ADNI→to→NACC | R2 | 0.5340 ± 0.0031 | 0.6735 ± 0.0214 | 0.4663 ± 0.0445 | 0.1544 ± 0.0936 | 0.9136 ± 0.0873 |
| ADNI→to→NACC | R3 | 0.5512 ± 0.0071 | 0.6913 ± 0.0096 | 0.4930 ± 0.0033 | 0.1765 ± 0.0208 | 0.9259 ± 0.0349 |
| ADNI→to→NACC | F3 | 0.5078 ± 0.0027 | 0.6582 ± 0.0096 | 0.4009 ± 0.0123 | 0.0588 ± 0.0208 | 0.9568 ± 0.0262 |
| NACC→to→ADNI | P0 | 0.6613 ± 0.0558 | 0.7725 ± 0.0497 | 0.6019 ± 0.1208 | 0.8111 ± 0.1728 | 0.5115 ± 0.2845 |
| NACC→to→ADNI | F0 | 0.5828 ± 0.1170 | 0.7310 ± 0.0217 | 0.5368 ± 0.1973 | 0.2000 ± 0.2828 | 0.9655 ± 0.0488 |
| NACC→to→ADNI | F1 | 0.5167 ± 0.0236 | 0.7407 ± 0.0126 | 0.4313 ± 0.0481 | 0.0333 ± 0.0471 | 1.0000 ± 0.0000 |
| NACC→to→ADNI | F2 | 0.5858 ± 0.0509 | 0.7700 ± 0.0092 | 0.5599 ± 0.0822 | 0.1889 ± 0.1100 | 0.9828 ± 0.0081 |
| NACC→to→ADNI | R1 | 0.5707 ± 0.1000 | 0.7000 ± 0.0558 | 0.5217 ± 0.1760 | 0.2333 ± 0.3300 | 0.9080 ± 0.1300 |
| NACC→to→ADNI | R2 | 0.5167 ± 0.0236 | 0.6766 ± 0.1030 | 0.4313 ± 0.0481 | 0.0333 ± 0.0471 | 1.0000 ± 0.0000 |
| NACC→to→ADNI | R3 | 0.5056 ± 0.0079 | 0.7678 ± 0.0105 | 0.4090 ± 0.0167 | 0.0111 ± 0.0157 | 1.0000 ± 0.0000 |
| NACC→to→ADNI | F3 | 0.5443 ± 0.0631 | 0.7180 ± 0.0314 | 0.4598 ± 0.0643 | 0.4333 ± 0.5814 | 0.6552 ± 0.4551 |

## 7. Mechanism diagnostics

Mean values across seeds for the primary direction. These are one-epoch screening diagnostics.

| Variant | Domain acc | Intensity acc | Domain AUC | Intensity AUC | Encoder grad norm | Discriminator grad norm | Adv RMS | Full RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 11.1490 | 0.0000 | 1.4112 | 1.4112 |
| F1 | 0.3531 | 0.0000 | 0.1954 | 0.0000 | 11.7518 | 0.0686 | 1.4384 | 1.4384 |
| F2 | 0.0000 | 0.4862 | 0.0000 | 0.4821 | 10.9815 | 0.0506 | 1.3924 | 1.3924 |
| R1 | 0.3714 | 0.0000 | 0.2097 | 0.0000 | 10.1994 | 0.0594 | 1.2826 | 1.3286 |
| R2 | 0.0000 | 0.5097 | 0.0000 | 0.4842 | 11.2901 | 0.0514 | 1.3290 | 1.3755 |
| R3 | 0.3597 | 0.5031 | 0.1704 | 0.4918 | 10.5103 | 0.0760 | 1.3414 | 1.3865 |
| F3 | 0.3653 | 0.4954 | 0.1673 | 0.4816 | 10.5304 | 0.0817 | 1.3754 | 1.3754 |

The diagnostics include non-zero gradient and discriminator fields where applicable, but discriminator accuracy/AUC alone does not prove alignment. The required paired before/after discrepancy and independently checkpoint-bound mechanism evidence remain incomplete.

## 8. Decision assessment

| Criterion | Assessment |
|---|---|
| P0 vs F0 | Raw-Fourier effect is seed- and direction-dependent; F0 does not uniformly improve P0. |
| F1/F2/F3 vs F0 | No uniform GRL benefit; several source/target BA values are near 0.5 under the one-epoch screen. |
| R1/R2/R3 vs F0 | Residual paths improve selected cells but not consistently across seeds and directions. |
| Full vs residual placement | Not established under the screening budget and current mechanism binding. |
| R3 complementarity | Not established; no formal complementarity claim is allowed. |
| Leakage audit | Passed at recorded flag level; target labels/metrics were not used for adaptation or selection. |

Overall decision: **screening complete; formal mechanism claim not supported**. The evidence is sufficient to retain the implementation for controlled follow-up, not to claim validated adaptation or scanner/biology correction.

## 9. Limitations and next actions

1. Re-run the registered matrix with the intended fixed UDA training budget rather than one epoch.
2. Persist per-epoch and selected-checkpoint diagnostics in `audit.json`, including head loss/accuracy/BA/AUROC, encoder and discriminator gradients, GRL coefficient, feature discrepancy, CAPM anchor drift, and checkpoint binding.
3. Add explicit before/after source-target discrepancy and frozen residual/full domain probes.
4. Recompute paired within-seed comparisons against F0 and compare R1/R2/R3 with their full-feature controls.
5. Keep target-test results exploratory and do not select variants or strengths using them.

## 10. Artifact inventory

```text
outputs/ds040_real/{ADNI_to_NACC,NACC_to_ADNI}/seed{42,43}/{P0,F0,F1,F2,F3,R1,R2,R3}/{report.json,audit.json,predictions.json,best.pt}
```

Execution code commit: `e263476`; generated report code SHA-256: `1d42237bbc5094101d5d9a599369a957e501297675081a0f76bcf30c7e156f7a`.
