# DS-038：MCI–AD 与 NC–MCI 双任务双方向两 Seed 泛化审计

- 日期：2026-08-25
- 状态：`INTERIM / EXPLORATORY`
- 设计：2 tasks × 2 directions × 4 variants × seeds42/43 = **32 cells**
- 变体：G0 no-GRL、G1 domain-only、G2 intensity-only、G3 both-GRL
- 指标：BA / AUROC / Macro-F1 / Sensitivity / Specificity；均为 mean ± sample SD。
- 口径：Source-test 检查源域疾病判别保持；Target-test 评估冻结 source-validation checkpoint 后的跨域泛化。

MCI–AD ADNI→NACC 的原始注册协议仍为 G0–G3 × seeds42–46 = 20 cells。本报告将其统一抽取为 seeds42/43 的 8-cell 子集，不能宣称完成原始 20-cell factorial。所有 target holdout 均为 exploratory。

## 1. Artifact 状态

| 任务 | 方向 | cells | 状态 |
|---|---|---:|---|
| MCI–AD | ADNI→NACC | 8/8 ✓ | seed42/43 子集；原注册 20 cells 未完成 |
| MCI–AD | NACC→ADNI | 8/8 ✓ | 协议外方向扩展 |
| NC–MCI | ADNI→NACC | 8/8 ✓ | seed43 G2/G3 已完成，best epoch=24/21 |
| NC–MCI | NACC→ADNI | 8/8 ✓ | 协议外方向扩展 |

四组 audit 均记录 target labels 未用于训练或 checkpoint selection，并包含 best checkpoint、mechanism diagnostics 与 frozen-feature probe 字段。

## 2. 五项指标

各表从 `predictions.json` 按 subject 聚合重算；同一 subject 的多次扫描概率取均值，阈值为 `probability >= 0.5`。

### 2.1 MCI–AD：ADNI→NACC

| Variant | Source BA | Source AUROC | Source Macro-F1 | Source Sens. | Source Spec. | Target BA | Target AUROC | Target Macro-F1 | Target Sens. | Target Spec. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | 0.5821±0.0551 | 0.6422±0.0718 | 0.5551±0.0190 | 0.4931±0.1317 | 0.6711±0.2419 | 0.5834±0.0352 | 0.6774±0.1085 | 0.5677±0.0551 | 0.2124±0.0779 | 0.9545±0.0075 |
| G1 | 0.6405±0.0524 | 0.7482±0.0855 | 0.6374±0.0461 | 0.6943±0.0553 | 0.5868±0.1600 | 0.6568±0.0394 | 0.8113±0.0019 | 0.6626±0.0481 | 0.3788±0.1385 | 0.9348±0.0596 |
| G2 | 0.6013±0.0006 | 0.6989±0.0332 | 0.5809±0.0133 | 0.5092±0.0601 | 0.6934±0.0614 | 0.5440±0.0126 | 0.7664±0.0375 | 0.4997±0.0283 | 0.1137±0.0457 | 0.9742±0.0205 |
| G3 | 0.5882±0.0044 | 0.6814±0.0506 | 0.5611±0.0146 | 0.4580±0.0349 | 0.7184±0.0261 | 0.5487±0.0053 | 0.7211±0.0075 | 0.4995±0.0162 | 0.1031±0.0187 | 0.9944±0.0080 |

G1 is the clearest target screening signal in this two-seed subset, but the original 20-cell protocol remains incomplete. G2/G3 are not supported.

### 2.2 MCI–AD：NACC→ADNI

| Variant | Source BA | Source AUROC | Source Macro-F1 | Source Sens. | Source Spec. | Target BA | Target AUROC | Target Macro-F1 | Target Sens. | Target Spec. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | 0.8261±0.0649 | 0.9170±0.0146 | 0.8130±0.0576 | 0.8462±0.1088 | 0.8060±0.0211 | 0.6113±0.0319 | 0.6559±0.0315 | 0.5957±0.0464 | 0.8281±0.1865 | 0.3944±0.2503 |
| G1 | 0.8165±0.0468 | 0.8955±0.0162 | 0.7965±0.0420 | 0.8718±0.0725 | 0.7612±0.0211 | 0.6184±0.0550 | 0.6792±0.0603 | 0.6128±0.0745 | 0.9030±0.0240 | 0.3339±0.1340 |
| G2 | 0.8208±0.0226 | 0.8988±0.0116 | 0.7938±0.0314 | 0.9103±0.0181 | 0.7313±0.0633 | 0.6271±0.0911 | 0.6780±0.1025 | 0.6075±0.1270 | 0.9044±0.0975 | 0.3499±0.2796 |
| G3 | 0.8180±0.0127 | 0.9060±0.0106 | 0.8108±0.0327 | 0.8077±0.0907 | 0.8284±0.1161 | 0.6221±0.0340 | 0.6455±0.0631 | 0.6251±0.0394 | 0.8690±0.0599 | 0.3753±0.0081 |

GRL variants trade Specificity for Sensitivity; this protocol-extension does not support adoption.

### 2.3 NC–MCI：ADNI→NACC

| Variant | Source BA | Source AUROC | Source Macro-F1 | Source Sens. | Source Spec. | Target BA | Target AUROC | Target Macro-F1 | Target Sens. | Target Spec. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | 0.6734±0.0505 | 0.7091±0.0600 | 0.6643±0.0440 | 0.6562±0.0000 | 0.6905±0.1010 | 0.5538±0.0188 | 0.6118±0.0218 | 0.5461±0.0280 | 0.2215±0.0382 | 0.8862±0.0005 |
| G1 | 0.6403±0.0395 | 0.7046±0.0179 | 0.6411±0.0435 | 0.8281±0.0221 | 0.4524±0.1010 | 0.5738±0.0161 | 0.6163±0.0062 | 0.5651±0.0067 | 0.4861±0.0668 | 0.6615±0.0346 |
| G2 | 0.6328±0.0616 | 0.7024±0.0379 | 0.6343±0.0629 | 0.7656±0.0221 | 0.5000±0.1010 | 0.5725±0.0081 | 0.6126±0.0100 | 0.5739±0.0092 | 0.3848±0.0214 | 0.7602±0.0377 |
| G3 | 0.6488±0.0400 | 0.7321±0.0105 | 0.6466±0.0356 | 0.7500±0.0884 | 0.5476±0.1684 | 0.5715±0.0120 | 0.6104±0.0231 | 0.5725±0.0139 | 0.3053±0.0632 | 0.8378±0.0392 |

All four variants are now two-seed complete. Target BA rises for G1–G3, but source-test degradation and Sensitivity/Specificity trade-offs weaken the mechanism interpretation.

### 2.4 NC–MCI：NACC→ADNI

| Variant | Source BA | Source AUROC | Source Macro-F1 | Source Sens. | Source Spec. | Target BA | Target AUROC | Target Macro-F1 | Target Sens. | Target Spec. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | 0.6055±0.0051 | 0.6775±0.0245 | 0.5963±0.0138 | 0.5000±0.1616 | 0.7110±0.1515 | 0.5199±0.0281 | 0.7917±0.0711 | 0.4278±0.0815 | 0.9943±0.0081 | 0.0455±0.0643 |
| G1 | 0.5945±0.0161 | 0.6505±0.0032 | 0.5760±0.0095 | 0.5786±0.0505 | 0.6104±0.0184 | 0.5299±0.0101 | 0.7729±0.0182 | 0.4463±0.0001 | 1.0000±0.0000 | 0.0598±0.0202 |
| G2 | 0.6383±0.0551 | 0.7019±0.0577 | 0.6189±0.0854 | 0.6143±0.1010 | 0.6623±0.2112 | 0.5114±0.0161 | 0.7826±0.0595 | 0.4082±0.0538 | 1.0000±0.0000 | 0.0227±0.0321 |
| G3 | 0.6205±0.0455 | 0.6740±0.0020 | 0.5771±0.0965 | 0.6857±0.1616 | 0.5552±0.2525 | 0.5197±0.0279 | 0.7856±0.0454 | 0.4377±0.0955 | 0.9713±0.0406 | 0.0682±0.0964 |

Target predictions are close to all-positive behavior (Sensitivity near 1, Specificity near 0); this is not useful balanced generalization.

## 3. Unified conclusion

1. MCI–AD ADNI→NACC G1 is a screening signal only, not a completed 20-cell factorial claim.
2. NC–MCI ADNI→NACC is now 8/8 complete, but BA gains are accompanied by source-test and class trade-offs.
3. Both NACC→ADNI directions are extensions and remain separate from ADNI→NACC.
4. G3 is not consistently better than G1/G2; GRL-head complementarity is not established.
5. No task/direction average is reported; no scanner/manufacturer/field-strength causal claim is supported.

## 4. Evidence paths

- `outputs/ds038_diagnostic_rerun/`
- `outputs/ds038_mci_ad_nacc_to_adni_2seed/`
- `outputs/ds038_nc_mci_2seed/`
- `outputs/ds038_nc_mci_nacc_to_adni_2seed/`
- `docs/EXPERIMENT_LEDGER_DS034_DS038_2026-08-24.md`
