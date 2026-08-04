# APIC v3 CN_vs_AD S1 主表（本机 3090）

**截稿：** 2026-08-04  
**代码：** `40c7371` · 分支 `run/apic-v3-s1-cn-3090` · **protocol_revision：** 3  
**矩阵：** CN_vs_AD × 2 directions × seeds {42,43} × {ce_x, mixstyle_x, apic_v3_x} = **12/12**  
**机读：** `apic_v3_cn_ad_s1_metrics_main_table.csv`  
**计划：** `review/plans/22_apic_v3_s1_3090_cn_ad_execution_plan_2026-08-03.md`  

说明：target · subject-mean；本表仅 CN 份额，**不构成**完整 APIC-V3-S1 Gate。

## 0. 主终点速览（balanced_accuracy）

| seed | direction | ce_x | mixstyle_x | apic_v3_x | Δ vs ce | Δ vs mix |
|---:|---|---:|---:|---:|---:|---:|
| 42 | adni_to_nacc | 0.8027 | 0.8210 | 0.7908 | −0.0119 | −0.0302 |
| 43 | adni_to_nacc | 0.6636 | 0.7615 | 0.7862 | +0.1226 | +0.0247 |
| 42 | nacc_to_adni | 0.7627 | 0.7588 | 0.7302 | −0.0325 | −0.0287 |
| 43 | nacc_to_adni | 0.7976 | 0.7269 | 0.6233 | −0.1744 | −0.1037 |
| **mean** | adni_to_nacc | 0.7332 | 0.7912 | 0.7885 | +0.0553 | −0.0028 |
| **mean** | nacc_to_adni | 0.7802 | 0.7429 | 0.6767 | −0.1034 | −0.0662 |

- 同时胜两基线：**1/4**（仅 seed43 · ADNI→NACC）
- 两 seed 平均对两基线均为正：**0/2** direction

## 1. Target · subject-mean · `adni_to_nacc`

| seed | variant | balanced_accuracy | auc | macro_f1 | sensitivity | specificity | accuracy | brier | ece | n_subjects | BA_CI_lo | BA_CI_hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | ce_x | 0.8027 | 0.9051 | 0.7991 | 0.6882 | 0.9173 | 0.8729 | 0.0982 | 0.0521 | 960 | 0.7734 | 0.8404 |
| 42 | mixstyle_x | 0.8210 | 0.9104 | 0.7904 | 0.7634 | 0.8786 | 0.8562 | 0.1169 | 0.0913 | 960 | 0.7910 | 0.8511 |
| 42 | apic_v3_x | 0.7908 | 0.9057 | 0.7855 | 0.6720 | 0.9096 | 0.8635 | 0.1004 | 0.0526 | 960 | 0.7637 | 0.8179 |
| 43 | ce_x | 0.6636 | 0.9172 | 0.7026 | 0.3441 | 0.9832 | 0.8594 | 0.1059 | 0.0814 | 960 | 0.6302 | 0.7005 |
| 43 | mixstyle_x | 0.7615 | 0.8548 | 0.7547 | 0.6290 | 0.8941 | 0.8427 | 0.1191 | 0.0198 | 960 | 0.7228 | 0.7967 |
| 43 | apic_v3_x | 0.7862 | 0.8968 | 0.7982 | 0.6344 | 0.9380 | 0.8792 | 0.1015 | 0.0832 | 960 | 0.7534 | 0.8226 |

## 2. Target · subject-mean · `nacc_to_adni`

| seed | variant | balanced_accuracy | auc | macro_f1 | sensitivity | specificity | accuracy | brier | ece | n_subjects | BA_CI_lo | BA_CI_hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | ce_x | 0.7627 | 0.8767 | 0.7397 | 0.6379 | 0.8875 | 0.7398 | 0.1852 | 0.1201 | 196 | 0.7056 | 0.8054 |
| 42 | mixstyle_x | 0.7588 | 0.8709 | 0.7395 | 0.6552 | 0.8625 | 0.7398 | 0.1806 | 0.1287 | 196 | 0.6903 | 0.8016 |
| 42 | apic_v3_x | 0.7302 | 0.8950 | 0.6986 | 0.5603 | 0.9000 | 0.6990 | 0.2228 | 0.2034 | 196 | 0.6730 | 0.7748 |
| 43 | ce_x | 0.7976 | 0.9033 | 0.7843 | 0.7328 | 0.8625 | 0.7857 | 0.1496 | 0.1021 | 196 | 0.7468 | 0.8442 |
| 43 | mixstyle_x | 0.7269 | 0.9055 | 0.6803 | 0.4914 | 0.9625 | 0.6837 | 0.2585 | 0.2365 | 196 | 0.6775 | 0.7745 |
| 43 | apic_v3_x | 0.6233 | 0.7795 | 0.5763 | 0.3966 | 0.8500 | 0.5816 | 0.2580 | 0.1921 | 196 | 0.5676 | 0.6741 |

## 3. ΔBA：apic_v3_x − baselines

| seed | direction | BA_apic | BA_ce | BA_mix | Δ vs ce | Δ vs mix | win_both |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | adni_to_nacc | 0.7908 | 0.8027 | 0.8210 | -0.0119 | -0.0302 |  |
| 43 | adni_to_nacc | 0.7862 | 0.6636 | 0.7615 | +0.1226 | +0.0247 | Y |
| 42 | nacc_to_adni | 0.7302 | 0.7627 | 0.7588 | -0.0325 | -0.0287 |  |
| 43 | nacc_to_adni | 0.6233 | 0.7976 | 0.7269 | -0.1744 | -0.1037 |  |

## 4. 两 seed 平均 ΔBA（按 direction）

| direction | mean Δ vs ce | mean Δ vs mix | both_positive |
| --- | --- | --- | --- |
| adni_to_nacc | +0.0553 | -0.0028 |  |
| nacc_to_adni | -0.1034 | -0.0662 |  |

- seed×direction 同时胜两基线：**1/4**
- direction 两 seed 平均对两基线均为正：**0/2**（完整 Gate 还需 MCI 的另外 2 个 direction）

## 5. apic_v3_x 审计摘要

| seed | direction | inference_path | valid_slots | last valid_intervention_frac |
| --- | --- | --- | --- | --- |
| 42 | adni_to_nacc | clean | 8 | 1.0000 |
| 42 | nacc_to_adni | clean | 8 | 1.0000 |
| 43 | adni_to_nacc | clean | 8 | 1.0000 |
| 43 | nacc_to_adni | clean | 8 | 1.0000 |

