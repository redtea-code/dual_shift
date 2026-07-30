# Dual-Shift Postfix Gate Results — 2026-07-25

Queue `20260724_144141` completed OK (~13.2 h).  
Artifacts: `outputs/journal/dual_shift_postfix/gate_report.json`, `stage2_metrics_table.csv`

## Target AUC / F1 (seed 42, post P0/P1)

| variant | ADNI→NACC AUC | F1 | NACC→ADNI AUC | F1 |
|---|---:|---:|---:|---:|
| ce_only | 0.9228 | 0.8077 | 0.8881 | 0.7663 |
| mixstyle | 0.9188 | 0.8105 | 0.9150 | 0.8310 |
| apis_only | 0.9239 | 0.7770 | **0.9398** | 0.8169 |
| cdt_only | 0.9094 | 0.8099 | 0.9125 | 0.7818 |
| dual_shift | **0.9264** | 0.7872 | 0.9133 | 0.8274 |

## Gate verdict

| gate | ADNI→NACC | NACC→ADNI |
|---|---|---|
| APIS (AUC non-inf / F1↑ / no collapse) | pass / fail / pass | **pass / pass / pass** |
| CDT (AUC non-inf / cal OK) | **fail** / fail | pass / pass |
| Joint (≥2 of AUC/F1/Brier) | **1/3 fail** | **2/3 pass** |

## Readout vs exploratory Stage-2

- Reverse **APIS recovered strongly** (0.940 vs prior 0.901 failure) after epoch-freeze / min_subjects / distance / schedule fixes.
- Forward CDT weakened vs CE; joint no longer clears ADNI→NACC.
- **Bilateral joint Go still not cleared**; APIS is the clearest improved signal under the repaired protocol.
