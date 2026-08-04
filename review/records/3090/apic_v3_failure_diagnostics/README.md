# APIC v3 CN failure diagnostics (3090)

- Host: local Windows RTX 3090
- Date: 2026-08-04
- Spec: `review/operations/24_apic_v3_failure_diagnostics_2026-08-04.md`
- Write-up: `../25_apic_v3_cn_failure_diagnosis_2026-08-04.md`

Layout:

- `history/` — layer-1 summarize outputs for CN_vs_AD s1
- `checkpoint_seed43_nacc_to_adni/` — worst-failure unit full diagnostics
- `checkpoint_seed43_adni_to_nacc/` — sole dual-baseline-win unit full diagnostics
- `stdout/` — script stdout / exit records

Checkpoints themselves are not stored in git; see SHA-256 in the write-up and each `diagnostic_summary.json`.
