# APIC v3 CN failure diagnostics archive (3090)

- Spec: `review/operations/24_apic_v3_failure_diagnostics_2026-08-04.md`
- Write-up: `../25_apic_v3_cn_failure_diagnosis_2026-08-04.md`
- Task: **CN_vs_AD only** (MCI lives under `review/records/5090/`)

## Layer-2 coverage (complete)

| Job | Role | Files |
|---|---|---|
| `checkpoint/seed43_nacc_to_adni/` | worst ΔBA failure | `diagnostic_summary.json`, `sample_diagnostics.csv` |
| `checkpoint/seed43_adni_to_nacc/` | sole dual-baseline win | same |
| `checkpoint/seed42_adni_to_nacc/` | seed42 A→N | same |
| `checkpoint/seed42_nacc_to_adni/` | seed42 N→A | same |

All four jobs ran **full four splits** (no `--max-samples` after smoke).

Also included:

- `history/` — layer-1 summarize outputs
- `layer2_*_stdout.txt` — exporter stdout / exit records
- `artifact_sha256.json` — hashes for diagnostics files + checkpoint/config metadata
- `ENVIRONMENT.md`
