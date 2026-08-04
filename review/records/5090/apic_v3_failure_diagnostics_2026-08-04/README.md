# APIC v3 failure diagnostics archive (5090 · MCI vs AD)

See companion write-up:

`../25_apic_v3_failure_diagnostics_mci_ad_2026-08-04.md`

Layout:

- `history/` — layer-1 `summarize_apic_v3_diagnostics.py` outputs
- `checkpoint/<seed>_<direction>/` — layer-2 `diagnostic_summary.json` + `sample_diagnostics.csv`
- `artifact_sha256.json` — config / manifest / checkpoint hashes (checkpoints not committed)
- `ENVIRONMENT.md` — host / torch / git
