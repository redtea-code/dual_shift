# APIC v3 failure diagnostics archive (5090 · MCI vs AD)

See companion write-up:

`../25_apic_v3_failure_diagnostics_mci_ad_2026-08-04.md`

Layout:

- `history/` — layer-1 `summarize_apic_v3_diagnostics.py` outputs
- `checkpoint/<seed>_<direction>/` — layer-2 for all 4 MCI cells (`seed{42,43}_{adni_to_nacc,nacc_to_adni}`)
- `artifact_sha256.json` — config / manifest / checkpoint hashes (checkpoints not committed)
- `ENVIRONMENT.md` — host / torch / git
