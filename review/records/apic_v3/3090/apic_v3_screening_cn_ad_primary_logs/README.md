# APIC v3 CN vs AD primary run logs (3090)

- Host: local Windows RTX 3090
- Date: 2026-08-03/04
- Source: `outputs/journal/apic_v3_screening_cn_ad/s1/logs/primary/`
- Matrix: seeds 42/43 × both directions × `{ce_x, mixstyle_x, apic_v3_x}` = 12/12
- Companion tables: `23_apic_v3_cn_ad_s1_metrics_2026-08-04.md`
- Execution plan: `review/plans/22_apic_v3_s1_3090_cn_ad_execution_plan_2026-08-03.md`

Files:

- `seed{42,43}_{adni_to_nacc,nacc_to_adni}.log` — per-job training logs from the screening launcher
- Note: parent PowerShell redirect of the launcher stdout was empty (`launcher.out` omitted); job-level logs above are the authoritative run records
