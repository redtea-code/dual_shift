# Environment

- Host: Windows RTX 3090
- Date: 2026-08-05
- Config: `configs/journal_dual_shift_apic_v3_2_screen_cn_ad.yaml`
- Units: seed42 adni_to_nacc (success); seed43 nacc_to_adni (failure)
- Layer-1: `experiments/summarize_apic_v3_diagnostics.py`
- Layer-2: `experiments/export_apic_v3_checkpoint_diagnostics.py --variant apic_v3_2_x`
