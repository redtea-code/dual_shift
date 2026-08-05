# Environment

- Host: Linux RTX 5090 (`an5bi4acenfa1-0`, CUDA device 5 for layer-2)
- Date: 2026-08-05
- Config: `journal_dual_shift_apic_v3_2_screen_mci_ad_remote.yaml`
- Units: seed42 nacc_to_adni (success); seed42 adni_to_nacc (failure)
- Layer-1: `experiments/summarize_apic_v3_diagnostics.py`
- Layer-2: `experiments/export_apic_v3_checkpoint_diagnostics.py --variant apic_v3_2_x`
