# dual_shift: Plan 34

This branch contains only the scan-filtered CAPM / IE-CAPM / APIC v3_2
experiment stack defined by Plan 34.

## Active entry points

- `review/plans/34_scan_filtered_capm_execution_plan_2026-08-08.md`
- `docs/IE_CAPM_APIC_V3_2_SCAN_FILTERED_1P5T_PROTOCOL.md`
- `docs/SCAN_FILTERED_MULTI_MACHINE_EXPERIMENT_PLAN.md`
- `journal_dual_shift_scan_filtered_1p5t_mci_ad.yaml`
- `journal_scale_table_scan_filtered_1p5t_mci_ad.yaml`
- `fmm_baseline_scan_filtered_1p5t_mci_ad.yaml`
- `experiments/train_journal.py`

## FMM comparison baseline

The independent raw-volume FMM runner is `experiments/train_fmm_baseline.py`.
It keeps target adaptation label-blind, selects the checkpoint on source
validation, and writes an audit record alongside metrics and predictions.

Run the five registered controls by changing `--variant` among `b0_ref`,
`b1_fmm`, `b1a_no_source_fft`, `b1b_no_attention`, and `b1c_no_grl`. Before
using the scan-filtered data, validate the complete path with:

```powershell
D:\Anaconda\envs\segment\python.exe experiments/train_fmm_baseline.py `
  --config_path fmm_baseline_scan_filtered_1p5t_mci_ad.yaml `
  --direction ADNI_to_NACC --variant b1_fmm --smoke-test `
  --output-dir outputs/fmm_smoke_b1
```

The smoke command is an implementation check, not a performance result.

Historic models, protocols, result archives, and legacy experiment scripts are
preserved on `archive/pre-plan34-cleanup-20260809`.

Distributed Plan 34 runs must use the immutable `plan34-scan-filtered-v2` tag,
not the moving branch head.
