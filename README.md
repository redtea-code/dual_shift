# dual_shift: Plan 34

This branch contains only the scan-filtered CAPM / IE-CAPM / APIC v3_2
experiment stack defined by Plan 34.

## Documentation workflow

Every new experiment gets an ID and one plan plus one result record:

- `docs/README.md` explains the documentation folders.
- `docs/EXPERIMENT_INDEX.md` is the status registry.
- `docs/plans/<ID>.md` records the question, protocol, controls, and stop rules.
- `docs/results/<ID>/README.md` records the command, evidence, result, and
  claim boundary; detailed reports live beside it.
- Use the templates in `docs/templates/` for new records.

Do not call an experiment complete without a result record. Smoke tests are
implementation evidence, not performance results.

## Experiment branch policy

Plans, reports, maps, and other research text are maintained on `main`.
New `exp/<id>-<short-name>` branches should contain only the model and any
necessary training-code changes. See
[`docs/EXPERIMENT_BRANCH_POLICY.md`](docs/EXPERIMENT_BRANCH_POLICY.md).

## Active entry points

- `docs/plans/DS-034.md`
- `docs/protocols/DS-034_SCAN_FILTERED_DATA_PROTOCOL.md`
- `docs/plans/DS-034_MULTI_MACHINE_EXECUTION.md`
- `docs/plans/DS-035_FMM_BASELINE.md`
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
python experiments/train_fmm_baseline.py `
  --config_path fmm_baseline_scan_filtered_1p5t_mci_ad.yaml `
  --direction ADNI_to_NACC --variant b1_fmm --smoke-test `
  --output-dir outputs/fmm_smoke_b1
```

The smoke command is an implementation check, not a performance result.

Historic models, protocols, result archives, and legacy experiment scripts are
preserved on `archive/pre-plan34-cleanup-20260809`.

Distributed Plan 34 runs must use the immutable `plan34-scan-filtered-v2` tag,
not the moving branch head.
