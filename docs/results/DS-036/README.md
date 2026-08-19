# DS-036: Result record

Status: RUNNING

Plan: [DS-036 plan](../../plans/DS-036_TARGET_STYLE_CAPM_UDA.md)

## Current state

The target-style transport implementation and label-blind protocol tests are
merged. No real-data multi-seed result is recorded yet.

## Verified evidence

- `tests/test_target_style_transport.py` and
  `tests/test_target_style_uda_protocol.py`: 4 passed.
- `Model/ablation/target_style_transport.py` compiles in the `segment`
  environment.
- The plan fixes the target contract, primary direction, seeds, matched CAPM
  control, and fresh-holdout promotion rule.

## Not verified

- Five-seed real-data execution.
- Fresh label-blind target holdout performance.
- CAPM-specific improvement over the matched control.

## Next action

Run the registered `ADNI_to_NACC` seeds 42-46 with frozen manifests, then add
per-seed metrics, provenance, and a GO/NO-GO decision here.
