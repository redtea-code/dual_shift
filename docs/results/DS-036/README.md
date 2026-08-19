# DS-036: Result record

Status: COMPLETED

Decision: NO-GO for promotion as a CAPM improvement

Plan: [DS-036 plan](../../plans/DS-036_TARGET_STYLE_CAPM_UDA.md)

## Current state

The target-style transport implementation and label-blind protocol tests are
merged. The registered five-seed paired comparison is complete. It remains an
exploratory benchmark because the internal frozen `T_test` appeared in
historical experiments.

## Verified evidence

- `tests/test_target_style_transport.py` and
  `tests/test_target_style_uda_protocol.py`: 4 passed.
- `Model/ablation/target_style_transport.py` compiles in the `segment`
  environment.
- The plan fixes the target contract, primary direction, seeds, matched CAPM
  control, and fresh-holdout promotion rule.
- [Five-seed comparison report](5SEED_COMPARISON_2026-08-19.md): all 10 paired
  runs completed; target-style CAPM was below matched CAPM on target BA in all
  five seeds.
- Target BA: CAPM `0.6663 +/- 0.0067`; target-style CAPM `0.6415 +/- 0.0214`;
  paired mean difference `-0.0248`.

## Not verified

- Fresh label-blind target holdout performance.
- CAPM-specific improvement over the matched control.
- Full manifest, environment, resolved-config, and source-commit provenance
  archive required by the main-branch E0/E2 standard.

## Next action

Do not promote the current transport configuration to a fresh confirmatory
holdout. Follow-up work is registered separately as DS-037 (amplitude
transport mechanism audit) and DS-038 (domain/intensity GRL factorial audit).
The protocol extension that uses all target data is intentionally deferred.
