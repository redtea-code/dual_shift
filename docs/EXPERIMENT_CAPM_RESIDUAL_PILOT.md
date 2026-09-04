# CAPM-conditioned residual adaptation pilot

## Motivation

DS-038 does not justify combining the two existing GRL modules: their gains
are task-specific and the current record is still `BLOCKED` because the
mechanism diagnostics were not persisted.  The next experiment therefore
changes the role of the representation instead of adding another adversary.

The working premise is that the three-variable CAPM branch (`age`, `sex`, and
`education`) is a diagnostic-support anchor, while batch effects can remain in
the CAPM-adjusted spatial representation.  The pilot evaluates CAPM with one
fixed source-only reference table, estimates source versus unlabeled
target-adaptation channel moments, and moves only a bounded,
discrepancy-weighted residual toward the source moments:

```text
F4 -> original CAPM(F4, z_ref) -> B4
   -> R(B4; S_train, T_adapt) -> classifier
```

The minimal builder uses one frozen source-only `z_ref` while constructing
the source/`T_adapt` moments, so the adaptation artifact itself remains
covariate-blind.  Final target reporting still calls the ordinary CAPM model
with the target row's three permitted covariates; this is not a claim that
demographic and batch shifts are already separated.  A later conditional
residual version must stratify or regress the discrepancy on `z` without
reading target labels.

This is intentionally the smallest test of the proposed functional division.
It has no target labels, no target covariates, no discriminator, no FFT, no
retraining, and no target-test model selection.  It is not yet evidence that
CAPM features are causal biology or that scanner effects have been removed.

## Files

- `Model/ablation/residual_adaptation.py`: moment schema, bounded adapter,
  CAPM-then-residual model wrapper, and image-only statistics builder.
- `experiments/build_capm_residual_stats.py`: creates a subject-disjoint
  `T_adapt/T_test` split and a JSON statistics artifact from a frozen
  `original_capm` checkpoint.
- `experiments/run_capm_residual_pilot.py`: evaluates source control, source
  preservation diagnostic, target control, and target-adapted cells.
- `tests/test_capm_residual_adaptation.py`: schema, bounded correction,
  feature-count, model-interface, and label-blind loader checks.

## Minimal run

Run from the repository root with the `segment` environment.  The checkpoint
must be an exact `original_capm` model using
`scale_table_ablation.preset: layer4_pixel` and the same source split as the
run.

The optional YAML section is intentionally one scalar:

```yaml
capm_residual: {max_strength: 0.25}
```

It is a fixed inference bound, not a target-test tuning parameter.

```powershell
conda run -n segment python experiments/build_capm_residual_stats.py `
  --config journal_scale_table_scan_filtered_1p5t_mci_ad.yaml `
  --direction ADNI_to_NACC `
  --source-split outputs/.../frozen_split.json `
  --checkpoint outputs/.../best_checkpoint.pt `
  --target-split outputs/capm_residual/ADNI_to_NACC_target_split.json `
  --output outputs/capm_residual/ADNI_to_NACC_stats.json `
  --adaptation-fraction 0.5 --seed 43 --batch-size 2

conda run -n segment python experiments/run_capm_residual_pilot.py `
  --config journal_scale_table_scan_filtered_1p5t_mci_ad.yaml `
  --direction ADNI_to_NACC `
  --source-split outputs/.../frozen_split.json `
  --target-split outputs/capm_residual/ADNI_to_NACC_target_split.json `
  --checkpoint outputs/.../best_checkpoint.pt `
  --stats outputs/capm_residual/ADNI_to_NACC_stats.json `
  --output outputs/capm_residual/ADNI_to_NACC_report.json
```

The first command never materializes target labels.  The second command reads
labels only from the disjoint `T_test` subjects for the final report.  The
pilot is a mechanism probe: promote it only after source-validation selection,
paired seed comparisons, residual diagnostics, and the existing batch/biology
preservation checks are all available.
