# DS-039: Result record

Status: **RUNNING**

Plan: [DS-039 plan](../../plans/DS-039_CAPM_CONDITIONED_RESIDUAL_ADAPTATION.md)

## 1. Current evidence state

DS-039 currently has a code-level pilot and no real ADNI/NACC performance
result. The implementation is in the companion checkout
`D:\ADNI\dual_shift_plan34_main`; the authoritative maintenance checkout has
not yet received an implementation commit or a dataset run artifact.

Therefore this record must not be read as evidence that the proposed module
improves target performance. Its current status is `RUNNING`, not
`COMPLETED`.

## 2. Implemented pilot

The pilot evaluates the following path:

```text
F4 -> original CAPM(F4, z_ref) -> B4
   -> bounded channel residual correction -> classifier
```

The correction matches source and image-only `T_adapt` channel moments using a
discrepancy-weighted affine residual. It is frozen, has no learned parameters,
and defaults to `max_strength=0.25`. `layer4_pixel` is fixed so that changing
feature size is not presented as the novelty; the functional division is the
CAPM diagnostic-support anchor followed by residual batch correction.

Implemented entry points in the companion checkout:

- `Model/ablation/residual_adaptation.py`
- `experiments/build_capm_residual_stats.py`
- `experiments/run_capm_residual_pilot.py`
- `tests/test_capm_residual_adaptation.py`

## 3. Verified code evidence

The following checks passed in the `segment` environment:

- new pilot tests: **7 passed**;
- existing scale-table and frequency-UDA tests: **12 passed**;
- `py_compile` for the model and both experiment scripts;
- synthetic forward and base-checkpoint state loading.

These checks establish interface and protocol behavior only. They do not
establish a target-domain gain, residual discrepancy reduction on ADNI/NACC,
or biology preservation.

## 4. Target-data contract

The statistics builder creates a subject-disjoint target split and exposes only
`image` and `subject_id` to the `T_adapt` loader. It records:

- `target_labels_read=false`;
- `target_metrics_read=false`;
- no target covariate/environment construction in the statistics stage;
- source checkpoint, source split, target split, and statistics provenance.

The final evaluation command reads labels only for the disjoint `T_test` report.
The ordinary CAPM target covariates used at final inference must be reported
separately from the label-free adaptation stage.

## 5. Not yet verified

- no real checkpoint has been evaluated by the pilot in this record;
- no source/target residual discrepancy or CAPM-anchor drift is available;
- no source-preservation or sensitivity/specificity audit is available;
- no seed comparison or target performance conclusion is available;
- no implementation commit has been attached to the authoritative checkout.

## 6. Required next action

Run the builder and evaluator on one matched `layer4_pixel` original-CAPM
checkpoint for `ADNI_to_NACC`, archive the split/statistics/report hashes, and
then update this record with RA0/RA1/RA-D evidence. Only after the source and
target audits pass should a second seed or a cross-scale/low-rank extension be
considered.

Until those artifacts exist, DS-039 remains **RUNNING** and the proposal is
not a validated BioCAPM-DA result.
