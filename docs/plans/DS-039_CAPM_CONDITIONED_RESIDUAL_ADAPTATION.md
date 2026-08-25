# DS-039: CAPM-conditioned biology-preserving residual adaptation pilot

Owner: cyh

Status: RUNNING

Result record: `docs/results/DS-039/README.md`

## Question

DS-038 does not establish that the domain and intensity GRL modules are
complementary. Their observed gains are task-specific, and the mechanism
diagnostics are not sufficient to justify unconditional alignment of the
complete visual representation. DS-039 asks a narrower question:

> Can batch-sensitive variation be reduced after CAPM conditioning, while the
> CAPM-adjusted diagnostic-support representation remains usable?

This is a frozen-checkpoint mechanism pilot, not a new multi-seed performance
claim and not a scanner-causality experiment.

## Motivation and hypothesis

AD MRI biology and cohort/batch effects can be coupled. Applying domain
alignment to every feature may remove both nuisance variation and diagnostic
structure. The three-variable CAPM branch (`age`, `sex`, `education`) is used
as a diagnostic-support anchor, without calling it a pure biological or causal
representation.

For a layer-4 spatial feature map `F4`, the pilot uses:

```text
B4 = CAPM(F4, z_ref)
B4' = B4 + alpha * m * (
        (B4 - mu_T) / sigma_T * sigma_S + mu_S - B4
      )
```

`mu_S/sigma_S` and `mu_T/sigma_T` are computed from source images and an
image-only, subject-disjoint `T_adapt` population. `m` is a normalized
channel discrepancy in `[0, 1]`; `alpha` has a fixed upper bound of `0.25` in
the initial pilot. A zero discrepancy or zero strength is an identity path.

The statistics builder uses a fixed source-only reference table `z_ref`, so
`T_adapt` does not materialize target labels, covariates, environments,
predictions, or metrics. Final target reporting may use the ordinary CAPM
covariates required by the frozen model; this distinction must remain explicit
in the audit.

## Locked protocol

- Direction: `ADNI_to_NACC` only for the primary pilot.
- Task: MCI vs AD, scan-filtered protocol.
- Backbone: matched `original_capm` with `layer4_pixel`.
- Source statistics: earliest scan per source subject from `S_train`.
- Target statistics: earliest scan per subject from unlabeled `T_adapt`.
- Target split: subject-disjoint `T_adapt` and internal frozen `T_test`.
- Checkpoint: frozen source checkpoint; no retraining in the first pilot.
- Selection: source-validation checkpoint only; no target-test tuning.
- Strength: fixed inference bound, not selected on `T_test`.
- Target-test status: exploratory because the holdout has appeared in historical
  work.

## Registered cells

| ID | Cell | Residual correction | Role |
|---|---|---:|---|
| RA0 | `capm_control` | disabled | matched original-CAPM control |
| RA1 | `capm_residual` | enabled, `max_strength=0.25` | proposed CAPM-conditioned residual adaptation |
| RA-D | `source_adapted_diagnostic` | enabled on source test | preservation/anchor-drift diagnostic; never a selector |

The first implementation is channel-wise and post-CAPM at `layer4_pixel`. It
does not claim low-rank biology projection, cross-scale protection, FFT
adaptation, or a learned discriminator. Those are deferred until this narrow
functional division has evidence.

## Required artifacts and audits

The stats stage must save:

- the target split JSON and subject digests;
- source checkpoint and source-split hashes;
- source/target channel moments and discrepancy vector;
- `target_labels_read=false` and `target_metrics_read=false`;
- the fixed source-only reference table and resolved strength bound.

The evaluation stage must save:

- source control, source-adapted diagnostic, target control, and target-adapted
  metrics;
- residual RMS, correction RMS, effective strength, identity loss, and finite
  checks;
- source diagnosis preservation and CAPM-anchor drift;
- sensitivity and specificity, so a one-sided threshold shift is not mistaken
  for adaptation success;
- checkpoint, configuration, split, and statistics provenance.

## Decision rule

The pilot is promising only if all of the following hold on the pre-registered
comparison:

1. target BA/AUC is not lower than the matched CAPM control;
2. source/target residual discrepancy decreases;
3. source diagnosis performance and CAPM-anchor behavior remain within the
   pre-set tolerance;
4. correction magnitude is finite and bounded; and
5. any BA gain is not explained only by sensitivity/specificity movement.

A positive target score without residual and preservation evidence is
insufficient. A discrepancy reduction with source degradation is negative
transfer. No result supports scanner, manufacturer, field-strength, or causal
biology language without separate probes and paired evidence.

## Implementation location

The minimal implementation is included in this branch:

- `Model/ablation/residual_adaptation.py`
- `experiments/build_capm_residual_stats.py`
- `experiments/run_capm_residual_pilot.py`
- `tests/test_capm_residual_adaptation.py`

The implementation is a frozen-checkpoint pilot only. It must not be marked
`COMPLETED` until a real ADNI/NACC run, output hashes, and the required
preservation/discrepancy audits are archived.
