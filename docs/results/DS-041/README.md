# DS-041: CAPM-conditioned source-free residual distribution alignment

Status: **PROPOSED / CODE READY / NO REAL-DATA RESULTS**

Plan: [DS-041 plan](../../plans/DS-041_CAPM_CONDITIONED_SOURCE_FREE_RESIDUAL_ALIGNMENT.md)

The DS-041 implementation is present and has passed the synthetic/interface
and focused regression gates. No real-data run, checkpoint, or target metric
is claimed yet.

Verified locally with the `segment` environment:

- `27 passed` across the DS-041 distribution-alignment tests and the related
  DS-039/DS-040 transport/GRL tests;
- both DS-041 command-line entry points import, compile, and expose `--help`;
- full repository collection remains blocked by the pre-existing missing
  `experiments.analyze_frequency_domain` module referenced by
  `tests/test_image_only_feature_frequency.py`.

The synthetic gate covers deterministic K=1/K=2 statistics, subject-level
duplicate protection, zero-strength identity, finite outputs, residual
task-support preservation, label-blind loader rejection, split disjointness,
checkpoint-compatible loading, and perturbation-audit interfaces. These are
interface/mechanism checks, not ADNI/NACC evidence.

The real-data order is fixed: create the frozen `original_capm` checkpoint and
source projector, run `build_capm_residual_distribution_stats.py` once per
direction/seed, then run `run_capm_residual_distribution_pilot.py`. The builder
never exposes target labels or covariates to the adaptation loader; the pilot
reads target labels only for the final exploratory report. The statistics path
uses the source-training mean CAPM table; ordinary target covariates are used
only in the explicitly marked final target-report cells.

The plan records the complete source traceability from the local MRI
harmonization reading set, the Causal_fusion CAPM implementation, DS-039, and
DS-040. The experiment must not be marked complete until the source-free access
audit, residual discrepancy diagnostics, CAPM-anchor preservation, synthetic
perturbation check, and paired source/target reports are archived.
