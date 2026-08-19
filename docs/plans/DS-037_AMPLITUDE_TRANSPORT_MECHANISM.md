# DS-037: Amplitude transport mechanism audit

Owner: cyh

Status: PROPOSED

Result record: `docs/results/DS-037/README.md`

## Question

The completed DS-036 comparison tested one configuration of feature-space
amplitude transport and found a negative target-side result. DS-037 does not
search for a better target score. It tests whether the failure is tied to the
transport strength or to the phase assumption used to reconstruct the
transported feature.

The experiment is a mechanism audit, not a new UDA method claim.

## Locked protocol

- Direction: `ADNI_to_NACC` only.
- Task: MCI vs AD, scan-filtered manifests.
- Seeds: `42, 43, 44, 45, 46`.
- Backbone: the same `layer4_pixel` CAPM backbone as DS-036.
- Source split: the DS-036 split policy and source-only validation selector.
- Target contract: subject-disjoint `T_adapt` and internal frozen `T_test`.
- `T_adapt`: image tensors only; no target labels, covariates, environments, or
  metrics are materialized during adaptation.
- `T_test`: used once after checkpoint selection; results remain exploratory
  because this holdout has appeared in historical work.
- No target-test tuning, early stopping, or variant deletion.
- The deferred full-target protocol extension is not part of DS-037.

## Registered variants

All variants use the same optimizer, loss weights, source rows, target rows,
and checkpoint selector. The transport operates on the shared layer4 feature
map and mixes only Fourier amplitude.

| ID | Variant | Strength | Phase used for reconstruction | Role |
|---|---|---:|---|---|
| AT0 | `capm_control` | none | none | matched no-transport control |
| AT1 | `source_phase_a025` | 0.25 | source phase | weak transport sensitivity |
| AT2 | `source_phase_a050` | 0.50 | source phase | DS-036 registered configuration |
| AT3 | `source_phase_a100` | 1.00 | source phase | full-amplitude replacement |
| AT4 | `target_phase_a050` | 0.50 | target phase | phase-assumption diagnostic |
| AT5 | `target_phase_a100` | 1.00 | target phase | aggressive phase diagnostic |

AT1-AT3 are the primary strength test. AT4-AT5 are secondary diagnostics:
target phase is not assumed to preserve source semantic layout and therefore
must not be described as a preferred method if it wins on this exploratory
holdout.

## Required implementation and audits

The runner must expose phase selection explicitly rather than silently
changing the existing source-phase behavior. For every run record:

- resolved `strength` and phase mode;
- amplitude L1 discrepancy before transport;
- source clean/mixed prediction agreement;
- transported-source CE and consistency loss;
- feature mean, standard deviation, norm, and finite-value checks;
- CAPM gate/effective-field summaries for clean and transported branches;
- source validation/test and exploratory target metrics.

The target-label audit must state that no target label, target covariate,
target environment, or target metric was read before final evaluation.

## Decision rule

Do not select a winning strength using `T_test`. Report all six variants per
seed and the paired differences against AT0. A mechanism signal is considered
credible only when it is directionally consistent across seeds, does not cause
source-label preservation failure, and is not explained by a large change in
source sensitivity/specificity alone.

Even a positive AT result supports only “the tested transport parameter or
phase choice is worth further investigation.” It does not support scanner,
manufacturer, or field-strength causal language.
