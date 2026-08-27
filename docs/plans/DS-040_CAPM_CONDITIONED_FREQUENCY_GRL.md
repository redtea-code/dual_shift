# DS-040: CAPM-conditioned frequency UDA with domain/intensity GRL

Owner: cyh

Status: PROPOSED

Result record: `docs/results/DS-040/README.md`

## 1. Question

DS-035 provides the clearest exploratory FMM signal: the complete FMM route
was above its matched reference in three seeds, while the joint no-GRL control
was not. DS-038 then showed that domain-only and intensity-only GRL effects are
task- and direction-dependent; it did not justify treating the two heads as
universally complementary. DS-039 demonstrated a small, bounded post-CAPM
residual pilot effect, but did not include a learned discriminator or Fourier
enhancement.

DS-040 asks the next mechanistic question:

> Can raw-Fourier source/target-style enhancement and domain/intensity GRL be
> made safer and more informative by applying the adversarial objectives after
> CAPM conditioning, preferably on a CAPM-conditioned batch residual rather
> than on the complete diagnostic representation?

This is a conditional UDA mechanism audit, not a claim that either GRL head is
biologically specific or that scanner causality has been identified.

## 2. Motivation and innovation hypothesis

AD biology and acquisition/cohort effects may be coupled. An unconditional
domain adversary can suppress both nuisance information and diagnostic
structure. CAPM supplies a three-variable (`age`, `sex`, `education`)
diagnostic-support condition. The proposed functional division is:

```text
frequency augmentation -> backbone F4 -> CAPM(F4, z) = B4
                         -> diagnostic anchor + batch residual
                         -> domain/intensity GRL on residual
                         -> classifier on the CAPM-conditioned path
```

The novelty is not merely adding a GRL, changing channel count, or changing a
feature size. It is to condition the domain objectives on the CAPM path and
restrict the adversarial pressure to a residual component. The complete CAPM
feature GRL is retained as a negative/control path to test whether this
restriction matters.

## 3. Relationship to prior frequency experiments

The frequency components have distinct roles and must not be conflated:

1. **FMM raw-Fourier enhancement** creates source intensity variants and
   source/target-style images using amplitude mixing while retaining the target
   phase in the inter-domain branch. Domain and intensity GRL operate on the
   resulting feature batches.
2. **DS-034/C4 feature-frequency gate** applies a feature-map rFFT amplitude
   gate and environment/GroupDRO controls. Its two-seed target BA point
   estimate was positive but paired uncertainty crossed zero, and its band
   ordering was seed-dependent. It is not the primary intervention in DS-040.
3. **DS-034 feature-frequency audit** is a domain-separability probe, not a
   trained adaptation method. It showed cohort-associated spectral information,
   not a causal link between lowering spectral separability and target BA.

DS-040 therefore uses the FMM raw-Fourier path for a fair domain-adaptation
factor, while keeping the C4 route as a separately identified future control.

## 4. Locked protocol

- Primary direction: `ADNI_to_NACC`.
- Prespecified extension: `NACC_to_ADNI`, reported separately and never
  averaged into the primary conclusion.
- Task: MCI vs AD, scan-filtered manifests.
- Backbone: ResNet10 CAPM backbone with `layers=[1,1,1,1]`, matching the
  selected frequency comparison; no FMM ten-convolution encoder in the
  primary matched-backbone analysis.
- Feature stage: `layer4_pixel` for the first matched implementation.
- Classifier feature width: 512 channels at layer4; all F0-F3/R1-R3 cells
  share this exact backbone capacity.
- CAPM variables: `[age, sex, education]` only.
- Seeds: `42, 43` for the initial two-seed screening, with no target-based
  strength or variant selection.
- `S_train`: diagnosis supervision and intensity-stage source training.
- `S_val`: sole epoch/checkpoint selector.
- `S_test`: source-side preservation report.
- `T_adapt`: image-only target batch and source/target membership; no target
  label, covariate, environment, prediction, or metric.
- `T_test`: one-time post-selection exploratory evaluation only.
- Subject split: `T_adapt` and `T_test` are subject-disjoint.
- Target labels: read only during final `T_test` reporting.
- No target-test tuning, threshold tuning, pseudo-labeling, or target-based
  early stopping.

The primary matrix is eight variants by seeds `42/43` (16 cells). The
NACC→ADNI extension, if run, uses the same 16-cell matrix and is reported as a
separate directional audit.

### 4.1 Conditional CAPM input contract

During training, the raw target image used in a source/target-style pair has no
target covariate. The pair therefore reuses the source sample's `z_s` when
computing CAPM for the synthetic target-style branch. This keeps the
domain/intensity contrast conditional on the same demographic input and does
not materialize target covariates. Final frozen target reporting follows the
ordinary CAPM evaluation contract and must record that distinction.

## 5. Registered variants

All variants use the same `layer4_pixel` CAPM backbone, raw-Fourier transform
parameters, source split, optimizer, and source-validation selector. The
frequency control includes a shared spatial-attention consistency branch so
that `T_adapt` has a defined training role even when both GRL coefficients are
zero. This attention branch is fixed across variants and is not a new factor.

| ID | Variant | Frequency enhancement | Domain GRL | Intensity GRL | GRL input | Role |
|---|---|---:|---:|---:|---|---|
| P0 | `capm_erm` | no | no | no | none | source-only CAPM baseline |
| F0 | `capm_fmm_control` | yes | no | no | none | CAPM + raw-Fourier/attention control |
| F1 | `capm_domain_full` | yes | yes | no | complete `B4` | full-CAPM domain adversary control |
| F2 | `capm_intensity_full` | yes | no | yes | complete `B4` | full-CAPM intensity adversary control |
| F3 | `capm_both_full` | yes | yes | yes | complete `B4` | full-CAPM two-head factorial control |
| R1 | `capm_domain_residual` | yes | yes | no | `R4=(I-P_task)B4` | proposed domain-residual path |
| R2 | `capm_intensity_residual` | yes | no | yes | `R4=(I-P_task)B4` | proposed intensity-residual path |
| R3 | `capm_both_residual` | yes | yes | yes | `R4=(I-P_task)B4` | primary conditional combination |

`P_task` is a frozen, source-only channel projector constructed from the P0
checkpoint: collect per-subject gradients of source CE with respect to pooled
CAPM features on `S_train`, form their centered covariance, and retain the top
`q=32` eigenvectors. `R4` is broadcast over spatial positions. This is a
source-task support projector, not a claim of a pure biological subspace.
All R variants use the same projector computed before any target adaptation.

P0 is first trained with the ordinary source-only CAPM protocol and selected
by source validation. F0-F3 and R1-R3 are initialized from that same
seed-specific P0 checkpoint and receive the same fixed UDA fine-tuning budget.
This makes the comparison a warm-start adaptation audit rather than seven
independent architecture searches.

The already-run DS-039 `RA0/RA1` frozen channel correction is a contextual
reference and is not substituted for F0. All eight cells are registered in the
primary direction; if compute is staged, the order of execution may change but
the final report must not delete F3 or R3 after inspecting `T_test`.

## 6. Model and loss definition

For source `x_s`, intensity-transformed source `x_i`, and Fourier-generated
target-style image `x_t^*`, let:

```text
F_s = E(x_s),       F_i = E(x_i),       F_t = E(x_t^*)
B_s = CAPM(F_s, z_s), B_i = CAPM(F_i, z_s), B_t = CAPM(F_t, z_s)
R_j = (I-P_task)B_j
```

The classifier receives `B_s` and source diagnosis labels. The adversarial
heads receive either `GAP(B)` for full variants or `GAP(R)` for residual
variants. The fixed projector is not trained by target data. The shared
spatial-attention map is used for a fixed source/target-style consistency term;
if the current CAPM backbone cannot expose such a map, the implementation must
add one common gate before CAPM rather than silently dropping the term.

```text
L_cls = 0.5 CE(C(B_s), y_s) + 0.5 CE(C(B_i), y_s)
L_dom = BCE(D_dom(GRL(G(B_s))), 0)
       + BCE(D_dom(GRL(G(B_t))), 1)
L_int = BCE(D_int(GRL(G(B_s))), 0)
       + BCE(D_int(GRL(G(B_i))), 1)
L_anchor = ||P_task B_s - P_task B_i||^2
L = L_cls + lambda_dom L_dom + lambda_int L_int
    + lambda_att L_attention + lambda_anchor L_anchor
```

Initial locked coefficients are `lambda_dom=lambda_int=1.0`,
`lambda_att=1.0`, and `lambda_anchor=0.1`; the GRL coefficient is `1.0` as in
DS-038. The sign and coefficient convention must be persisted explicitly. A
discriminator loss near `log(2)` is not sufficient evidence of alignment.

Each discriminator is an independent `512 -> 64 -> 1` MLP applied after
global-average pooling. Full variants use the pooled 512-dimensional `B4`;
residual variants use the pooled `R4`. Discriminator capacity and dropout are
fixed across all cells.

The raw-Fourier transforms are fixed to the prior FMM audit settings:

```text
x_i = iFFT(A( T_int(x_s) ), P(x_s)),    scale in [0.8, 1.2], noise std=0.05
x_t^* = iFFT((1-lambda) A(x_t) + lambda A(x_s), P(x_t)), lambda in [0, 1]
```

The source row's `z_s` is reused for `B_t` because `x_t^*` has no target
covariate. This is an algorithmic condition match, not target demographic
access.

## 7. Required diagnostics

Every epoch and selected checkpoint must persist:

- domain/intensity discriminator loss, accuracy, balanced accuracy, and AUC;
- GRL coefficient and shared-encoder gradient norm;
- discriminator parameter gradient norm;
- source clean/source-intensity and source/target-style classification
  agreement or CE where applicable;
- source/target feature mean, covariance or CORAL/MMD proxy before and after
  the adversarial path;
- task-support projector rank, eigenvalue mass, and residual/full feature
  domain-probe comparison;
- CAPM anchor drift between clean and adapted features/logits;
- residual norm, correction norm, and finite-value checks;
- attention consistency loss and raw-Fourier amplitude diagnostics;
- source validation/test and exploratory target BA/AUC, sensitivity, and
  specificity;
- source/target subject digests, config/checkpoint hashes, and exact variant
  flags.

## 8. Decision rules

The primary mechanism comparison is paired within seed against `F0`; full and
residual placements must also be compared within the same GRL head:

- `F0` vs `P0`: effect of raw-Fourier enhancement without GRL;
- `F1` vs `F0`: effect of domain GRL on complete CAPM feature;
- `R1` vs `F0`: effect of domain GRL on CAPM residual;
- `R1` vs `F1`: whether residual restriction improves preservation;
- corresponding `F2/R2` comparisons for intensity GRL;
- `R3` may support conditional complementarity only if it exceeds both `R1`
  and `R2` with consistent paired evidence and mechanism diagnostics.

Promising evidence requires, jointly:

1. no target BA/AUC loss relative to `F0`;
2. reduced residual/domain discrepancy;
3. bounded CAPM anchor drift and source-task preservation;
4. finite, non-collapsed discriminator/encoder gradients; and
5. no apparent gain explained only by sensitivity/specificity movement.

If full-CAPM GRL degrades while residual GRL preserves performance, that is
evidence for the proposed placement hypothesis, not proof that the residual is
pure biology. If domain-only and intensity-only change direction by task or
direction, retain that interaction rather than averaging it away.

## 9. Stop conditions and claim boundary

Stop the run family if target labels enter `T_adapt`, target-test metrics affect
selection, source/target subjects overlap, the CAPM input contract changes, or
the diagnostic schema is incomplete. Do not expand strengths, layers, or
additional seeds after inspecting target rankings.

Even a positive result supports only “CAPM-conditioned GRL placement is worth
further investigation” under this UDA protocol. It does not establish scanner,
manufacturer, field-strength, or causal-biology correction.

## 10. Implementation and result location

The planned implementation should extend the existing FMM runner and CAPM
backbone with explicit full-feature and residual GRL branches. The result
record must include code commit, resolved configuration, manifest hashes,
artifact inventory, and the complete paired comparison before changing this
experiment to `COMPLETED`.
