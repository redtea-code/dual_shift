# DS-041: CAPM-conditioned source-free residual distribution alignment

Owner: cyh

Status: PROPOSED

Result record: `docs/results/DS-041/README.md`

## 1. Question

DS-039 showed that a small, fixed channel-statistic correction after CAPM can
produce a positive but seed-sensitive pilot signal. DS-040 showed that applying
domain/intensity GRL to the complete CAPM feature is not reliably beneficial,
and that residual placement is not yet proven because the screen used one epoch
and incomplete checkpoint-bound mechanism evidence.

DS-041 asks a narrower and more source-free question:

> Can distribution alignment be restricted to a source-task residual after CAPM,
> using only an unlabeled target adaptation population and compact source
> summary statistics, while the CAPM diagnostic-support component remains stable?

This is a mechanism and protocol pilot. It is not a scanner-causality study,
not an image harmonization claim, and not evidence that the residual is a pure
biological or non-biological subspace.

## 2. Motivation and proposed contribution

MRI biology and acquisition/cohort effects can be coupled. Aligning an entire
feature representation may remove both batch variation and diagnostic structure.
The proposed functional division is:

```text
MRI -> 3D backbone F4 -> CAPM(F4, z) = B4
                         |
                         +--> task-support component B_task
                         +--> residual component R4
                               |
                               +--> source-free distribution transport
                         B4' = B_task + transported(R4)
                         -> frozen/source-trained classifier
```

The contribution under test is the combination of four restrictions:

1. CAPM is computed first and retained as the diagnostic-support anchor.
2. A frozen source-only task-support projector defines the residual direction.
3. Only `R4` is aligned; `B_task` is not directly exposed to the alignment
   objective or transport operator.
4. Target adaptation uses unlabeled target images and source summary statistics,
   with no target labels, covariates, predictions, or target-test selection.

The claim is deliberately narrower than "biology-preserving harmonization":
the experiment tests reduced damage to a source-task support and measured
diagnostic/structural proxies. It does not identify a causal biological
representation.

## 3. Complete source traceability

The idea is assembled from the following local reading notes and existing
DualShift evidence. These are inspirations and baselines, not claims that
DS-041 reproduces their full models.

### 3.1 Direct CAPM source

The `original_capm` block is a source-faithful port of
`redtea-code/Causal_fusion` (`PatchwiseBackdoorBlock`, reference commit
`d1a37d92ba6e646b3ac710cfff44fb737daafaac`). It maps the three table variables
`[age, sex, education]` to patch-wise modulation of a spatial feature map.
DS-041 does not claim this CAPM operation as new.

The repository handoff and alignment record identify the source explicitly:

- `D:\ADNI\copy_project\review\journal_baseline_alignment.md`
- `D:\ADNI\copy_project\review\journal_implementation_handoff.md`
- `Model/ablation/scale_table_transformer.py::OriginalPatchwiseCAPM`

### 3.2 Statistical alignment source: SFHarmony

`reading_note_SFHarmony.md` in the local reading set describes source-free
feature adaptation: share source model weights and compact per-feature GMM
statistics, fit target GMMs from unlabeled target data, align distributions with
Bhattacharyya distance, freeze the source task predictor, and stabilize small
batch EM with memory. DS-041 borrows the access boundary, summary-statistic
provenance, frozen-head idea, and small-batch estimation discipline.

It does not copy SFHarmony unchanged: DS-041 restricts alignment after CAPM to
`R4`, and must explicitly test marginal-alignment failure under label shift.

### 3.3 Residual/statistic transport sources

`reading_note_BlindHarmony.md` motivates a target distribution as a soft prior
combined with a structure-distance term. `reading_note_ESPA.md` uses simulated
tissue-contrast or additive scanner residuals and checks whether the known
perturbation can be removed. `reading_note_IGUANe.md` uses 3D residual learning,
age-balanced sampling, and biological-model-based checkpoint selection.

DS-041 borrows these as constrained diagnostics: target statistics are a soft
prior, synthetic perturbation removal is a sanity check, and population balance
and biological preservation are reported. It does not introduce a flow model,
CycleGAN, or a multi-generator image harmonizer.

### 3.4 Biological-preservation source: ImUnity and DeepResBat

`reading_note_ImUnity.md` combines site confusion with a biological-preservation
head. The DeepResBat notes in the project literature record a related principle:
estimate or preserve covariate/biological effects, harmonize a residual, and
recombine them. DS-041 translates this principle into a CAPM anchor plus a
source-task residual; it does not equate CAPM with a causal biological effect.

### 3.5 Structure and evaluation source: MISPEL

`reading_note_MISPEL.md` uses matched scans, embedding coupling, reconstruction,
and multi-level structural/biomarker checks. DS-041 borrows the evaluation
hierarchy (distribution, structure, biomarker/task) and the warning that a
domain metric alone cannot establish preservation. Matched scans, if ever
available, are an external sanity check and are not required for the primary
non-paired ADNI/NACC protocol.

### 3.6 Internal predecessors and the novelty boundary

- DS-039 already implements bounded diagonal channel-statistic transport after
  the complete CAPM feature. Therefore "mean/std matching" and "bounded
  residual" are not new by themselves.
- DS-040 already implements complete-feature and task-residual GRL paths using
  a source-task projector. Therefore "residual-only GRL" is an existing
  project control, not the sole DS-041 contribution.
- Existing APIS/APIC modules already contain bounded residual interventions and
  style-statistic transport.

DS-041 is justified only if it tests a more specific combination: source-free
summary-statistic alignment on the CAPM-conditioned task residual, with a
measurable anchor-preservation contract and a controlled perturbation recovery
test. If those factors are not isolated, the result should be recorded as a
reimplementation or ablation, not a new method.

## 4. Model definition

### 4.1 Frozen CAPM path

Use the matched `original_capm` backbone from DS-039/040:

- ResNet10 configuration `layers=[1,1,1,1]` for the initial pilot;
- `layer4_pixel` feature preset;
- 512 feature channels;
- CAPM variables `[age, sex, education]`;
- source checkpoint selected only on `S_val` and then frozen.

For an image `x`:

```text
F4 = E(x)
B4 = CAPM(F4, z)
```

For target adaptation statistics, `z` is a fixed source-only reference table
`z_ref` (the source-training covariate mean after the frozen preprocessor). This
keeps the target adaptation loader image-only. Final target reporting may use
the ordinary target covariates required by the frozen CAPM model, and this
distinction must be recorded.

### 4.2 Task-support and residual decomposition

The primary decomposition reuses the DS-040 source-only projector:

```text
B_task = P_task B4
R4     = (I - P_task) B4
```

`P_task` is computed before target adaptation from source-training gradients of
the diagnosis CE with respect to pooled CAPM features. The rank `q=32` is fixed
before target evaluation. It is a source-task support projector, not a proof of
biological purity. A rank-zero/identity control is not allowed to be silently
substituted.

### 4.3 Source-free residual transport

The primary operator uses source summary statistics and unlabeled target
`T_adapt` statistics on `R4`.

For the GMM variants, the statistical observation unit is fixed before any
target result is read: one subject-level pooled residual vector
`u_n = GAP(R4_n)` per earliest scan per subject. GMM components are fitted per
channel over these subject-level vectors, with each subject contributing one
observation. Spatial activations must not be silently treated as independent
subjects. The exact DS-039 spatial-moment implementation may be retained as a
historical/contextual control, but it is not interchangeable with this
subject-level GMM statistic.

The K=1 subject-level control is the DS-039-style diagonal transport pattern
applied to the fixed pooled descriptors:

```text
A(R4) = (R4 - mu_T) / sigma_T * sigma_S + mu_S
R4'   = R4 + alpha * g * (A(R4) - R4)
```

where `alpha <= 0.25` is fixed and `g` is a frozen channel discrepancy gate.

The proposed distribution variant replaces a single Gaussian per channel with
`K=2` residual GMM summaries. Source GMM parameters are stored before target
adaptation; target GMM parameters are fitted from image-only `T_adapt`. Component
matching must be deterministic and audited. The alignment score is a weighted
Bhattacharyya distance or an equivalent fixed distribution distance. No
component count, strength, or target reference is chosen using `T_test`.

The transport must be bounded, finite, and identity at zero strength. The
classifier receives `B4'`; the source-task projector and classifier are frozen
in the first stage. Any trainable residual adapter is a separately registered
second-stage extension and cannot be mixed with the fixed-transport result.

## 5. Registered experiment cells

The first stage is deliberately small. All cells share the same checkpoint,
backbone, preprocessing, source split, target split, and final reporting code.

| ID | Variant | Alignment input | Statistics | Role |
|---|---|---|---|---|
| C0 | `capm_control` | none | none | Frozen original-CAPM control |
| C1 | `full_diag_transport` | complete `B4` | K=1 diagonal moments | Full-feature transport control; isolates scope risk |
| C2 | `residual_diag_transport` | `R4` | K=1 diagonal moments | Matched DS-039-style residual control |
| C3 | `residual_gmm_transport` | `R4` | K=2 GMM + distance | Primary SFHarmony-inspired candidate |
| C4 | `residual_gmm_perturbation` | `R4` | C3 plus synthetic perturbation recovery | Mechanism sanity check, not a target-test selector |

Optional only after the first stage satisfies the mechanism gate:

| ID | Variant | Purpose |
|---|---|---|
| C5 | `residual_gmm_anchor_loss` | Train a small residual adapter with an explicit CAPM-anchor loss |
| C6 | `residual_gmm_conditional_stats` | Stratify source summaries by source CAPM bins; target remains label-free |

C5/C6 are not part of the initial claim. They require a new registered budget
and must not be added after inspecting target rankings.

## 6. Synthetic perturbation sanity check

The ESPA-inspired C4 check uses source images only to create known, controlled
appearance perturbations. Candidate perturbations are:

- tissue-contrast changes sampled from fixed source-derived tissue statistics;
- mild raw-Fourier amplitude changes with source phase retained;
- bounded additive intensity residuals.

The perturbation parameters are fixed before running the real target test. The
question is whether the residual transport removes the known perturbation while
keeping the CAPM anchor and source diagnosis output stable. This does not prove
that the real ADNI/NACC shift has the same cause.

## 7. Data and access protocol

- Task: scan-filtered MCI vs AD.
- Primary direction: `ADNI_to_NACC`.
- Prespecified extension: `NACC_to_ADNI`, reported separately.
- Source split: `S_train`, `S_val`, `S_test`, subject-disjoint.
- Target split: subject-disjoint `T_adapt` and `T_test`.
- `T_adapt`: image and subject ID only; no target label, covariate,
  environment, prediction, or metric.
- Source summary: generated from the frozen source checkpoint and source-only
  `S_train`/summary protocol.
- Target summary: generated once from unlabeled `T_adapt`; no target-test
  recomputation after seeing results.
- Selection: source validation only; target test is one-time exploratory
  reporting because the internal holdout has historical exposure.
- No target threshold tuning, pseudo-labeling, target early stopping, or target
  reference selection.

For strict source-free runs, only the source checkpoint, source summary, frozen
projector, and target adaptation images are available to the adapter. If source
images are retained for a separate UDA comparison, that cell must be labeled
non-SFDA and reported separately.

## 8. Required artifacts and diagnostics

### 8.1 Provenance

Save for every seed and direction:

- resolved configuration and code commit/hash;
- source checkpoint and source split hashes;
- target split JSON and subject digests;
- source summary/GMM parameters and target summary/GMM parameters;
- GMM initialization, component matching, convergence and occupancy;
- explicit `target_labels_read=false` and `target_metrics_read=false` for the
  statistics stage;
- exact projector rank, eigenvalue mass and construction digest.

### 8.2 Distribution and intervention diagnostics

Before and after transport, report:

- diagonal mean/std discrepancy;
- CORAL-style covariance discrepancy where numerically feasible;
- MMD or energy-distance proxy;
- GMM Bhattacharyya distance and component occupancy;
- frozen full-feature and residual domain probes;
- residual RMS, correction RMS, relative strength and finite checks;
- identity loss and fraction of channels receiving nonzero correction.

### 8.3 Biology/task preservation

Report:

- source and target BA, AUROC, macro-F1, sensitivity and specificity;
- source validation and source-test preservation;
- CAPM anchor drift, including `||P_task(B4' - B4)||` and logit KL;
- diagnosis probe and age/sex/education probes where data permit;
- subgroup results by available CAPM variables and class balance;
- ROI/structure metrics if a fixed, independent pipeline is available;
- synthetic perturbation recovery and clean-vs-perturbed prediction agreement.

No single domain accuracy, loss near `log(2)`, SSIM, or target BA is sufficient
to declare success.

## 9. Decision rules

The primary candidate C3 is promising only if all conditions hold relative to
the matched controls:

1. residual distribution discrepancy decreases before/after transport;
2. C3 is no worse than C2 on the source-task preservation and CAPM-anchor
   criteria, while improving or maintaining target performance relative to C0;
3. full-feature C1 does not provide the same preservation profile, supporting
   the importance of restricting the alignment scope;
4. correction is finite, bounded, and not dominated by a small set of unstable
   GMM components;
5. the result is not explained solely by sensitivity/specificity threshold
   movement;
6. synthetic perturbation recovery succeeds without unacceptable anchor drift;
7. the direction and seed pattern are reported separately and do not rely on a
   single favorable cell.

If C3 reduces discrepancy but harms source diagnosis, CAPM anchor, or biological
probes, classify it as negative transfer. If C3 improves target BA without a
measured residual discrepancy reduction, classify it as an unexplained pilot
signal, not harmonization evidence.

## 10. Staged execution and compute gate

### Stage A: interface and synthetic smoke

- Run C0-C4 on synthetic tensors.
- Verify exact identity at zero strength, deterministic GMM component matching,
  finite gradients/outputs, and checkpoint round-trip.
- Do not inspect target test labels.

### Stage B: real-data mechanism pilot

- Run primary ADNI->NACC with seeds 42 and 43 under the fixed source/target
  protocol.
- Run the prespecified NACC->ADNI extension only with the same cells.
- Persist every required artifact before reading exploratory target metrics.

### Stage C: expansion gate

Do not add C5/C6, more ranks, strengths, or additional seeds unless Stage B
passes the discrepancy, preservation, provenance, and synthetic-recovery gates.
If it passes, register the expanded matrix in a new plan revision before
execution. Two seeds are a screening set, not a claim of universal behavior.

## 11. Claim boundary

A positive DS-041 result can support only:

> Under the registered ADNI/NACC protocol, source-free alignment restricted to
> a CAPM-conditioned residual is a viable and more auditable candidate than
> unrestricted feature-statistic transport.

It cannot support claims that:

- CAPM is a pure biological or causal representation;
- the residual is pure batch/scanner information;
- scanner, manufacturer, field-strength, or site causality has been identified;
- the method is stable across arbitrary tasks, cohorts, or seeds;
- GMM, AdaIN, CORAL, SFHarmony, or residual transport is novel in isolation.

## 12. Planned implementation

The implementation should reuse the DS-039/040 contracts and add only the
minimum new surface:

- `Model/ablation/capm_residual_distribution_alignment.py`
- `experiments/build_capm_residual_distribution_stats.py`
- `experiments/run_capm_residual_distribution_pilot.py`
- `tests/test_capm_residual_distribution_alignment.py`
- `ds041_capm_residual_distribution_scan_filtered_1p5t_mci_ad.yaml`

The result record must include the exact source references, baseline mapping,
configuration, split and checkpoint hashes, all cell artifacts, and a paired
decision table before the experiment status can move from `PROPOSED`.
