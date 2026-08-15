# Frequency-Guided UDA Experiment Plan

Version: 1.1

Status: preregistered execution plan. This document does not report a new
performance result.

## 1. Question And Claim Boundary

This experiment asks whether a label-free target-domain feature-spectrum prior
can improve cross-cohort MCI-vs-AD classification beyond a source-selected
`original_capm` route.

The method is **unsupervised domain adaptation (UDA)**, not zero-shot domain
generalization: it uses target images in `T_adapt`, but never target diagnoses,
target predictions, or target metrics before the final `T_test` evaluation.

The allowed claim is restricted to improved performance on a subject-disjoint,
unlabeled-target-adaptation protocol. It does not establish field-strength
causality, scanner harmonization, or universal frequency-shift removal. The
existing raw-image audit shows a strong ADNI/NACC frequency association, but
cohort, site, acquisition, preprocessing, and biology remain confounded.

## 2. Fixed Data Contract

For each direction and seed, construct the following sets before model fitting:

| Set | Content | May use labels? | May influence checkpoint/configuration? |
|---|---|---:|---:|
| `S_train`, `S_val`, `S_test` | Frozen source split | Yes | Source only |
| `T_adapt` | Target images and tables, subject-disjoint from `T_test` | No | Frequency prior only |
| `T_test` | Held-out target subjects | Final evaluation only | No |

- Split the target cohort by `subject_id` only, with `adaptation_fraction=0.5`
  and a recorded adaptation seed. Do not stratify this split by diagnosis.
- For the frequency prior, retain only the earliest valid visit per subject;
  break ties by folder/path. This prevents repeat scans from changing the
  source/target spectral population.
- The primary ADNI-to-NACC route uses the NACC 3T target cohort. For the
  NACC-to-ADNI primary route, prepare an ADNI common-support 3T manifest before
  the `T_adapt/T_test` split. ADNI 1.5T is an explicitly separate unsupported
  protocol stress test and must not select a model or frequency prior.
- Preserve the scan-filtered task mapping, preprocessing, subject split, input
  shape, ResNet10 `layers=[1,1,1,1]`, source-validation selector, and collapse
  guard from the current ResNet10 reports.

## 3. Method

The new branch is fixed as:

```text
image -> ResNet layers 1-4 -> DGF-Gate -> layer5 -> original CAPM(table) -> classifier
```

`DGF-Gate` is the `DomainGuidedFrequencyGate3D` module. A source-selected,
frozen `original_capm` checkpoint produces the pre-CAPM layer4 maps used for
the prior. It computes source and `T_adapt` low/mid/high spectral fractions,
standardizes their difference, and stores the resulting discrepancy vector in
`frequency_prior.json`.

The runtime gate applies a bounded, residual 3-D rFFT amplitude attenuation to
each sample before layer5. It preserves phase and has no domain-ID input. The
model's high band is `r >= 0.35`, rather than the raw-audit descriptive
`[0.35, 0.50]` interval, because the three model masks must partition every
rFFT coefficient without dropping 3-D corner frequencies.

Each source training sample receives exactly one supervised frequency
environment: `original`, `lowpass`, `downsample_resample`, or `mild_blur`.
GroupDRO operates on these four environment IDs. There is no paired scan,
paired loss, target pseudo-label, domain-adversarial classifier, or target
metric-based selection.

The prior-extraction loader emits only `image` and `subject_id`; it does not
request label-bearing dataset samples. Target diagnosis labels remain sealed
until final `T_test` scoring.

## 4. Locked Comparison Matrix

All variants use `layer5_pixel + original_capm`, the same frozen
source-selected checkpoint as their initialization, source split, seed,
optimizer, second-stage training budget, selector, and final `T_test` rows.

| ID | Variant | Purpose |
|---|---|---|
| C0 | `frequency_uda_baseline` | Same source initialization, no frequency environment or gate |
| C1 | `frequency_uda_env_dro` | Frequency environments plus GroupDRO, without a frequency gate |
| C2 | `frequency_uda_uniform` | Gate with equal band discrepancies; controls for gate capacity |
| C3 | `frequency_uda_permuted` | Gate with a fixed non-identity permutation of real discrepancies |
| C4 | `frequency_uda` | Full target-spectrum-guided gate |

The primary contrast is `C4 - C0`. `C4 - C1` isolates the added gate beyond
frequency-environment training, `C4 - C2` tests target-specific weighting over
uniform weighting, and `C4 - C3` tests whether the observed band ordering
matters. No extra preset, band edge, strength, or architecture may be selected
after reading `T_test` results.

## 5. Execution Order

For each fixed direction/seed, use dedicated output directories and resolve
the YAML paths before running. Do not reuse one direction's prior, split, or
checkpoint in the other direction.

1. Train the source-only CAPM preparation checkpoint and retain its split. It
   is the common initialization for C0-C4 and supplies the prior features:

```text
<PYTHON> experiments/train_journal.py \
  --config_path <RESOLVED_UDA_CONFIG> \
  --direction <DIRECTION> \
  --variants original_capm \
  --source-only \
  --output-dir <PREPARE_DIR> \
  --device cuda
```

2. Build the target split and label-free frequency prior:

```text
<PYTHON> experiments/build_frequency_uda_prior.py \
  --config <RESOLVED_UDA_CONFIG> \
  --direction <DIRECTION> \
  --source-split-manifest <PREPARE_DIR>/split_manifest.json \
  --source-checkpoint <PREPARE_DIR>/original_capm/best_checkpoint.pt \
  --prior-output <PREPARE_DIR>/frequency_prior.json \
  --target-split-output <PREPARE_DIR>/target_adapt_test_split.json \
  --adaptation-fraction 0.5 \
  --adaptation-seed <SEED> \
  --device cuda
```

3. Set `frequency_uda.source_split_manifest`, `base_checkpoint`,
`prior_path`, and `target_split_manifest` in that direction/seed's resolved
YAML. The trainer verifies the source split, target split, prior, and baseline
checkpoint hashes before starting a frequency-gate variant.

4. Run the locked matrix:

```text
<PYTHON> experiments/train_journal.py \
  --config_path <RESOLVED_UDA_CONFIG> \
  --direction <DIRECTION> \
  --variants frequency_uda_baseline frequency_uda_env_dro frequency_uda_uniform frequency_uda_permuted frequency_uda \
  --output-dir <FINAL_DIR> \
  --device cuda
```

Use seeds `43` and `44`. The two directions are independent experiments; do
not average them into a single primary effect.

## 6. Required Artifacts

Every direction/seed must retain:

```text
resolved_config.yaml
source split_manifest.json
source original_capm/best_checkpoint.pt
frequency_prior.json
target_adapt_test_split.json
target_adapt_test_split hash
final split_manifest.json
variant checkpoints, metrics, predictions, and paired_comparisons.json
summary.json
```

The prior JSON must certify `target_labels_read=false` and
`target_metrics_read=false`. The final manifest must identify the target split
hash and the subject digests of `T_adapt` and `T_test`.

## 7. Evaluation And Decision Rules

Report subject-level balanced accuracy, AUROC, macro-F1, sensitivity,
specificity, Brier score, ECE, and paired bootstrap contrasts against C0 on
the same `T_test` rows. Report individual seeds before their mean.

The frequency-guided branch is eligible for the next model-development phase
only when all of the following hold:

1. C4 improves target BA over C0 in both seeds for the predeclared direction.
2. C4 is not worse than C1 and C2 in either seed, and has a positive mean
   contrast against both controls.
3. Source validation passes the existing collapse guard, and the final audit
   confirms disjoint source, `T_adapt`, and `T_test` subjects.
4. All prior/checkpoint/split hashes agree and no target label or target metric
   was read before final evaluation.

If C1 matches C4, the conclusion is frequency-environment robustness, not
target-spectrum guidance. If C2 or C3 matches C4, the conclusion is a bounded
spectral-gate effect, not evidence that the learned domain-frequency ordering
is essential. If no condition is met, retain the result as a negative UDA
experiment and do not add SoftRegion or a second new mechanism.
