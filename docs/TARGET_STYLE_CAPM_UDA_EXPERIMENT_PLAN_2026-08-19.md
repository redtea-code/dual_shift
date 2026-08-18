# Target-style transport + CAPM UDA probe

## Positioning

This is an implementation of a necessary experimental control, not a novelty
claim. It is intentionally close to the FMM/spectral UDA family so that the
question is isolated: does an unlabeled target image improve the existing CAPM
backbone when the target information is restricted to a feature-space style
statistic?

The existing C0-C4 experiments and their checkpoints are not changed.

## Exact target-data contract

The target cohort is split by subject into `T_adapt` and a frozen internal
`T_test`. During training:

- `T_adapt` contributes only target image tensors;
- the image is encoded by the shared layer4 encoder and only its spatial FFT
  amplitude is used;
- source feature phase is retained, and source labels supervise the transported
  source feature;
- target labels, target covariates, target predictions, and target metrics are
  not read by the adaptation loss or checkpoint selector.

The target covariates are still transformed for the ordinary target inference
call, because CAPM requires the same table contract as the source model. They
are not passed to the transport module. `T_test` is evaluated once after the
source-validation checkpoint is selected.

## Models

All variants use the same `layer4_pixel` scale-table CAPM backbone and the same
source split, optimizer, class weighting, and source-only checkpoint selector.

1. `capm`: CAPM with no target adaptation (`strength=0` control).
2. `target_style_capm`: source clean CE + source-label CE on target-style
   transported features + a small clean/mixed prediction consistency term.

The transport is bounded by `strength` (default `0.5`), initialized as an
identity control, and target features are detached before amplitude mixing.

## Multi-seed protocol

Run the primary direction `ADNI_to_NACC` for seeds `42, 43, 44, 45, 46` with
the same frozen scan-filtered manifest and split policy:

```powershell
python experiments/train_target_style_uda.py `
  --config target_style_capm_uda_scan_filtered_1p5t_mci_ad.yaml `
  --direction ADNI_to_NACC `
  --seeds 42 43 44 45 46 `
  --variants capm target_style_capm `
  --device cuda
```

The reverse direction is exploratory stress evidence only and must not be
pooled into the primary claim. Do not use target-test performance to tune
`strength`, loss weights, or the target split ratio.

## Required reporting

Report per seed and mean +/- standard deviation for source validation,
source test, and frozen `T_test`:

- balanced accuracy (primary), ROC-AUC, macro-F1, sensitivity, specificity;
- subject-aggregated and scan-level counts;
- target amplitude-discrepancy audit and the fraction of transported batches;
- CAPM gate/effective-field summaries;
- subgroup metrics by sex, age bin, and education bin where cell counts permit;
- source-label preservation: clean-vs-transported prediction agreement and
  transported-source CE.

Also run an image-only `interaction="image_only"` control if the CAPM-specific
increment must be separated from a generic target-style increment. This is a
secondary control, not a replacement for the matched `capm` comparison.

## Go / no-go rule

Proceed to a fresh label-blind target holdout only if the five-seed paired
difference

`BA(target_style_capm) - BA(capm)`

is positive in the point estimate, does not rely on a single seed, and does not
produce an unacceptable source-validation drop or a concentrated demographic
subgroup failure. A positive result supports “unlabeled target-style transport
is worth further study”; it does not support scanner-causal or field-strength
causal language.
