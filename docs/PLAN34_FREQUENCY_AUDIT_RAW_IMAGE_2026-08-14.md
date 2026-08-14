# Plan 34 Raw-Image Frequency Audit (2026-08-14)

## 1. Scope and protocol

- Input: scan-filtered preprocessed NIfTI images from ADNI 1.5T and NACC 3T manifests.
- Deduplication: `earliest_visit`; one earliest scan per subject, tie-broken by `image_path`.
- Frequency bands were fixed before inspection: low `[0.00, 0.15)`, mid `[0.15, 0.35)`, high `[0.35, 0.50]`.
- No diagnosis labels, target metrics, or model checkpoints were used.
- This is a cohort association audit; it cannot establish that field strength caused the observed differences.

## 2. Data accounting

| Cohort | Manifest rows | Selected subjects | Duplicate rows discarded | Processed | Failed |
|---|---:|---:|---:|---:|---:|
| ADNI | 1108 | 338 | 770 | 338 | 0 |
| NACC | 1595 | 1292 | 303 | 1292 | 0 |

The final table contains **1,630 subject-level rows** (338 ADNI, 1,292 NACC); the failure list is empty.

## 3. Frequency summary comparison

| Feature | ADNI mean | NACC mean | Cohen d (ADNI−NACC) | Mann–Whitney p | Bootstrap mean-difference 95% CI |
|---|---:|---:|---:|---:|---:|
| total_log_power | 13.4739 | 13.4496 | 0.398 | 1.62e-10 | [0.0175017, 0.0310126] |
| low_fraction | 0.925937 | 0.913927 | 1.255 | 2.73e-83 | [0.011128, 0.0128697] |
| mid_fraction | 0.0598595 | 0.0678554 | -1.025 | 2.02e-59 | [-0.00871686, -0.00727878] |
| high_fraction | 0.0142031 | 0.0182178 | -1.782 | 7.48e-125 | [-0.004249, -0.00376792] |
| spectral_centroid | 0.058109 | 0.0593575 | -0.249 | 2.71e-07 | [-0.00173748, -0.000769197] |
| spectral_slope | 2.64592 | 2.53617 | 2.677 | 5.3e-154 | [0.104892, 0.114804] |

## 4. Frequency-only domain classifier

A fixed standardized logistic-regression classifier using only the six frequency summaries achieved cross-validated balanced accuracy **0.925** and AUROC **0.987** (5-fold stratified subject-level CV, seed 42). These values are descriptive and were not used to tune the bands or select models.

## 5. Interpretation

- If the classifier separates cohorts and the separation is concentrated in mid/high-frequency summaries, the results support continuing to model-feature spectral analysis.
- If separation is concentrated in total power or low-frequency summaries, residual intensity, preprocessing, anatomy, site, or cohort composition remain plausible explanations; it should not be labeled a high-frequency scanner effect.
- These data support testing frequency characteristics as one possible component of the ADNI/NACC domain shift, but do not distinguish scanner field strength from other cohort confounders.

## 6. Reproducibility

- Features: `outputs/frequency_audit/raw_image/frequency_features.csv`
- Statistics: `outputs/frequency_audit/raw_image/frequency_statistics.json`
- Selection audit: `outputs/frequency_audit/raw_image/selection_audit.json`
- Failed images: `outputs/frequency_audit/raw_image/failed_images.json`
- Input hashes: `outputs/frequency_audit/raw_image/input_sha256.txt`
- Configuration: `outputs/frequency_audit/raw_image/frequency_config.json`
