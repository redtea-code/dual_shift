# Plan 34 Image-Only Feature Frequency Audit (2026-08-14)

## Scope and safeguards

- Frozen seed=42 E2 `image_only` checkpoints only; no training, fine-tuning, or checkpoint reselection was performed.
- Coverage: `layer3_patch2`, `layer4_pixel`, and `layer5_pixel`, each for ADNI→NACC and NACC→ADNI.
- Each checkpoint was evaluated on both source and frozen target cohorts after per-cohort `earliest_visit` subject deduplication.
- Each feature map is reduced by per-channel 3D FFT, then arithmetic mean power across channels. The frequency bands were fixed before extraction: low `[0.00, 0.15)`, mid `[0.15, 0.35)`, high `[0.35, 0.50]`.
- Diagnosis labels, classifier labels, task predictions, logits, and target performance were excluded from subject selection and frequency-feature construction.

## Data accounting and feature contracts

| Preset | Direction | Captured layer | Feature map `[C,D,H,W]` | Source subjects | Target subjects | Extraction failures |
|---|---|---|---|---:|---:|---:|
| `layer3_patch2` | `ADNI_to_NACC` | `layer3` | `(256, 10, 13, 10)` | 338 | 1292 | 0 |
| `layer3_patch2` | `NACC_to_ADNI` | `layer3` | `(256, 10, 13, 10)` | 1292 | 338 | 0 |
| `layer4_pixel` | `ADNI_to_NACC` | `layer4` | `(512, 5, 7, 5)` | 338 | 1292 | 0 |
| `layer4_pixel` | `NACC_to_ADNI` | `layer4` | `(512, 5, 7, 5)` | 1292 | 338 | 0 |
| `layer5_pixel` | `ADNI_to_NACC` | `layer5` | `(512, 3, 4, 3)` | 338 | 1292 | 0 |
| `layer5_pixel` | `NACC_to_ADNI` | `layer5` | `(512, 3, 4, 3)` | 1292 | 338 | 0 |

All six runs provided 1,630 feature-frequency summaries (338 ADNI + 1,292 NACC) with zero extraction failures. Numeric summaries were finite and low/mid/high frequency fractions summed to one per subject.

## Frequency-only domain classification

The fixed classifier is standardized logistic regression with 5-fold stratified subject-level CV (seed 42). It receives only the nine predefined feature-frequency summaries.

| Preset | Direction | Feature-domain BA | Feature-domain AUROC | Raw-image BA / AUROC |
|---|---|---:|---:|---:|
| `layer3_patch2` | `ADNI_to_NACC` | 0.806 | 0.931 | 0.925 / 0.987 |
| `layer3_patch2` | `NACC_to_ADNI` | 0.837 | 0.947 | 0.925 / 0.987 |
| `layer4_pixel` | `ADNI_to_NACC` | 0.741 | 0.896 | 0.925 / 0.987 |
| `layer4_pixel` | `NACC_to_ADNI` | 0.558 | 0.744 | 0.925 / 0.987 |
| `layer5_pixel` | `ADNI_to_NACC` | 0.870 | 0.962 | 0.925 / 0.987 |
| `layer5_pixel` | `NACC_to_ADNI` | 0.678 | 0.852 | 0.925 / 0.987 |

## Key spectral effect sizes

Cohen’s d is source minus target. Positive values mean the source cohort has a larger feature value; p-values are two-sided Mann–Whitney U values. These comparisons are descriptive (six fixed checkpoints and nine fixed summaries), not a new model-selection procedure.

| Preset | Direction | Low fraction d | Mid fraction d | High fraction d | Spectral centroid d | Spectral slope d |
|---|---|---:|---:|---:|---:|---:|
| `layer3_patch2` | `ADNI_to_NACC` | 0.774 | -1.321 | 0.176 | 0.243 | 0.799 |
| `layer3_patch2` | `NACC_to_ADNI` | -1.711 | 0.599 | 0.817 | 0.804 | 0.307 |
| `layer4_pixel` | `ADNI_to_NACC` | 1.163 | 0.942 | -1.527 | -1.517 | -1.576 |
| `layer4_pixel` | `NACC_to_ADNI` | 0.288 | -0.466 | -0.081 | -0.254 | 0.335 |
| `layer5_pixel` | `ADNI_to_NACC` | -1.410 | 1.577 | 0.400 | 1.317 | -0.393 |
| `layer5_pixel` | `NACC_to_ADNI` | 0.295 | -0.588 | 0.289 | -0.199 | 0.426 |

## Interpretation

1. **Cohort-level frequency separation persists in frozen image-only representations.** All six feature-domain AUROCs exceed 0.74; the range is 0.744–0.962. Thus the raw-image cohort signal is not erased by the backbone.
2. **Layer4 is the least separable representation only for the NACC→ADNI-trained checkpoint.** Its BA/AUROC is 0.558/0.744, versus 0.741/0.896 for ADNI→NACC layer4. This direction dependence means a single layer-level ranking is not supported.
3. **Layer5 preserves or amplifies a strong domain signal.** It yields the highest feature-domain classifier performance in both directions (BA/AUROC 0.870/0.962 and 0.678/0.852). This is consistent with domain information remaining available in deep image-only features, but does not prove it harms or causes classification transfer performance.
4. **Feature-domain separability is lower than raw-image separability in every run.** The raw-image reference is BA 0.925 and AUROC 0.987. That attenuation is compatible with partial domain-invariant transformation, but its size varies by checkpoint direction and depth.
5. **Do not interpret layer3 versus layer4 as depth-only.** `layer3_patch2` and `layer4_pixel` differ in selected feature stage and patch/token contract; although the image-only path does not apply patchwise aggregation, the comparison remains tied to separately trained preset-specific checkpoints.
6. **Causal scope remains limited.** Scanner field strength is confounded with sites, manufacturers, recruitment, cohort composition, and possible residual preprocessing differences. These results establish a cohort-associated spectral signature, not a 1.5T/3T causal effect.

## Reproducibility outputs

- Per-run subject summaries: `outputs/frequency_audit/image_only_features/{preset}/{direction}_seed42/subject_frequency_features.csv`
- Per-run provenance, input/checkpoint/config hashes, and failures: sibling `provenance.json`, `selection_audit.json`, `failed_images.json`.
- Per-run statistics and domain predictions: sibling `feature_frequency_statistics.json`, `feature_domain_predictions.csv`.
- Combined summary: `outputs/frequency_audit/image_only_features/feature_frequency_statistics.json`, `feature_domain_classifier_summary.csv`, `feature_domain_predictions.csv`.
- Input matrix: `outputs/frequency_audit/image_only_features/run_manifest.json`.
- Raw-image reference: `docs/PLAN34_FREQUENCY_AUDIT_RAW_IMAGE_2026-08-14.md`.
