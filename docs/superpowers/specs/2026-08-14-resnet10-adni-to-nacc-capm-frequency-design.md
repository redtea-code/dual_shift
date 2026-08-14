# ResNet10 ADNI-to-NACC CAPM and Feature-Frequency Design

## Goal

Complete the missing reverse-direction comparison by training frozen ResNet10 image-only and original-CAPM models on ADNI and blind-testing them on NACC. Then apply the same task-filtered feature-frequency audit used for the existing NACC-to-ADNI ResNet10 checkpoints.

## Fixed experiment matrix

| Direction | Backbone | Seeds | Presets | Variants | Checkpoints |
|---|---|---|---|---|---:|
| ADNI to NACC | ResNet10, `layers=[1,1,1,1]` | 43, 44 | `layer3_patch2`, `layer4_pixel`, `layer5_pixel` | `image_only`, `original_capm` | 12 |

The new matrix is paired to the already-complete NACC-to-ADNI ResNet10 matrix with the same seeds, presets, variants, scan-filtered task protocol, input shape, source-validation selection rule, and blind frozen target evaluation.

## Training and evaluation protocol

1. Derive one resolved config per preset/seed from the established ResNet10 configurations. Preserve `layers=[1,1,1,1]`, optimizer, split seed, input shape, model contract, and all scan-filtering constraints.
2. Change only the direction to `ADNI_to_NACC` and output root to a new, dedicated output directory. No prior output directory may be modified.
3. Train both variants independently. Select `best_checkpoint.pt` solely from ADNI source validation balanced accuracy plus the existing collapse guard.
4. Evaluate each selected checkpoint once on the frozen NACC target. Target metrics must not select epochs, variants, presets, or thresholds.
5. Retain per-run resolved config, source split manifest, training metrics, frozen checkpoint, target metrics, target predictions, and hashes.

## Feature-frequency audit

For all new ADNI-to-NACC checkpoints and all existing paired NACC-to-ADNI checkpoints:

- Reconstruct task-filtered source and target datasets from each frozen resolved configuration.
- Extract the selected map: layer3 for `layer3_patch2`, layer4 for `layer4_pixel`, layer5 for `layer5_pixel`.
- Compute a 3D FFT per feature channel and average frequency-bin power across channels.
- Use fixed radial bands low `[0.00,0.15)`, mid `[0.15,0.35)`, high `[0.35,0.50]`.
- Record total log power, band energies/fractions, spectral centroid, and spectral slope.
- Compare source/target distributions with Cohen's d, Mann-Whitney U, and bootstrap CIs. Use fixed standardized logistic-regression 5-fold CV to assess domain separability from frequency summaries only.

## Primary comparisons

1. For each seed/preset/direction, compare `original_capm` against its matched `image_only` checkpoint.
2. Compare the feature-domain classifier and spectrum statistics across ADNI-to-NACC versus NACC-to-ADNI.
3. Relate the pre-specified CAPM minus image-only target BA difference to the pre-specified feature-domain separation difference. This is an association analysis only.
4. Report seed-specific results before cross-seed averages. Do not choose a preferred preset or variant from target metrics.

## Interpretation limits

- Existing NACC-to-ADNI evidence shows initial paired benefit for `original_capm` at layer4 across seeds 43/44; it does not establish a universal ResNet10 or CAPM advantage.
- This experiment will test directional reproducibility, not establish field-strength causality. Scanner field strength remains confounded with site, manufacturer, cohort composition, and residual preprocessing differences.
- Layer3 patch2 versus layer4 pixel differs in selected feature stage and preset/token contract, so it is not a depth-only comparison.

## Success criteria

- All 12 ADNI-to-NACC training runs produce source-selected frozen checkpoints that pass the collapse guard.
- All 12 frozen checkpoints complete blind target evaluation.
- All 24 new source/target feature frequency tables are generated without label/prediction leakage and pass finite-value, unique-subject, and band-fraction integrity checks.
- The final report contains direction, seed, preset, variant, target metrics, feature-domain BA/AUROC, spectrum statistics, provenance, and the interpretation limits above.
