# Plan 34 Raw Frequency Feature Skip Design

## 1. Status and Question

- Status: source-only robustness candidate and falsifiable implementation; not a target-performance result.
- Question: raw MRI spectra can distinguish ADNI/NACC more readily than deeper CNN maps. Does a bounded raw-frequency residual injected into an intermediate CNN stage change the learned representation and improve robustness across pre-registered source frequency environments?

The premise does not imply that dataset identity is disease information, scanner causality, or a useful target correction. Cross-cohort spectral differences can also reflect preprocessing, site, demographics, and disease composition. The skip branch is therefore not allowed to read target images, labels, domain identifiers, target metrics, or a target-derived prior.

## 2. Candidate Model

The baseline remains `layer5_pixel + original_capm(age, sex, education)`.

```text
x (MRI) -> CNN stem -> layer1 -> layer2 -> layer3 -> layer4 -> layer5 -> original CAPM -> classifier
    |                                            ^
    +-> rFFTN(x) -> compact raw spectral encoder -+
```

For an input volume `x`, the branch forms an rFFT before the CNN:

```text
F = rFFTN(x)
q = AdaptivePool(log(1 + |F|), cos(angle(F)), sin(angle(F)))
r = Project(Conv3D(q))
h' = h + a * RMS(h) / RMS(r) * Resize(r)
a = a_max * tanh(theta), |a| < a_max
```

The compact grid is fixed (`8 x 8 x 8` by default), so full MRI volume resolution does not create a second full-resolution backbone. The default injection point is `layer3`; `layer2` and `layer4` are pre-registered alternatives, not target-selected settings.

## 3. Explicit Audits

Each forward records:

- raw low/mid/high rFFT-band fractions;
- compact descriptor RMS;
- bounded skip strength;
- residual-to-feature relative RMS;
- non-identity fraction.

These demonstrate whether the raw branch has a measurable effect. They do not demonstrate that the effect improves diagnosis, identifies a scanner mechanism, or removes a domain shift.

## 4. Minimal Experiment

All runs use the frozen scan-filtered ADNI source split, same initialization, optimizer, epoch budget, source validation selector, and source frequency environments.

| ID | Variant | Question |
|---|---|---|
| R0 | `original_capm` | Does the current source baseline meet its source-environment reference? |
| R1 | `raw_frequency_skip` | Does a pre-CNN raw-frequency residual have nonzero, stable feature effect while preserving clean source validation? |
| R2 | `raw_frequency_skip`, shuffled raw descriptor | Is any R1 effect due to the matching sample spectrum rather than extra capacity? |

R2 is a required negative control before interpreting a source-side gain. It is deliberately not included in the first implementation because the runner must first establish a correct R1 audit path.

## 5. Gates

R1 can proceed to R2 only when:

1. `residual_relative_rms` and `nonidentity_fraction` are nonzero and stable across the source batches;
2. clean `S_val` CE/BA does not exceed the pre-registered R0 degradation tolerance;
3. per-environment CE/BA and worst-environment risk are reported for `original`, `lowpass`, `downsample_resample`, and `mild_blur`;
4. the raw-frequency audit and source manifest/checkpoint/config hashes are saved with the run.

No current `T_test` row may select the injection stage, grid size, residual cap, checkpoint, or promotion decision. A candidate that passes source-only gates is frozen before any new unread holdout is considered.

## 6. Configuration and Entry Point

```yaml
raw_frequency_skip:
  skip_stage: layer3
  spectral_grid: [8, 8, 8]
  hidden_channels: 32
  max_residual: 0.15
  gate_init: 0.1
  use_source_frequency_environments: true
  lowpass_kernel: 3
  blur_sigma: 0.8
```

Run the candidate only through the source-only path:

```text
<PYTHON> experiments/train_journal.py \
  --config_path <RESOLVED_CONFIG> \
  --direction ADNI_to_NACC \
  --variants original_capm raw_frequency_skip \
  --source-only
```

The repository change supplies the model, runner wiring, auditable metrics, and synthetic tests. It does not run the real-data experiment or claim a diagnosis or external-generalization improvement.
