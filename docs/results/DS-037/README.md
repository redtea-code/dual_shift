# DS-037: Result record

Status: COMPLETED

Plan: [DS-037 plan](../../plans/DS-037_AMPLITUDE_TRANSPORT_MECHANISM.md)

Code commit: `experiment/ds037-amplitude-audit` worktree based on `origin/main` at `b0231f8`; the executed implementation commit must be recorded when this worktree is committed.

Configuration: `ds037_amplitude_transport_3seed.yaml`, with the original DS-036 configuration left unchanged.

Data contract: scan-filtered ADNI 1.5T → NACC 3T, MCI vs AD, `ADNI_to_NACC`; layer4_pixel CAPM backbone, source-only validation selector, subject-disjoint target adaptation/test split.

## Protocol revision

The original registered plan specifies seeds 42–46. The approved execution revision reduced the matrix to seeds 42–44 to align with the completed DS-035 matrix. The original plan remains unchanged; all conclusions here are three-seed exploratory mechanism evidence and do not complete the original five-seed registration.

## 1. Command

The 18 cells were run as AT0–AT5 × seeds 42/43/44. The five seed44 retry cells were written to an isolated retry output root after the first multi-seed processes stopped before producing metrics.

```text
CUDA_VISIBLE_DEVICES=<gpu> python -m experiments.train_target_style_uda \
  --config ds037_amplitude_transport_3seed.yaml \
  --direction ADNI_to_NACC \
  --output-root outputs/ds037_amplitude_transport_3seed[_retry] \
  --variants <AT variant> --seeds <seed> --device cuda
```

Checkpoint selection used source-validation balanced accuracy only. No target-test result was used for tuning, early stopping, variant deletion, or phase/strength selection.

## 2. Completion and verified evidence

- Planned/completed units: **18/18**.
- Every unit has `metrics.json`, `best_checkpoint.pt`, a 50-epoch history, resolved phase/strength, and protocol audit fields.
- Every non-control transport unit has 50 mechanism diagnostic records. AT0 is the exact no-transport control and appropriately has no transport diagnostics.
- All completed audits record `target_label_access_before_final_evaluation: false`, `target_metric_access_before_final_evaluation: false`, no target covariate/environment construction for `T_adapt`, and `target_test_used_for_selection: false`.
- Source-label behavior is represented by source validation/test metrics and clean/mixed prediction agreement diagnostics.

## 3. Results

Target test is an internal holdout already used in historical work. It is therefore exploratory, not a fresh confirmatory endpoint. Values below are subject-level balanced accuracy and AUC, mean ± sample SD across seeds 42–44.

| Variant | Strength | Phase | Target BA mean ± SD | Target AUC mean ± SD | Δ BA vs AT0 mean ± SD |
|---|---:|---|---:|---:|---:|
| AT0 `capm` | 0.00 | none | 0.6488 ± 0.0391 | 0.7380 ± 0.0070 | — |
| AT1 `source_phase_a025` | 0.25 | source | 0.6184 ± 0.0431 | 0.7189 ± 0.0401 | −0.0304 ± 0.0296 |
| AT2 `source_phase_a050` | 0.50 | source | 0.6227 ± 0.0038 | 0.7458 ± 0.0191 | −0.0261 ± 0.0417 |
| AT3 `source_phase_a100` | 1.00 | source | 0.5925 ± 0.0283 | 0.7361 ± 0.0149 | −0.0563 ± 0.0180 |
| AT4 `target_phase_a050` | 0.50 | target | 0.6419 ± 0.0341 | 0.7327 ± 0.0028 | −0.0069 ± 0.0508 |
| AT5 `target_phase_a100` | 1.00 | target | 0.6283 ± 0.0426 | 0.7503 ± 0.0136 | −0.0205 ± 0.0657 |

Seed-level target BA values were: AT0 `0.6931, 0.6192, 0.6341`; AT1 `0.6485, 0.5691, 0.6377`; AT2 `0.6192, 0.6223, 0.6267`; AT3 `0.6190, 0.5627, 0.5960`; AT4 `0.6510, 0.6705, 0.6041`; AT5 `0.6052, 0.6024, 0.6775` for seeds 42, 43, 44 respectively.

## 4. Mechanism diagnostics

The implementation records resolved strength/phase, amplitude L1 discrepancy, transported-source CE, consistency loss, clean/mixed prediction agreement, feature mean/std/norm/finite checks, and CAPM gate/effective-field summaries. These diagnostics are stored in each unit's `metrics.json` under `history` and `mechanism_diagnostics`.

The primary directional signal is negative: source-phase transport at α=1.00 is below AT0 in all three seeds (paired Δ BA `−0.0741, −0.0565, −0.0381`). AT1 and AT2 are also negative on average but not seed-consistent: AT1 is positive only in seed44, and AT2 is positive only in seed43. Target-phase diagnostics are not directionally credible: AT4 is positive in seed43 and negative in seeds42/44; AT5 is positive only in seed44. No tested transport variant shows a three-seed positive BA signal over AT0.

## 5. Decision rule and claim boundary

Under the registered exploratory decision rule, no amplitude/phase transport mechanism is directionally credible as a positive improvement: a positive signal would require consistency across all three seeds, preserved source-label behavior, and no explanation by sensitivity/specificity shifts. AT3 instead provides a consistent negative signal under source phase at full strength.

Decision: **NO-GO for selecting a transport strength or phase mode from this exploratory audit.** This does not authorize a fresh confirmatory holdout and does not support scanner-, manufacturer-, or field-strength-causal language. A positive AUC or isolated seed-level BA does not override the paired, three-seed BA rule.

## 6. Provenance and limitations

The run-level outputs preserve resolved configuration, split subjects, checkpoint epoch, metrics, predictions, and protocol boundary fields. The five seed44 retry outputs are separate from the initial incomplete directories; the retry provenance is recorded by the independent command logs. A standalone main-branch E0 manifest/environment archive and the exact committed execution hash were not available when this report was drafted. The three-seed revision is exploratory and lower-powered than the original five-seed registration.

## 7. Unexecuted controls and next action

No additional full-target adaptation protocol, fresh external label-blind holdout, or independent domain/intensity GRL factorial was executed under DS-037. DS-038 remains the registered follow-up for separating domain and intensity GRL effects.
