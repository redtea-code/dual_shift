# DS-038: Result record

Status: BLOCKED

Plan: [DS-038 plan](../../plans/DS-038_GRL_FACTORIAL_MECHANISM.md)

Configuration: `ds038_grl_factorial_scan_filtered_1p5t_mci_ad.yaml`.

Data contract: scan-filtered ADNI 1.5T → NACC 3T, MCI vs AD, `ADNI_to_NACC`; the same source-only selector and internal exploratory target holdout contract as DS-035.

## 1. Scope and completion

- Registered matrix: G0–G3 × seeds 42–46.
- Real-data execution: 20/20 cells completed with exit code 0.
- G0 `no_grl`: domain GRL 0, intensity GRL 0.
- G1 `domain_only`: domain GRL 1, intensity GRL 0.
- G2 `intensity_only`: domain GRL 0, intensity GRL 1.
- G3 `both_grl`: domain GRL 1, intensity GRL 1.
- All cells retain `best.pt`, `summary.json`, `audit.json`, `config.yaml`, and `predictions.json`.

## 2. Protocol and audit boundary

All variants retain source Fourier synthesis and attention consistency. Target adaptation uses image/domain membership only; target labels are not used during training or checkpoint selection. Target test is evaluated after source-validation checkpoint selection and remains exploratory because the holdout has appeared in historical work.

The 20 run-level audits record `target_label_access_during_training: false` and `target_labels_used_for_selection: false`. No target-test tuning, variant deletion, or full-target protocol extension was performed.

## 3. Target results

Values are subject-level target balanced accuracy and AUC, mean ± sample SD across seeds 42–46. Target results are exploratory.

| Variant | Target BA mean ± SD | Target AUC mean ± SD | Mean paired Δ BA vs G0 ± SD |
|---|---:|---:|---:|
| G0 `no_grl` | 0.5697 ± 0.0766 | 0.7295 ± 0.0553 | — |
| G1 `domain_only` | 0.6038 ± 0.0251 | 0.7541 ± 0.0496 | +0.0341 ± 0.0773 |
| G2 `intensity_only` | 0.5801 ± 0.0371 | 0.7567 ± 0.0581 | +0.0104 ± 0.0722 |
| G3 `both_grl` | 0.6308 ± 0.0618 | 0.7477 ± 0.0472 | +0.0611 ± 0.0746 |

Seed-level target BA in seed order 42, 43, 44, 45, 46:

- G0: `0.5259, 0.7026, 0.5658, 0.5384, 0.5158`
- G1: `0.6097, 0.6003, 0.6335, 0.6109, 0.5647`
- G2: `0.6208, 0.6128, 0.5327, 0.5773, 0.5569`
- G3: `0.5983, 0.6579, 0.6787, 0.6813, 0.5378`

Paired Δ BA versus G0: G1 `+0.0838, −0.1024, +0.0677, +0.0724, +0.0488`; G2 `+0.0949, −0.0898, −0.0331, +0.0389, +0.0410`; G3 `+0.0724, −0.0448, +0.1129, +0.1429, +0.0220`.

## 4. Source-test results

Source-test metrics are reported as a training-domain safeguard and were not used for checkpoint selection; source validation balanced accuracy remained the selector. Values are subject-level source-test BA and AUC, mean ± sample SD across seeds 42–46.

| Variant | Source-test BA mean ± SD | Source-test AUC mean ± SD | Mean paired Δ BA vs G0 ± SD |
|---|---:|---:|---:|
| G0 `no_grl` | 0.5893 ± 0.0848 | 0.6466 ± 0.0899 | — |
| G1 `domain_only` | 0.7000 ± 0.0699 | 0.7415 ± 0.0879 | +0.1107 ± 0.1046 |
| G2 `intensity_only` | 0.6429 ± 0.0740 | 0.7184 ± 0.0804 | +0.0536 ± 0.0695 |
| G3 `both_grl` | 0.6298 ± 0.0902 | 0.6990 ± 0.1140 | +0.0405 ± 0.0957 |

Seed-level source-test BA in seed order 42, 43, 44, 45, 46:

- G0: `0.4881, 0.6190, 0.6131, 0.7024, 0.5238`
- G1: `0.7440, 0.7560, 0.5952, 0.7440, 0.6607`
- G2: `0.6250, 0.6667, 0.5595, 0.7560, 0.6071`
- G3: `0.6964, 0.6190, 0.6131, 0.7262, 0.4940`

Paired source-test BA differences versus G0 are: G1 `+0.2560, +0.1369, −0.0179, +0.0417, +0.1369`; G2 `+0.1369, +0.0476, −0.0536, +0.0536, +0.0833`; G3 `+0.2083, 0.0000, 0.0000, +0.0238, −0.0298`. These source-test changes are not causal evidence and do not override the target-side exploratory decision rule.

## 5. Decision boundary

The registered exploratory rule requires a directionally consistent signal across seeds. No factorial cell is positive versus G0 in all five seeds. G1 is positive in four of five seeds, G2 in three of five, and G3 in four of five; therefore no isolated head or complementarity claim is supported by the paired BA rule. The higher G3 mean is not sufficient evidence of complementarity because its paired differences are not seed-consistent.

This result does not support retaining a GRL head as an adopted module, does not authorize a fresh confirmatory holdout, and does not support scanner-, manufacturer-, or field-strength-causal language.

## 6. Required-diagnostics gap

The registered plan requires per-epoch and checkpoint-level domain/intensity discriminator loss, accuracy/AUC/balanced accuracy, GRL coefficient, gradient norm, source clean/shift CE, and feature discrepancy diagnostics. The implementation computed some head metrics in memory, but the final `summary.json` files do not contain a persisted `mechanism_diagnostics` field. Consequently, this is **not** a complete mechanism audit: the performance matrix and protocol boundary are verified, while discriminator/gradient mechanism evidence is unverified.

The correct status is BLOCKED pending a diagnostic persistence fix and rerun of the affected matrix. The current 20 completed runs must not be presented as a fully verified DS-038 mechanism result.

## 7. Provenance and next action

The outputs preserve resolved configs, source-selected checkpoint epochs, metrics, predictions, audit files, subject digests, and per-run configuration hashes. The next action is to persist the required diagnostics, run targeted tests against the saved schema, and rerun G0–G3 × seeds 42–46 before making a final mechanism attribution decision.
