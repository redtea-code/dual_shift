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

## 4. Decision boundary

The registered exploratory rule requires a directionally consistent signal across seeds. No factorial cell is positive versus G0 in all five seeds. G1 is positive in four of five seeds, G2 in three of five, and G3 in four of five; therefore no isolated head or complementarity claim is supported by the paired BA rule. The higher G3 mean is not sufficient evidence of complementarity because its paired differences are not seed-consistent.

This result does not support retaining a GRL head as an adopted module, does not authorize a fresh confirmatory holdout, and does not support scanner-, manufacturer-, or field-strength-causal language.

## 5. Required-diagnostics gap

The registered plan requires per-epoch and checkpoint-level domain/intensity discriminator loss, accuracy/AUC/balanced accuracy, GRL coefficient, gradient norm, source clean/shift CE, and feature discrepancy diagnostics. The implementation computed some head metrics in memory, but the final `summary.json` files do not contain a persisted `mechanism_diagnostics` field. Consequently, this is **not** a complete mechanism audit: the performance matrix and protocol boundary are verified, while discriminator/gradient mechanism evidence is unverified.

The correct status is BLOCKED pending a diagnostic persistence fix and rerun of the affected matrix. The current 20 completed runs must not be presented as a fully verified DS-038 mechanism result.

## 6. Provenance and next action

The outputs preserve resolved configs, source-selected checkpoint epochs, metrics, predictions, audit files, subject digests, and per-run configuration hashes. The next action is to persist the required diagnostics, run targeted tests against the saved schema, and rerun G0–G3 × seeds 42–46 before making a final mechanism attribution decision.
