# DS-035: FMM three-seed ablation results

Status: COMPLETED exploratory baseline

## 1. Scope and completion

- Direction/task: ADNI 1.5T → NACC 3T, MCI vs AD.
- Seeds: 42, 43, 44.
- Matrix: B0-ref, B1-fmm, B1a-no-source-fft, B1b-no-attention, B1c-no-grl.
- Planned/completed cells: 15/15. No cell was deleted, replaced, or selected using target metrics.
- Primary endpoint: target subject-mean balanced accuracy. Target test is an internal frozen holdout that appeared in historical work; all target rows below are exploratory.

## 2. Protocol and boundaries

All variants use the independent FMM 10-convolution encoder, 50 epochs, source-only validation checkpoint selection, and the same scan-filtered source/target data contract. `T_adapt` exposes images and domain membership only; `T_test` is evaluated only after the source-selected checkpoint is restored.

The run audits record `target_label_access_during_training: false`, `target_labels_used_for_selection: false`, and `target_test_evaluation_after_fit: true` for every completed cell. Each cell retains `summary.json`, `audit.json`, `predictions.json`, `config.yaml`, and `best.pt`.

## 3. Target result table

| Variant | seed42 BA/AUC | seed43 BA/AUC | seed44 BA/AUC | BA mean ± SD | AUC mean ± SD | Δ BA vs B0 |
|---|---:|---:|---:|---:|---:|---:|
| B0-ref | 0.5403 / 0.6912 | 0.5767 / 0.7741 | 0.6310 / 0.6815 | 0.5827 ± 0.0456 | 0.7156 ± 0.0509 | — |
| B1-fmm | 0.6009 / 0.7748 | 0.6064 / 0.8190 | 0.6321 / 0.7870 | 0.6131 ± 0.0166 | 0.7936 ± 0.0225 | +0.0305 |
| B1a-no-source-fft | 0.6929 / 0.8260 | 0.5591 / 0.8344 | 0.6040 / 0.7487 | 0.6187 ± 0.0681 | 0.8031 ± 0.0474 | +0.0360 |
| B1b-no-attention | 0.6647 / 0.7964 | 0.5412 / 0.8185 | 0.6794 / 0.7571 | 0.6285 ± 0.0759 | 0.7907 ± 0.0311 | +0.0458 |
| B1c-no-grl | 0.5606 / 0.7142 | 0.5834 / 0.7707 | 0.5738 / 0.6090 | 0.5726 ± 0.0114 | 0.6980 ± 0.0825 | −0.0101 |

Mean differences are computed from matched seed-level BA values, not from separately rounded means. B1 is above B0 in all three seeds, but its seed44 increment is only +0.0011. B1a/B1b exhibit higher mean BA with substantially larger variance, so neither is a stable replacement claim.

## 4. Source-side diagnostics

| Variant | Source validation BA mean ± SD | Source test BA mean ± SD | Target best-epoch range |
|---|---:|---:|---:|
| B0-ref | 0.6560 ± 0.0169 | 0.5655 ± 0.0630 | 12–46 |
| B1-fmm | 0.6762 ± 0.0343 | 0.6171 ± 0.0506 | 22–49 |
| B1a-no-source-fft | 0.6643 ± 0.0129 | 0.5635 ± 0.0305 | 20–48 |
| B1b-no-attention | 0.6726 ± 0.0258 | 0.6369 ± 0.0449 | 20–46 |
| B1c-no-grl | 0.6714 ± 0.0214 | 0.5813 ± 0.0578 | 24–45 |

Source validation and source test rankings differ, so source-side outcomes cannot substitute for the registered exploratory target endpoint.

## 5. Component interpretation and limits

- Source Fourier synthesis: B1a has high mean target BA but is unstable across seeds; the seed42 high point does not establish a robust benefit from removing source FFT.
- Attention consistency: B1b also has high mean BA but large seed variability. This is insufficient evidence that attention is harmful or unnecessary.
- GRL: B1c is below B0 on average and below B1 in every seed except no seed shows a positive B1c advantage. This is an exploratory signal that the combined GRL path contributes under this implementation. B1c simultaneously disables domain and intensity heads, so it cannot identify which head matters.

## 6. Claim boundary and decision

This result supports an exploratory baseline conclusion only: complete FMM has a modest, directionally consistent subject-level target BA advantage over its FMM reference encoder in three seeds. It does not support scanner/field-strength causal language, a fresh-holdout claim, or adoption of B1a/B1b as superior methods.

Decision: close DS-035 as an exploratory baseline/component screen. DS-038 is required for domain versus intensity GRL attribution.

## 7. Provenance gaps

The per-run audits preserve configuration hashes, dataset hashes, subject digests, and upstream reference commit `580625cee5bfc1474fe8700e530ade07ac5e9776`. A separate main-branch E0 archive containing manifest hashes, environment lock, and source commit record was not produced; this limits reproducibility evidence but does not alter the recorded label-blind boundaries.
