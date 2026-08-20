# DS-035: Result record

Status: COMPLETED

Plan: [DS-035 plan](../../plans/DS-035_FMM_BASELINE.md)

Code commit: `9d3b61e9eeddb21da7f6e7f3dfdd5549fe933222` for the original FMM worktree; the seed42–44 artifacts retain their own audit commits and configuration hashes.

Configuration: `fmm_baseline_scan_filtered_1p5t_mci_ad.yaml`, with seed-specific copies for 42, 43, and 44.

Manifest / data version: scan-filtered ADNI 1.5T → NACC 3T, MCI vs AD; the run-level audit files preserve dataset hashes and subject digests, but no independently archived main-branch E0 manifest record accompanies this experiment.

## 1. Command

Each cell was trained for 50 epochs using the independent FMM runner with source-validation balanced accuracy as the sole checkpoint selector:

```text
python -m experiments.train_fmm_baseline --config_path configs_multiseed/fmm_seed<seed>.yaml --direction ADNI_to_NACC --variant <variant> --output-dir outputs/fmm_multiseed/ADNI_to_NACC/seed_<seed>/<variant> --device cuda:0
```

## 2. Results

The registered B0–B1c matrix is complete for seeds 42, 43, and 44: 15/15 runs. Detailed seed-level, mean ± SD, source-side, target-side, and audit results are in [the three-seed report](3SEED_ABLATION_2026-08-19.md). The exploratory target BA summaries are:

| variant | Target BA mean ± SD | Target AUC mean ± SD | Δ BA vs B0 |
|---|---:|---:|---:|
| B0-ref | 0.5827 ± 0.0456 | 0.7156 ± 0.0509 | — |
| B1-fmm | 0.6131 ± 0.0166 | 0.7936 ± 0.0225 | +0.0305 |
| B1a-no-source-fft | 0.6187 ± 0.0681 | 0.8031 ± 0.0474 | +0.0360 |
| B1b-no-attention | 0.6285 ± 0.0759 | 0.7907 ± 0.0311 | +0.0458 |
| B1c-no-grl | 0.5726 ± 0.0114 | 0.6980 ± 0.0825 | −0.0101 |

## 3. Verified evidence

- All 15 runs contain `summary.json`, `audit.json`, `predictions.json`, `config.yaml`, and `best.pt`.
- Each audit records no target-label access during training, no target-label checkpoint selection, and post-fit target evaluation.
- Source train/validation/test, target adapt/test, and source/target subject sets are checked as disjoint in the runner audit.
- B1 exceeds its matched B0 target BA in each of the three completed seeds, with a mean paired difference of +0.0305.

## 4. Not verified

- A fresh label-unread confirmatory target holdout. The current T_test appeared in historical experiments, so target results remain exploratory.
- Faithful reproduction of every underspecified upstream FMM detail.
- Independent domain versus intensity GRL attribution: B1c removes both heads jointly.
- Full main-branch E0/E2 provenance archive, including standalone manifest hash, environment record, and committed resolved-config bundle.

## 5. Claim boundary

The completed matrix supports only an exploratory observation that full FMM had a small, directionally consistent target-BA advantage over its matched FMM reference encoder in these three training seeds. It does not establish scanner-, manufacturer-, or field-strength-causal correction, nor does it establish that any ablated component is universally necessary. B1a and B1b have larger mean target BA but much larger seed variability; they are not stable improvements.

## 6. Decision and next action

Decision: COMPLETE as an exploratory baseline and component screen; no confirmatory promotion.

Next action: execute DS-038, the registered domain/intensity GRL factorial audit, to separate the two heads that B1c removed jointly.
