# DS-038: Domain/Intensity GRL Factorial Audit

Status: **BLOCKED — interim diagnostic rerun only**

Plan: [DS-038 plan](../../plans/DS-038_GRL_FACTORIAL_MECHANISM.md)

Configuration: `ds038_grl_factorial_scan_filtered_1p5t_mci_ad.yaml`.

Data contract: scan-filtered ADNI 1.5T → NACC 3T, MCI vs AD, `ADNI_to_NACC`. Target labels are excluded from training and checkpoint selection; target-test metrics are exploratory.

## 1. Scope and verified completion

- Registered matrix: G0–G3 × seeds 42–46 (20 cells).
- Current verified completion: **9/20 cells**.
- Complete cells currently available for the common two-seed comparison:
  - G0 `no_grl`: seeds 42, 43, 44.
  - G1 `domain_only`: seeds 42, 43.
  - G2 `intensity_only`: seeds 42, 43.
  - G3 `both_grl`: seeds 42, 43.
- G0 seed45–46, G1 seed44–46, G2 seed44–46, and G3 seed44–46 do not currently have a complete, independently verifiable artifact set in this workspace.
- A cell is counted only when `status.json` is `complete` and all of `best.pt`, `summary.json`, `audit.json`, `config.yaml`, and `predictions.json` exist.
- GPU training processes were stopped after this inventory. No incomplete cell was promoted to a complete result.

This is an interim report, not a completed five-seed factorial analysis.

## 2. Diagnostic protocol

The current runner persists mechanism diagnostics in both `summary.json` and `audit.json`. Each head record contains:

- `head`, `epoch`, `step`, and `is_best_checkpoint`;
- discriminator loss, accuracy, balanced accuracy, AUROC;
- GRL coefficient;
- shared encoder feature gradient norm;
- discriminator parameter gradient norm.

Each complete cell also contains a frozen-feature domain probe evaluated after loading the source-validation-selected checkpoint. The probe records domain balanced accuracy, AUROC, mean discrepancy (MMD proxy), and source/target feature norms. It is diagnostic only and does not use target labels or affect selection.

Verified schema observations:

- G0 cells contain zero domain and zero intensity head records.
- G3 cells contain both domain and intensity records.
- G0/G3 complete cells contain best-checkpoint flags and frozen-feature probe output.
- The probe is not evidence that raw target alignment has succeeded by itself; it measures the separability of the collected source and target-style feature samples.

## 3. Interim performance results

The following values are descriptive only and are not five-seed estimates. The two-seed comparison uses the common seeds 42 and 43; G0 additionally has seed44.

| Variant | Seeds used | Target BA mean ± sample SD | Target AUC mean ± sample SD | Source-test BA mean ± sample SD | Source-test AUC mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| G0 `no_grl` | 42,43 | 0.5853 ± 0.0350 | 0.6763 ± 0.1188 | 0.6250 ± 0.0337 | 0.6845 ± 0.0902 |
| G1 `domain_only` | 42,43 | 0.6601 ± 0.0392 | 0.8114 ± 0.0089 | 0.6786 ± 0.0000 | 0.7806 ± 0.0241 |
| G2 `intensity_only` | 42,43 | 0.5452 ± 0.0136 | 0.7643 ± 0.0507 | 0.6667 ± 0.0505 | 0.7338 ± 0.0397 |
| G3 `both_grl` | 42,43 | 0.5496 ± 0.0051 | 0.7218 ± 0.0054 | 0.5982 ± 0.0042 | 0.6437 ± 0.0421 |

Seed-level target results:

- G0 BA: seed42 `0.5606`, seed43 `0.6100` (seed44 additionally `0.5200`).
- G1 BA: seed42 `0.6323`, seed43 `0.6878`.
- G2 BA: seed42 `0.5548`, seed43 `0.5356`.
- G3 BA: seed42 `0.5460`, seed43 `0.5532`.

Common-seed paired target BA differences versus G0 are:

- G1: `+0.0717, +0.0778`; mean `+0.0748 ± 0.0043`.
- G2: `−0.0057, −0.0745`; mean `−0.0401 ± 0.0486`.
- G3: `−0.0146, −0.0568`; mean `−0.0357 ± 0.0298`.

These are two-seed screening statistics, not registered five-seed evidence.

## 4. Interim mechanism observations

Frozen-feature probe results for the complete cells were:

- G0 seed42: domain BA `0.8579`, AUROC `0.9308`, MMD proxy `0.3232`.
- G0 seed43: domain BA `0.8118`, AUROC `0.8859`, MMD proxy `0.1044`.
- G0 seed44: domain BA `0.8574`, AUROC `0.9361`, MMD proxy `0.2871`.
- G1 seed42: domain BA `0.8165`, AUROC `0.9000`, MMD proxy `0.2500`.
- G1 seed43: domain BA `0.8272`, AUROC `0.9251`, MMD proxy `0.2667`.
- G2 seed42: domain BA `0.8606`, AUROC `0.9398`, MMD proxy `0.5166`.
- G2 seed43: domain BA `0.8112`, AUROC `0.9520`, MMD proxy `0.8672`.
- G3 seed42: domain BA `0.8275`, AUROC `0.9252`, MMD proxy `0.1987`.
- G3 seed43: domain BA `0.8473`, AUROC `0.9314`, MMD proxy `0.1571`.

The G1 and G3 probe metrics are lower than G0 in seed42 but not uniformly across the available seeds. G2 has the highest MMD proxy in both available seeds. The probe therefore does not establish stable effective domain confusion or a monotonic alignment mechanism.

The G3 records contain 6,000–6,100 per-step records per head and 120–122 best-checkpoint records per head. This indicates that diagnostics are now persisted and checkpoint-bound, but record volume is implementation-level evidence rather than proof of causal mechanism success.

## 5. Protocol and provenance audit

For the complete cells currently inspected:

- target labels were not used during training;
- target labels were not used for checkpoint selection;
- target-test evaluation occurred after source-validation checkpoint selection;
- source/target subject digests and configuration hashes are stored in `audit.json`;
- the frozen probe is post-hoc and label-blind.

Because the matrix is incomplete, these audit statements do not authorize a final factorial claim.

## 6. Decision boundary

No adoption or mechanism attribution decision is made from this interim evidence. In particular, the current artifacts do **not** support any of the following claims:

- domain GRL is superior to G0;
- intensity GRL is superior to G0;
- the two GRL heads are complementary;
- GRL reduces source–target feature separability consistently across seeds;
- the result generalizes to scanner, manufacturer, or field-strength causality.

The registered decision remains **BLOCKED** until G0–G3 × seeds42–46 are complete with the same diagnostic schema and paired seed alignment.

## 7. Required next action

1. Re-run the 15 missing cells with one process per GPU and unique output directories.
2. Verify every cell using `status.json`, the five required artifacts, and diagnostic schema validation.
3. Recompute target/source metrics and paired differences only on the common seed set 42–46.
4. Summarize head diagnostics at the selected checkpoint, including discriminator metrics, encoder/discriminator gradient norms, feature discrepancies, and probe results.
5. Update this report only after all 20 cells pass the artifact and protocol audit.
