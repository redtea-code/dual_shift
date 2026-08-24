# DS-038: Domain/Intensity GRL Factorial Audit

Status: **BLOCKED**

Plan: [DS-038 plan](../../plans/DS-038_GRL_FACTORIAL_MECHANISM.md)

Configuration: ds038_grl_factorial_scan_filtered_1p5t_mci_ad.yaml.

Data contract: scan-filtered ADNI 1.5T -> NACC 3T, MCI vs AD. Target labels are excluded from training and checkpoint selection; target-test metrics remain exploratory.

## 1. Two Artifact Generations

DS-038 now contains two non-poolable artifact generations:

1. **Historical performance matrix.** The original G0-G3 x seeds 42-46 execution produced 20/20 performance artifacts, but did not persist the registered mechanism diagnostics schema.
2. **Diagnostic-persistence rerun.** The updated runner persists the required diagnostics and completion status. At the interim inventory point, 9/20 cells were independently verified complete.

The interim values are not a completed remeasurement of the historical five-seed matrix. Do not pool their metrics, use interim two-seed values to overwrite historical five-seed values, or make a final mechanism claim from either generation alone.

## 2. Historical Five-Seed Performance Evidence

Registered matrix: G0-G3 x seeds 42-46. All variants retain source Fourier synthesis and attention consistency. G0 has neither GRL, G1 has domain GRL only, G2 has intensity GRL only, and G3 has both heads.

### 2.1 Target Test

Values are subject-level target balanced accuracy and AUC, mean +/- sample SD across seeds 42-46. T_test is an internal holdout previously used in historical work and is exploratory.

| Variant | Target BA mean +/- SD | Target AUC mean +/- SD | Mean paired delta BA vs G0 +/- SD |
|---|---:|---:|---:|
| G0 no_grl | 0.5697 +/- 0.0766 | 0.7295 +/- 0.0553 | - |
| G1 domain_only | 0.6038 +/- 0.0251 | 0.7541 +/- 0.0496 | +0.0341 +/- 0.0773 |
| G2 intensity_only | 0.5801 +/- 0.0371 | 0.7567 +/- 0.0581 | +0.0104 +/- 0.0722 |
| G3 both_grl | 0.6308 +/- 0.0618 | 0.7477 +/- 0.0472 | +0.0611 +/- 0.0746 |

Paired target BA deltas versus G0, in seed order 42, 43, 44, 45, 46:

- G1: +0.0838, -0.1024, +0.0677, +0.0724, +0.0488.
- G2: +0.0949, -0.0898, -0.0331, +0.0389, +0.0410.
- G3: +0.0724, -0.0448, +0.1129, +0.1429, +0.0220.

No cell is positive versus G0 in all five seeds: G1 is positive in 4/5 seeds, G2 in 3/5, and G3 in 4/5.

### 2.2 Source Test

Source-test is a training-domain safeguard and was not used for checkpoint selection; source-validation balanced accuracy remained the selector.

| Variant | Source-test BA mean +/- SD | Source-test AUC mean +/- SD | Mean paired delta BA vs G0 +/- SD |
|---|---:|---:|---:|
| G0 no_grl | 0.5893 +/- 0.0848 | 0.6466 +/- 0.0899 | - |
| G1 domain_only | 0.7000 +/- 0.0699 | 0.7415 +/- 0.0879 | +0.1107 +/- 0.1046 |
| G2 intensity_only | 0.6429 +/- 0.0740 | 0.7184 +/- 0.0804 | +0.0536 +/- 0.0695 |
| G3 both_grl | 0.6298 +/- 0.0902 | 0.6990 +/- 0.1140 | +0.0405 +/- 0.0957 |

These source-side changes are not UDA or causal evidence and do not override the target-side paired rule.

### 2.3 Historical Decision Boundary

The historical matrix is performance evidence only. It does not support retention of either head as an adopted module, a complementarity claim, a fresh confirmatory holdout, or scanner/manufacturer/field-strength causal language. The registered discriminator and feature diagnostics were absent from final summary JSON, so head-level mechanism attribution remained unverified.

## 3. Diagnostic-Persistence Interim Rerun

The updated runner saves mechanism diagnostics in both summary JSON and audit JSON, adds status JSON, and verifies that best checkpoint, summary, audit, config, and prediction artifacts exist before marking a run complete.

Each enabled-head record contains:

- head, epoch, step, and best-checkpoint flag;
- discriminator loss, accuracy, balanced accuracy, and AUROC;
- GRL coefficient;
- shared-encoder feature gradient norm;
- discriminator-parameter gradient norm.

A post-hoc, target-label-blind frozen-feature probe also records domain balanced accuracy, AUROC, an MMD proxy, and source/target feature norms. It is a diagnostic of feature separability, not by itself evidence of successful alignment.

### 3.1 Verified Interim Coverage

- Registered rerun matrix: 20 cells.
- Independently verified complete cells: 9/20.
- G0 no_grl: seeds 42, 43, 44.
- G1 domain_only: seeds 42, 43.
- G2 intensity_only: seeds 42, 43.
- G3 both_grl: seeds 42, 43.

G0 seed45-46, G1 seed44-46, G2 seed44-46, and G3 seed44-46 are not counted as complete in this interim report.

### 3.2 Interim Two-Seed Screen

The common-seed comparison is restricted to seeds 42 and 43. These values are descriptive only.

| Variant | Target BA mean +/- SD | Target AUC mean +/- SD | Source-test BA mean +/- SD | Source-test AUC mean +/- SD |
|---|---:|---:|---:|---:|
| G0 no_grl | 0.5853 +/- 0.0350 | 0.6763 +/- 0.1188 | 0.6250 +/- 0.0337 | 0.6845 +/- 0.0902 |
| G1 domain_only | 0.6601 +/- 0.0392 | 0.8114 +/- 0.0089 | 0.6786 +/- 0.0000 | 0.7806 +/- 0.0241 |
| G2 intensity_only | 0.5452 +/- 0.0136 | 0.7643 +/- 0.0507 | 0.6667 +/- 0.0505 | 0.7338 +/- 0.0397 |
| G3 both_grl | 0.5496 +/- 0.0051 | 0.7218 +/- 0.0054 | 0.5982 +/- 0.0042 | 0.6437 +/- 0.0421 |

Common-seed paired target BA deltas versus G0 are G1 +0.0717, +0.0778; G2 -0.0057, -0.0745; and G3 -0.0146, -0.0568.

The available frozen-feature probes do not establish stable domain confusion: G1 and G3 are lower than G0 in seed42 but not uniformly across the available seeds; G2 has the highest MMD proxy in both available seeds.

## 4. Protocol And Claim Boundary

For the audited cells, target labels were not used during training or checkpoint selection; target-test evaluation followed source-validation checkpoint selection; subject digests and configuration hashes are saved in audit JSON; and the frozen probe is post-hoc and target-label-blind.

The following claims remain unsupported:

- domain GRL is a stable improvement over G0;
- intensity GRL is a stable improvement over G0;
- the two heads are complementary;
- GRL consistently reduces feature separability;
- the result identifies scanner, manufacturer, or field-strength causality.

## 5. Required Next Action

1. Run or recover the missing diagnostic-rerun cells with one process per GPU and unique output roots.
2. Validate each cell's status and required artifact schema.
3. Recompute paired target/source results only on the common complete seed set 42-46.
4. Summarize head metrics at the selected checkpoint, including discriminator behavior, encoder/discriminator gradient norms, source clean/shift CE, feature discrepancy, and the frozen probe.
5. Update this result record only after all 20 diagnostic-rerun cells pass the artifact and protocol audit.

Until then, DS-038 remains **BLOCKED**.
