# DS-039: CAPM-conditioned biology-preserving residual adaptation pilot

Status: **RUNNING / PILOT**

Plan: [DS-039 plan](../../plans/DS-039_CAPM_CONDITIONED_RESIDUAL_ADAPTATION.md)

## 1. Executive summary

DS-039 evaluates whether a bounded, channel-wise residual correction applied after a
frozen CAPM representation can reduce batch-sensitive variation while preserving
diagnostic utility. The registered model is `original_capm` with the
`layer4_pixel` feature preset. The residual module has no learned parameters and
uses a fixed `max_strength=0.25`.

The primary protocol is ADNI→NACC. NACC→ADNI was run as a prespecified extension
using the same task, model, feature preset, and two seeds (42 and 43). Both
directions have complete real-data pilot artifacts. The results provide pilot
 evidence only; they do not establish a stable performance gain or scanner/biology
causality.

## 2. Question and evaluated path

The question is:

> Can batch-sensitive variation be reduced after CAPM conditioning while the
> CAPM-adjusted diagnostic-support representation remains usable?

The evaluated path is:

```text
F4 -> original CAPM(F4, z_ref) -> B4
   -> bounded channel-wise residual correction -> classifier
```

The residual correction uses source statistics and image-only, subject-disjoint
`T_adapt` statistics. It is applied after CAPM conditioning and is not a learned
alignment discriminator.

## 3. Protocol and registered cells

| Item | Locked value |
|---|---|
| Task | MCI vs AD, scan-filtered |
| Primary direction | ADNI→NACC |
| Extension direction | NACC→ADNI |
| Backbone | `original_capm` |
| Feature preset | `layer4_pixel` |
| Seeds | 42, 43 |
| Checkpoint | Frozen source checkpoint |
| Selection | Source-validation-only checkpoint selection |
| Residual bound | `max_strength=0.25` |
| Adaptation population | Earliest scan per subject from unlabeled `T_adapt` |
| Target split | Subject-disjoint `T_adapt` and internal `T_test` |

Registered cells:

| ID | Report field | Role |
|---|---|---|
| RA0 | `source_control`, `target_control` | Matched original-CAPM control |
| RA1 | `source_adapted_diagnostic`, `target_adapted` | CAPM-conditioned residual adaptation |
| RA-D | `source_adapted_diagnostic` | Source-side preservation diagnostic; never a selector |

## 4. Audit and provenance boundary

The statistics stage records `target_labels_read=false` and
`target_metrics_read=false`. The evaluator records
`target_labels_read_for_adaptation=false`, while target labels are read only for
the final exploratory `T_test` report. Target-test results must therefore be
interpreted as exploratory because the holdout has appeared in historical work.

Every seed has the required files:

```text
target_split.json
residual_stats.json
report.json
build_stats.log
run_pilot.log
```

The statistics artifacts contain source/target channel moments, discrepancy
vectors, a fixed source-only CAPM reference table, source split provenance, and
checkpoint SHA-256 values. All recorded residual diagnostics are finite.

## 5. Target-test results: ADNI→NACC primary pilot

Values are subject-aggregated metrics. Deltas are RA1 minus RA0.

| Seed | RA0 BA | RA1 BA | Δ BA | RA0 AUROC | RA1 AUROC | Δ AUROC | RA0 Macro-F1 | RA1 Macro-F1 | Δ Macro-F1 | Δ Sensitivity | Δ Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6305 | 0.6453 | +0.0148 | 0.7549 | 0.7559 | +0.0010 | 0.6362 | 0.6551 | +0.0189 | +0.0353 | −0.0056 |
| 43 | 0.5384 | 0.5442 | +0.0059 | 0.7795 | 0.7802 | +0.0007 | 0.4840 | 0.4948 | +0.0108 | +0.0118 | 0.0000 |
| Mean | 0.5844 | 0.5948 | +0.0104 | 0.7672 | 0.7681 | +0.0009 | 0.5601 | 0.5750 | +0.0149 | +0.0235 | −0.0028 |

The target BA and AUROC changes are positive in both seeds, but the effect is
small. The source-side paired BA change is +0.0140 on average; specificity falls
by 0.0308 while sensitivity rises by 0.0588, so the BA change should not be read
without the operating-point shift.

## 6. Target-test results: NACC→ADNI extension

| Seed | RA0 BA | RA1 BA | Δ BA | RA0 AUROC | RA1 AUROC | Δ AUROC | RA0 Macro-F1 | RA1 Macro-F1 | Δ Macro-F1 | Δ Sensitivity | Δ Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6307 | 0.6435 | +0.0128 | 0.7570 | 0.7573 | +0.0003 | 0.6383 | 0.6530 | +0.0147 | +0.0256 | 0.0000 |
| 43 | 0.5260 | 0.5260 | 0.0000 | 0.6254 | 0.6254 | 0.0000 | 0.5205 | 0.5205 | 0.0000 | 0.0000 | 0.0000 |
| Mean | 0.5784 | 0.5848 | +0.0064 | 0.6912 | 0.6914 | +0.0002 | 0.5794 | 0.5867 | +0.0073 | +0.0128 | 0.0000 |

The extension result is driven by seed 42. Seed 43 is unchanged on every listed
target metric. This is evidence of a small, seed-sensitive pilot effect rather
than a stable directional improvement.

## 7. Source-test preservation

| Direction | Mean RA0 BA | Mean RA1 BA | Δ BA | Mean RA0 AUROC | Mean RA1 AUROC | Δ AUROC | Δ Sensitivity | Δ Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ADNI→NACC | 0.6685 | 0.6825 | +0.0140 | 0.7697 | 0.7678 | −0.0019 | +0.0588 | −0.0308 |
| NACC→ADNI | 0.7451 | 0.7451 | 0.0000 | 0.8617 | 0.8599 | −0.0018 | 0.0000 | 0.0000 |

Source BA is not degraded on average, but ADNI→NACC shows a sensitivity increase
and specificity decrease. This supports a preservation audit interpretation, not
a claim that the residual is biologically neutral.

## 8. Residual mechanism diagnostics

| Direction | Seed | Correction RMS | Residual RMS | Effective strength | Gate activity | Identity loss |
|---|---:|---:|---:|---:|---:|---:|
| ADNI→NACC | 42 | 0.03188 | 0.22321 | 0.25 | 1.0 | 0.000573 |
| ADNI→NACC | 43 | 0.01876 | 0.14656 | 0.25 | 1.0 | 0.000276 |
| NACC→ADNI | 42 | 0.03642 | 0.26812 | 0.25 | 1.0 | 0.000561 |
| NACC→ADNI | 43 | 0.03734 | 0.27978 | 0.25 | 1.0 | 0.000705 |

The correction is bounded and finite in all four runs. However, `residual_rms`
is a magnitude diagnostic, not a direct before/after source-target discrepancy
comparison. The current pilot therefore does not prove that the source-target
feature discrepancy decreased. A post-hoc feature discrepancy artifact remains a
recommended follow-up before any stronger mechanism claim.

## 9. Decision-rule assessment

| Criterion | Assessment |
|---|---|
| Target BA/AUROC not lower than matched control | Met numerically in both directions; effect is small and not uniformly present at seed level for NACC→ADNI |
| Source/target residual discrepancy decreases | Not fully established; current artifacts provide residual magnitude and channel statistics but not a paired before/after discrepancy summary |
| Source diagnosis and CAPM-anchor behavior preserved | Partially supported by source metrics; explicit CAPM-anchor drift tolerance is not yet archived |
| Correction finite and bounded | Met; effective strength is 0.25 and all recorded diagnostics are finite |
| BA gain not explained only by sensitivity/specificity movement | Not fully established; ADNI→NACC includes a sensitivity/specificity trade-off |

Overall decision: **PILOT INCONCLUSIVE / NO-GO FOR A FORMAL ADAPTATION CLAIM**.
The results justify retaining the implementation as a diagnostic pilot, but not
promoting it to a validated BioCAPM-DA result or using scanner, manufacturer,
field-strength, or causal-biology language.

## 10. Artifact hashes

SHA-256 hashes of the principal artifacts are listed below.

| Direction | Seed | `target_split.json` | `residual_stats.json` | `report.json` |
|---|---:|---|---|---|
| ADNI→NACC | 42 | `c772e7c30727021efdf7ba741c2909bf122f59818e118704539d0d5c3c45c42e` | `7fa93a3b11c415932b6e1aca4a3b588a32afba6a2f726e5f46d91473a10c40ea` | `fcd32730e4bccb7dda4549aa0892a72ffaa1d4d6827f7752f9abae3ba9a57743` |
| ADNI→NACC | 43 | `53c76d4a8fab1d0e647dcc27eaf5d6a9860959c98f20ec99dbb32022fe60eb4d` | `067cd6f1150ecacdab35183df5763659c0bc481102341896f79503339aa03418` | `7cbb24a1abdd9b35e2cc61b500bc3715597c0b679fc570852f027698417f7c46` |
| NACC→ADNI | 42 | `80d1b73c15617bc521ba6130c27764e7292eccb86408759abf6aa0141b3df1e1` | `f1d53fa7e623ba690b93638426bd864ffe12a5c861edf259cfcaa219f2ffef06` | `1cca3c310a44aa72025afe7edbe87bfab3a183301a4d8be2bd7c1552c68414a9` |
| NACC→ADNI | 43 | `b994c7ab72e6a76c699d48b88cab3c1920106eb042994fbc57758f9b2353c351` | `aadf09be1e6f0d44e66dd3c03fea49c3a389ba8a119cffc3059976eda0969a41` | `8e3a0d802c93a872b7955af606a558e33f00e82e98515a903992bf8e60994ec2` |

## 11. Recommended next action

1. Preserve this report and the four pilot artifact sets as the DS-039 evidence
   record.
2. Add an explicit paired before/after feature-discrepancy summary, including
   channel mean/covariance or CORAL/MMD-style distances.
3. Archive explicit CAPM-anchor drift and finite-value checks at report level.
4. Keep DS-039 at `RUNNING / PILOT` unless those audits satisfy the locked
   decision rule.
5. Do not expand to cross-scale, low-rank, learned-discriminator, or strength
   sweeps until the narrow pilot has mechanism-level evidence.
