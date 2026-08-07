# IE-CAPM paired experiment plan

## Status

- Date: 2026-08-07
- Status: planned; no outcome claim is authorized by this plan.
- Protocol alignment: APIC v3_2 revision 4 data, split, direction, and
  subject-level evaluation conventions.
- Candidate: `Model/backbone/evidence_calibrated_capm.py`
- Mechanism note: `review/analysis/31_ie_capm_self_review_and_implementation_2026-08-07.md`
- Data boundary: `docs/SCAN_AWARE_DATA_REALITY_AND_CLAIM_BOUNDARY.md`

## Question and claim boundary

Question: does image-evidence calibration improve the reliability of CAPM's
table-conditioned MRI fusion, beyond an equally trained CAPM control?

This is not a causal-adjustment, scanner-correction, or domain-generalization
claim.  It must not treat clinical variables (MMSE, CDRSB, ADAS11, FAQ) as
generic nuisance variables.  The primary claim, if earned, is a predictive and
mechanistic fusion claim: table modulation is selectively retained where a
table-free MRI stream supports it.

## Frozen comparison matrix

All rows use the same subject-level split manifests, MRI preprocessing,
backbone depth, classifier head, optimizer schedule, augmentation policy,
epoch budget, checkpoint rule, and random seeds.  The IE-CAPM architecture is
not claimed to reproduce APIC v3_2 performance; the alignment is at the data
protocol and evaluation level, while every row in this comparison is
capacity-matched to the other rows.

| ID | Model | Purpose |
| --- | --- | --- |
| B0 | image-only ResNet | establish image contribution |
| B1 | CAPM | primary performance baseline |
| B2 | IE-CAPM, `force_capm=True` | exact ungated control using the IE-CAPM implementation and parameters |
| P1 | IE-CAPM | proposed model |
| A1 | CAPM plus one shared image gate | test whether variable-specific gates matter |
| A2 | table-conditioned variable gate | negative control for tabular shortcut risk |

`B1` and `B2` must be checked for numerical and metric agreement before P1 is
interpreted.  A1/A2 are only started after the B1/P1 paired comparison is
complete.

## Data, table, and split protocol

### Task and table contract

The formal IE-CAPM experiment uses one fixed three-variable table:

```text
[age, sex, education]
```

There is no nine-variable or scan-metadata condition in this protocol.  The
image-only ResNet B0 is retained as a baseline with no table input.  CAPM,
IE-CAPM, and all table-gate ablations use exactly the same three columns and
the same preprocessing, missingness handling, and column order.  The task is
`MCI_vs_AD` with `MCI=0` and `AD=1`, matching APIC v3_2 revision 4.

### Paired field-strength exclusion

Before any split, remove every ADNI subject listed under
`subjects_all_paired` in:

```text
data/claim/paired_holdout_subjects.json
```

This is the complete 73-subject set with both 1.5T and 3T records in the
paired-field-strength manifest.  Do not substitute `subjects_le_30d` (33) or
`subjects_le_7d` (25).  The exclusion is subject-wide: remove all scans and
visits for a listed subject.  This prevents the model or the evaluation from
using a 3T/1.5T pair belonging to the same person, while making no claim that
scan field strength is available as an input feature.

The exclusion is applied before train/validation/test construction in either
direction.  It therefore applies when ADNI is the source and when ADNI is the
external target.

### Direction and subject-level split

Run exactly the two APIC v3_2 directions:

| Direction | Source cohort | Source split | External target |
| --- | --- | --- | --- |
| `ADNI_to_NACC` | ADNI after paired-subject exclusion | subject-level 60/20/20 | all eligible NACC subjects |
| `NACC_to_ADNI` | all eligible NACC subjects | subject-level 60/20/20 | ADNI after paired-subject exclusion |

The 6/2/2 split is performed independently within each source cohort using
`split_seed=42`, with no subject appearing in more than one source partition.
Repeated scans remain in the same subject partition.  The external target is
never used for fitting, early stopping, regularizer tuning, gate calibration,
or checkpoint selection.  Unlike APIC's prototype mechanism, IE-CAPM has no
reason to carve a mechanism subset out of the target; target evaluation stays
fully untouched.

### Preprocessing and runtime settings

Use the APIC v3_2 revision 4 preprocessing and runtime contract:

- preprocessing: `skullstrip+n4+mni+crop+normalize`;
- train/evaluation batch sizes: 4/2;
- epochs: 50;
- learning rate / weight decay: `1e-4 / 1e-4`;
- class-weighted cross-entropy;
- metadata matching: maximum 90 days, diagnosis match required, mismatches
  excluded;
- no scan field-strength, vendor, site, or protocol variable enters the model.

Do not add CN_vs_AD, extra table variables, or a second task until the complete
MCI_vs_AD matrix and its pre-registered decision are finished.

Primary endpoint: paired per-seed target balanced accuracy with subject-level
mean aggregation on the untouched external cohort.  Secondary endpoints:
AUROC where defined, sensitivity, specificity, calibration, and subgroup gaps
for observed demographic groups.  Record every per-subject prediction so the
paired change can be inspected.

For consistency with APIC v3_2, set `aggregate=subject_mean`,
`cluster_by_subject=true`, and `label_conflict=earliest_visit`.  Subject-level
bootstrap confidence intervals use 200 resamples and never resample individual
scans independently of their subject.

## Seed, selection, and stopping rules

- Use the APIC v3_2 seed set `[42, 43]` for B1, B2, and P1.  This is a
  protocol-aligned replication matrix, not yet a five-seed journal claim.
- Choose checkpoints using validation data only; apply the same rule to every
  row.  Do not select on the external cohort.
- P1 advances only if it beats B1 in the mean paired primary metric, does not
  show a material loss in the fixed three-variable regime, and the direction
  is consistent in both pre-registered seeds and both directions.  This is a
  screening gate, not evidence of a stable five-seed journal gain.
- A journal mechanism claim additionally requires reproducible gate summaries
  across seeds and no evidence that A2 alone explains the gain.
- Stop and report a negative result if B2 fails the CAPM equivalence check, if
  P1 fails the paired criterion, or if gate summaries are unstable.

## Training interface

```python
from Model.backbone import resnet18_ie_capm

var_specs = [
    {"name": "age", "type": "continuous", "min_val": 55, "max_val": 95, "n_bases": 6},
    {"name": "sex", "type": "categorical", "n_cats": 2, "n_bases": 2},
    {"name": "education", "type": "continuous", "min_val": 0, "max_val": 22, "n_bases": 4},
]
model = resnet18_ie_capm(txt_dim=3, num_classes=2, var_specs=var_specs)
logits, audit = model(mri, table, return_audit=True)
regularizers = model.regularization_losses()
loss = classification_loss + sum(weights[name] * value for name, value in regularizers.items())

# Same learned variable fields, with all evidence gates fixed to one.
capm_logits = model(mri, table, force_capm=True)
```

Pre-register the regularizer weights before the first held-out evaluation.
Start with small values for `gate_anchor`, `gate_floor`, and
`modulation_preservation`; tune them only on the validation set and retain the
same values for all seeds.  Do not add an attribute-invariance loss to clinical
variables.

## Required artifacts

For every completed row, save the resolved config, split-manifest identifier,
seed, git commit, validation checkpoint selection record, predictions, metrics,
and stage-level gate summaries.  Also save the SHA-256 or equivalent content
hash of `data/claim/paired_holdout_subjects.json` and the resolved list of
excluded subject IDs.  Store only aggregate gate maps or approved derived
artifacts; no raw patient data enters the repository.
