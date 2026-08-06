# IE-CAPM paired experiment plan

## Status

- Date: 2026-08-07
- Status: planned; no outcome claim is authorized by this plan.
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

All rows use the same patient-level split manifests, MRI preprocessing,
backbone depth, classifier head, optimizer schedule, augmentation policy,
epoch budget, checkpoint rule, and random seeds.

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

## Data regimes and endpoints

Run each comparison in both table regimes already supported by CAPM evidence:

1. complete observed table;
2. demographics-only table, with the exact retained columns recorded.

Use the pre-existing primary clinical task and its held-out protocol.  Repeat
the same matrix on the second task only after the primary task decision is
fixed.  Do not use unidentifiable ADNI/NACC scan parameters as features or as
stratification labels.

Primary endpoint: paired per-seed balanced accuracy (and the project primary
metric, if different) on the untouched held-out cohort.  Secondary endpoints:
AUROC where defined, sensitivity, specificity, calibration, and subgroup gaps
for observed demographic groups.  Record every per-subject prediction so the
paired change can be inspected.

## Seed, selection, and stopping rules

- Use at least five pre-registered seeds for B1, B2, and P1.
- Choose checkpoints using validation data only; apply the same rule to every
  row.  Do not select on the external cohort.
- P1 advances only if it beats B1 in the mean paired primary metric, does not
  show a material loss in either table regime, and the direction is consistent
  in at least four of five seeds.
- A journal mechanism claim additionally requires reproducible gate summaries
  across seeds and no evidence that A2 alone explains the gain.
- Stop and report a negative result if B2 fails the CAPM equivalence check, if
  P1 fails the paired criterion, or if gate summaries are unstable.

## Training interface

```python
from Model.backbone import resnet18_ie_capm

model = resnet18_ie_capm(txt_dim=9, num_classes=3, var_specs=var_specs)
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
and stage-level gate summaries.  Store only aggregate gate maps or approved
derived artifacts; no raw patient data enters the repository.
