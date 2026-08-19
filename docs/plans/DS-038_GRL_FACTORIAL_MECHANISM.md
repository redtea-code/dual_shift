# DS-038: Domain/intensity GRL factorial audit

Owner: cyh

Status: PROPOSED

Result record: `docs/results/DS-038/README.md`

## Question

The DS-035 ablation `B1c` removed both GRL discriminators at once. Its result
therefore cannot identify whether the exploratory signal came from the domain
head, the intensity head, or their combination. DS-038 performs the missing
two-factor audit while keeping the other FMM components fixed.

This is a component attribution experiment, not a claim that GRL explains
scanner or field-strength causality.

## Locked protocol

- Direction: `ADNI_to_NACC` only.
- Task: MCI vs AD, the same scan-filtered manifests as DS-035.
- Seeds: `42, 43, 44, 45, 46`.
- Backbone and all non-GRL FMM components: identical across variants.
- Source Fourier synthesis: enabled in every cell.
- Attention consistency: enabled in every cell.
- `T_adapt`: unlabeled target image/domain membership only.
- `T_test`: internal frozen post-fit exploratory evaluation only.
- Checkpoint selection: source validation only.
- No target-test tuning, target-label access, or full-target protocol extension.

## Registered variants

The two GRL loss coefficients are the only intended factors. The exact
coefficient is the existing FMM value `1.0`; a zero coefficient disables the
corresponding head while retaining the same model capacity and logging path.

| ID | Variant | Domain GRL | Intensity GRL | Role |
|---|---|---:|---:|---|
| G0 | `no_grl` | 0 | 0 | matched non-adversarial FMM control; equivalent to the missing isolated control |
| G1 | `domain_only` | 1 | 0 | domain-invariance contribution |
| G2 | `intensity_only` | 0 | 1 | intensity-invariance contribution |
| G3 | `both_grl` | 1 | 1 | complete FMM GRL combination |

`B0-ref` is retained as a contextual reference only. It is not part of the
four-cell factorial because its source Fourier and attention paths differ.

## Required mechanism diagnostics

Performance alone is insufficient. Every run must record, by epoch and at the
selected checkpoint where possible:

- domain and intensity discriminator loss;
- discriminator accuracy/AUC and balanced accuracy;
- GRL coefficient and gradient norm reaching the shared encoder;
- source classification CE and source clean/shift CE;
- post-hoc frozen-feature domain probe, trained without target labels;
- target/source feature discrepancy before and after each enabled head;
- source validation/test and exploratory target BA/AUC.

Loss near `log(2)` is not evidence of successful alignment by itself: it may
mean domain confusion, a weak discriminator, or an optimization failure.
The discriminator metrics and gradient audit must be interpreted together.

## Decision rule

Use paired target BA/AUC differences against G0, with source validation and
source-label preservation as safeguards:

- G1 > G0 while G2 does not: domain GRL is the supported candidate;
- G2 > G0 while G1 does not: intensity GRL is the supported candidate;
- G3 > both single-head cells: complementarity is plausible;
- no consistent positive cell: do not retain GRL as an adopted module.

These are exploratory mechanism criteria on the internal frozen `T_test`.
They do not authorize promotion to a fresh confirmatory holdout without a new
registered protocol and complete provenance archive.
