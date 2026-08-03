# APIS v2 claim validation plan

**Status:** preregistration draft for collaborator execution

**Model specification:** `Model/APIS_V2_DATA_CONSTRAINED_DESIGN.md`

**Unit tests:** `tests/test_apis_v2.py`

## 1. Primary claim

APIS v2 uses source-observed acquisition descriptors to apply bounded,
directed residual interventions to early CNN features. The intended claim is
that this training intervention improves robustness to acquisition-protocol
shift while preserving diagnosis-relevant information.

APIS v2 does **not** claim to:

- simulate MRI acquisition physics;
- identify the isolated causal effect of field strength;
- extrapolate a continuous response surface for TR, TE, or TI;
- guarantee anatomical preservation without anatomical supervision.

## 2. Data boundary

The current usable cohorts contain:

| Cohort | Scans | Subjects | 1.5T scans | 3T scans |
|---|---:|---:|---:|---:|
| ADNI | 1,292 | 342 | 1,108 | 181 |
| NACC | 1,595 | 1,292 | 0 | 1,595 |

ADNI contains 73 subjects observed at both field strengths. Their nearest
cross-field scans are not generally synchronous: 33 subjects have a pair
within 30 days and 25 within 7 days. Field strength is also entangled with
manufacturer, site, sequence, and protocol parameters.

Every split must be subject-disjoint. Subjects used for a paired-field
mechanism test must be excluded entirely from model training and model
selection.

## 3. Confirmatory experiments

### E1. External cohort generalization

Run both directions without target-cohort model selection:

1. train and tune on ADNI; test once on NACC;
2. train and tune on NACC; test once on ADNI, with separate 1.5T and 3T reports.

The second direction tests cohort-plus-protocol shift. Because NACC contains no
1.5T scans, it cannot establish an isolated field-strength mechanism.

**Primary endpoint:** subject-level balanced accuracy.

**Secondary endpoints:** ROC-AUC, sensitivity, specificity, macro-F1, and
worst-class recall.

### E2. Held-out observed protocol clusters

Define source protocol clusters using manufacturer, field strength, and
sequence. Repeatedly hold out one eligible cluster for testing and train on the
remaining clusters. Also report leave-one-site-out and, where sample support is
adequate, leave-one-manufacturer-out results.

Report the mean, dispersion, and worst-cluster performance. Refer to
field-strength splits as composite protocol shifts, not causal field-strength
effects.

### E3. Paired ADNI field-strength mechanism test

Use the 33 subjects with a nearest 1.5T/3T pair within 30 days as the primary
paired set. Use the 25 subjects within 7 days as a stricter sensitivity set and
all 73 paired subjects as an exploratory analysis adjusted for time gap.

For each subject, compare:

- clean embedding distance between the 1.5T and 3T scans;
- absolute difference in diagnostic probability;
- agreement of predicted diagnosis;
- change in these quantities relative to CE, MixStyle, and metadata baselines.

Use paired subject-level inference. Verify that lower cross-field distance is
not accompanied by reduced inter-class separation or representation collapse.

## 4. Required baselines

All methods must use the same backbone, split, preprocessing, optimizer budget,
early-stopping rule, and seed list.

1. weighted cross-entropy;
2. MixStyle at the same feature stages;
3. legacy protocol-guided AdaIN APIS;
4. direct metadata concatenation;
5. FiLM or conditional normalization;
6. parameter-count-matched unconditional residual operator;
7. APIS v2;
8. CDT plus APIS v2 where the joint claim is evaluated.

Hyperparameter search budgets must be comparable across methods.

## 5. Negative controls

These tests determine whether acquisition descriptors carry the benefit rather
than acting as generic noise or additional capacity.

- shuffle descriptors within each minibatch;
- globally shuffle descriptors subject-wise;
- replace descriptors with a zero vector;
- retain the residual operator but use random coefficients;
- sample a random target without protocol-distance filtering;
- train with field strength only, manufacturer only, and continuous sequence
  parameters only.

The mechanism claim requires correct descriptors to outperform shuffled and
unconditional controls.

## 6. Structural ablations

| Ablation | Question |
|---|---|
| remove `e' - e` | Is intervention direction necessary? |
| use only `e' - e` | Do factual and target states add information? |
| remove RMS cap | Does the bound improve stability? |
| one residual basis | Is basis diversity useful? |
| replace 3D basis with channel-only basis | Is spatial response useful? |
| layer1 only | Where should intervention occur? |
| layer2 only | Where should intervention occur? |
| layer1 plus layer2 | Does multistage intervention help? |
| remove coefficient penalty | Does intervention regularization matter? |

Record realized intervention strength and the fraction of samples or batches
for which no valid target protocol exists.

## 7. Representation evidence

Freeze each trained backbone and fit identical linear probes at multiple layers
for diagnosis and for acquisition attributes: field strength, manufacturer,
site, sequence, and binned TR/TE/TI where support permits.

The desired pattern is reduced protocol decodability in later features without
reduced diagnosis decodability. This is evidence of reduced linear protocol
information, not proof that all protocol information has been removed.

For the preservation claim, report clean-versus-intervened prediction JS
divergence, deep-feature cosine similarity, saliency-region overlap, and
brain-region activation stability. Unless segmentation or regional anatomical
supervision is added, use the wording `diagnosis-relevant semantic
preservation` rather than `guaranteed anatomical preservation`.

## 8. Statistical analysis

- use at least five fixed random seeds for confirmatory comparisons;
- aggregate repeated scans at subject level before computing primary metrics;
- use subject-level paired bootstrap confidence intervals for method
  differences;
- use paired inference for the cross-field mechanism sets;
- report effect sizes and 95% confidence intervals, not only p-values;
- apply Holm correction across confirmatory secondary comparisons;
- do not choose checkpoints, thresholds, or hyperparameters on the external
  target cohort.

The primary success criterion is a positive subject-level balanced-accuracy
difference versus both MixStyle and direct metadata conditioning on the
external-cohort test, with a 95% paired-bootstrap confidence interval excluding
zero. Mechanism experiments are supporting evidence and must not replace this
performance criterion.

## 9. Readiness gates

Before launching shared experiments:

1. restore or provide `training/dual_shift_loop.py`, which is imported by
   `experiments/train_journal.py` but absent on the current branch;
2. repair the legacy `Model/__init__.py` imports for modules absent on the
   current branch, or import the dual-shift package through a self-contained
   entry point;
3. connect `output.extras["apis_coefficient_l2"]` to
   `compute_dual_shift_loss(intervention_penalty=...)`;
4. implement and log a per-sample valid-intervention mask instead of skipping a
   whole batch when one sample has no eligible target;
5. run one-source, one-seed smoke training before dispatching the full matrix.

The component-level APIS v2 tests cover identity behavior, residual bounds,
gradient flow, directed conditioning, and observed-protocol sampling. They do
not substitute for the end-to-end readiness gates above.
