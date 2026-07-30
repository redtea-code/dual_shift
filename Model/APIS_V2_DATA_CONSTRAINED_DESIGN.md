# APIS v2: data-constrained protocol residual intervention

## Claim boundary

APIS now means **Anatomy-Preserving Interventional Steering**. It is a
training-time robustness module over acquisition protocol clusters observed in
the source training set. It does not claim to simulate MRI physics, identify an
isolated field-strength effect, or extrapolate a continuous response surface for
TR/TE/TI.

This boundary follows the available data:

- ADNI: 1,108 usable 1.5T scans, 181 usable 3T scans, and 73 subjects with both
  field strengths;
- only 33 of those subjects have a nearest cross-field scan pair within 30 days;
- NACC: 1,595 usable scans, all 3T;
- field strength is entangled with manufacturer, sequence, site, and protocol
  values.

The defensible question is therefore whether an observed-protocol intervention
improves robustness, not whether the model recovers a causal MRI acquisition
law.

## Intervention

For factual acquisition descriptor `e` and a source-observed target protocol
descriptor `e'`, APIS forms the directed condition

```text
c = [e, e', e' - e].
```

At layer `l`, APIS owns a fixed bank of low-rank spatial residual operators
`B_lk`. A small controller produces only their mixture coefficients:

```text
R_l(H, c) = sum_k tanh(a_lk(c)) B_lk(H)
H_shift   = H + alpha R_l(H, c).
```

The controller never generates convolution weights. The residual RMS is capped
relative to the factual feature RMS, and `alpha <= alpha_max`. The implementation
uses early `layer1` and `layer2` features only.

## Source support

Target protocols are sampled only from the epoch-frozen source prototype bank.
The sampler excludes the current protocol and same manufacturer/field-strength
cluster. If a batch contains a sample without a valid observed target, APIS
returns the clean path instead of extrapolating.

Prototype channel means and standard deviations remain in checkpoints for
backward compatibility, but APIS v2 does not inject or replace them.

## Objective

The shifted path retains the established safeguards:

- shifted classification loss;
- clean/shifted prediction JS consistency;
- clean/shifted deep-feature cosine consistency;
- a small coefficient-energy penalty;
- bounded realized intervention strength.

CDT may continue to use the harder of factual and intervened per-sample losses.

## Evaluation hierarchy

1. **Performance:** subject-disjoint ADNI-to-NACC and NACC-to-ADNI evaluation.
2. **Observed-cluster mechanism:** manufacturer/field/protocol stratified source
   validation without target-driven tuning.
3. **Cross-field mechanism:** the 33 ADNI subjects with a nearest 1.5T/3T pair
   within 30 days; the 25 subjects within 7 days form a stricter sensitivity set.
4. **Negative control:** shuffled acquisition descriptors should remove or reverse
   the mechanism benefit.

No scan from a subject in a paired mechanism set may enter model training.

## Required baselines

- weighted CE;
- MixStyle;
- legacy protocol-guided AdaIN APIS;
- metadata concatenation;
- FiLM or conditional normalization;
- APIS v2 without directed delta;
- APIS v2 with shuffled protocol descriptors.

The main APIS v2 claim requires improvement over both MixStyle and a direct
metadata-conditioning baseline, not only over CE.
