# DS-040: Result record

Status: **PROPOSED**

Plan: [DS-040 plan](../../plans/DS-040_CAPM_CONDITIONED_FREQUENCY_GRL.md)

## 1. Current evidence state

DS-040 is a registered plan only. No real ADNI/NACC DS-040 training run, checkpoint, target
adaptation split, mechanism diagnostic, or target report has been produced.

The plan combines the FMM raw-Fourier source/target-style enhancement with the
domain and intensity GRL factors, but moves the principal adversarial pressure
to a CAPM-conditioned residual branch. The complete-CAPM GRL variants are
retained as controls.

## 2. Registered model family

The first implementation is a matched ResNet10 (`layers=[1,1,1,1]`)
`layer4_pixel` CAPM backbone with `[age, sex, education]`. The source path computes CAPM features, the frequency
branches create source-intensity and target-style images, and the GRL heads
operate either on the complete CAPM feature or on the frozen source-task
residual complement.

Registered cells are P0, F0-F3, and R1-R3 in the plan. `R3` is the proposed
conditional two-head path; no complementarity claim is allowed unless it beats
both residual single-head cells with mechanism evidence.

## 3. Evidence boundary

DS-035, DS-038, and DS-039 motivate the plan but do not validate it:

- DS-035 supplies an exploratory full-FMM signal but cannot isolate the two
  GRL heads;
- DS-038 is incomplete/blocked for the registered factorial and does not show
  universal domain/intensity complementarity;
- DS-039 shows a small residual-pilot effect but no learned GRL or frequency
  mechanism.

All DS-040 target results will remain exploratory on the historical internal
holdout. `T_adapt` must be image-only and subject-disjoint from `T_test`.

## 4. Required next action

The minimal implementation is now present in this branch:

- `Model/ablation/capm_frequency_grl.py`
- `experiments/run_capm_frequency_grl.py`
- `tests/test_capm_frequency_grl.py`

The runner includes a deterministic `--smoke-test` path, source-CE-gradient
projector fitting, raw-Fourier source/intensity and target-style construction,
full/residual GRL heads, and source-validation-only checkpoint selection. The
smoke path is an interface check only; it does not create DS-040 evidence.
Each invocation writes `report.json`, `audit.json`, `predictions.json`, and
`best.pt` under the requested output directory; these are provenance containers
until a real ADNI/NACC run is completed.

Interface smoke command:

```text
D:\Anaconda\envs\segment\python.exe experiments/run_capm_frequency_grl.py --smoke-test --variant R3 --output outputs/ds040_smoke/report.json
```

Before real-data execution, run all registered cells under the locked
source-validation selector and archive complete provenance, mechanism
diagnostics, and paired seed comparisons. The result record must remain
`PROPOSED` until those artifacts exist.

Until those artifacts exist, DS-040 remains **PROPOSED** and has no performance
conclusion.
