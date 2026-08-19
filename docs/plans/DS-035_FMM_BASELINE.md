# DS-035: FMM Baseline Comparison Design

Owner: cyh

Status: RUNNING

Result record: `docs/results/DS-035/README.md`

## 1. Status And Scope

Status: implementation complete for the independent runner and smoke path. No real-data FMM checkpoint or performance claim has been produced in this worktree.

This is an independent comparison line for Frequency Mixup Manipulation (FMM), not a modification of the existing C0-C4 frequency-prior route. The question is:

> Under the DualShift ADNI 1.5T to NACC 3T data contract, does the published raw-image frequency UDA baseline work, and which components are responsible for any observed change?

Primary source: Shin et al., Frequency Mixup Manipulation Based Unsupervised Domain Adaptation for Brain Disease Identification, ACPR 2023, DOI 10.1007/978-3-031-47665-5_11. Local PDF: D:\1\定方向\2026AD\域适应\978-3-031-47665-5_11.pdf.

The authors state that code is available at https://github.com/ku-milab/FMM. The public checkout used for source comparison is commit `580625cee5bfc1474fe8700e530ade07ac5e9776`. It is a reference only: the checkout is incomplete for a clean run because the training scripts import files that are absent from the tree. The local implementation records this commit and does not import the repository at runtime.

## 2. Why FMM Is The First Baseline

FMM is the closest available method because it is UDA with source diagnostic labels and unlabeled target images, uses 3D structural MRI for brain-disease classification, mixes raw MRI amplitudes, and has both source-side intensity robustness and target adaptation stages.

It is not equivalent to C4. C4 applies a bounded gate to layer4 feature-map spectra. FMM mixes raw MRI amplitudes, retains target phase in its inter-domain synthetic image, uses attention consistency, and trains gradient-reversal domain discriminators. Neither a positive nor a negative FMM result proves the C4 mechanism.

DyMix is a later FMM extension. It dynamically changes the frequency region using an AUC-based scheduler whose validation-label provenance is insufficiently specified for the current label-blind target contract. FMM is therefore the first baseline; DyMix is deferred.

## 3. Data And Selection Contract

Primary direction only:

~~~
ADNI 1.5T (MCI vs AD) -> NACC 3T (MCI vs AD)
~~~

| Population | Allowed use |
|---|---|
| S_train | Source classification and source-side FMM pretraining |
| S_val | Sole checkpoint, epoch, and hyperparameter selector |
| S_test | Post-training source diagnostic only |
| T_adapt | Unlabeled FMM inter-domain batches; source/target membership only |
| T_test | Final score only; never read during fitting or selection |

All partitions remain subject-disjoint and preserve the current scan-filtered task mapping, preprocessing, input shape, source split seed, subject-mean aggregation, and bootstrap reporting. A target training loader must not request a diagnosis label.

### Historical Test-Set Limitation

The current T_test has already appeared in historical C0-C4 result reports. A new FMM score on it is therefore an exploratory benchmark, not a fresh confirmatory claim. The FMM configuration must be locked before the run; no follow-up variant may be selected from that result.

A future confirmatory claim requires a newly registered, label-unread target holdout excluded from the new model's adaptation loader. This worktree does not silently create one.

## 4. Published FMM Core

For independent source and target batch samples (x_s, y_s) and x_t, let A(.) and P(.) denote raw-volume FFT amplitude and phase.

### Stage I: Source Intensity-Invariant Pretraining

The paper creates an intensity-transformed source x_is and synthesizes:

~~~
x_dis = iFFT(A(FFT(x_is)), P(FFT(x_s)))
~~~

The source classifier receives source supervision and an intensity discriminator is trained through gradient reversal. The paper describes x_is as a random-noise/intensity transformation, without an exact transform distribution or hyperparameters.

### Stage II: Inter-Domain Adaptation

FMM mixes full raw amplitudes without a biological pair:

~~~
A_mix = (1 - lambda) * A(FFT(x_t)) + lambda * A(FFT(x_s))
x_dmt = iFFT(A_mix, P(FFT(x_t)))
lambda ~ Uniform(fixed range)
~~~

The output retains target phase and is unlabeled; it must not receive y_s. The paper makes spatial attention maps consistent between source and target-style branches, while a domain classifier uses gradient reversal:

~~~
L_total = L_cls + lambda_att * L_att - lambda_dom * L_dom
~~~

The paper prints unit coefficients in its final expression. This design keeps the coefficients explicit because their values and the GRL schedule must be recovered from the author implementation or pre-registered.

## 5. Comparison Matrix

The initial matrix separates a literature-method comparison from a same-backbone comparison. Otherwise, an apparent effect could come from the FMM ten-layer attention encoder rather than frequency UDA.

| ID | Model family | Target images during training | Purpose |
|---|---|---:|---|
| B0-ref | FMM reference encoder, source ERM only | No | Controls the FMM architecture |
| B1-fmm | FMM reference encoder, full two-stage FMM | T_adapt only | Primary published-method baseline |
| B1a-no-source-fft | B1-fmm without Stage-I Fourier synthesis | T_adapt only | Tests source intensity manipulation |
| B1b-no-attention | B1-fmm without L_att | T_adapt only | Tests attention consistency |
| B1c-no-grl | B1-fmm without both GRL discriminators | T_adapt images only | Identifies the non-adversarial spectral component |
| B2-capm | Existing layer5_pixel + original_capm | Current protocol | Project-level clinical baseline |
| B3-fmm-core-capm | FMM raw-image procedure ported to current backbone | T_adapt only | Later fair-backbone port; not faithful FMM |

B0-ref versus B1-fmm is the first required contrast. B2-capm versus B3-fmm-core-capm is deferred until the reference route is complete.

## 6. Implementation Boundary

Implemented as a new runner and model family: `experiments/train_fmm_baseline.py`, `Model/ablation/fmm_baseline.py`, `training/fmm_frequency.py`, and `training/fmm_protocol.py`. The entry point is configured by `fmm_baseline_scan_filtered_1p5t_mci_ad.yaml`. It does not overload C4 frequency_uda configuration or reinterpret the layer4 gate as FMM.

The ten-convolution encoder, 3D spatial gate, and two discriminator heads are
ported independently. Two engineering differences are explicit: convolution
padding keeps the configured MRI crop shape stable, and adaptive pooling
replaces the reference hard-coded `128*2*3*2` flatten size. The default channel
schedule and classifier hidden width remain those of the reference model.

Each UDA step needs two independent loaders:

~~~
source loader -> image, diagnosis label, subject ID
target-adapt loader -> image, domain membership only, subject ID
~~~

No source and target subject are clinically paired. Batch pairing is an algorithmic sampling operation only; sampled subject IDs must be logged.

The paper's ten convolutional layers, BatchNorm/ReLU blocks, even-layer downsampling, spatial attention, and two discriminators are ported before CAPM is introduced. If the author repository differs from the paper, record the source commit and behavior difference.

The missing paper-level choices are registered in the YAML rather than tuned on T_test: intensity scale `[0.8, 1.2]`, noise standard deviation `0.05`, amplitude mixing interval `[0, 1]`, unit GRL/domain/attention coefficients, and a deterministic 50/50 target adaptation/holdout subject split. These settings make the experiment runnable, but do not convert it into an exact reproduction claim.

## 7. Pre-Implementation Unknowns

The PDF does not fully specify:

1. the random intensity/noise transform used for x_is;
2. the interval used to sample lambda;
3. GRL coefficients, discriminator architecture, and scheduling;
4. the exact attention-map pairing and reduction for L_att;
5. the hold-out provenance for its AUC model selector.

These are blockers for an exact-reproduction claim, not permission to tune on T_test. The author repository must be inspected first. Any missing value becomes a pre-registered FMM-DualShift choice, not a claim of faithful reproduction.

## 8. Required Tests And Audit Artifacts

Before real-data training:

1. iFFT(A(x), P(x)) reconstructs x within numerical tolerance.
2. lambda=0 preserves target amplitude and phase; lambda=1 combines source amplitude with target phase.
3. Target diagnosis labels are inaccessible to all FMM training functions.
4. Source, T_adapt, and T_test subject digests are disjoint.
5. GRL reverses, rather than merely scales, the encoder gradient.
6. B1c-no-grl removes both discriminator losses with no unused-gradient side effects.

The implementation smoke path is exercised with:

~~~
D:\Anaconda\envs\segment\python.exe experiments/train_fmm_baseline.py --config_path fmm_baseline_scan_filtered_1p5t_mci_ad.yaml --direction ADNI_to_NACC --variant b1_fmm --smoke-test --output-dir outputs/fmm_smoke_b1
~~~

Each run saves: Git commit, upstream reference commit, configuration hash, subject digests, target-label/metric access flags, seed-derived selector history, branch losses, subject-level predictions, sampled target subject IDs, and raw spectral diagnostics. Report clean source validation, source test, and only the pre-registered exploratory T_test row.

## 9. Promotion Rule

Do not promote FMM because a discriminator is active, FFT intervention is nonzero, or one target score rises. Only after the reference baseline and its ablations complete may frequency-difference weighting or a low-rank domain conditioner be compared against both FMM and a same-backbone control.
