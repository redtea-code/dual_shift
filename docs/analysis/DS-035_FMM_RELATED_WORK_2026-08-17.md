# FMM Baseline Related Work And Applicability Review

## 1. Purpose

This note separates three questions:

1. Is there a mature frequency UDA baseline for 3D brain-disease MRI?
2. Which published assumptions are permitted by the DualShift comparison protocol?
3. Which ideas are later improvements rather than substitutes for the baseline?

All source PDFs were read from the project literature set. Their scores are
not compared directly with DualShift outcomes because task definitions,
target-data access, labels, preprocessing, and selection rules differ.

## 2. Direct Baseline Family

| Work | Task and access regime | Mechanism | Role here | Compatibility |
|---|---|---|---|---|
| FMM, ACPR 2023 | 3D sMRI UDA; labeled source, unlabeled target | Source intensity Fourier synthesis; full amplitude mix; target phase; attention consistency; GRL | Primary B1-fmm | Independent UDA branch |
| DyMix, arXiv 2024 | 3D sMRI AD UDA | FMM-like two stages plus dynamic frequency-region scheduler | Deferred extension | Selector must be shown label-blind |
| SAMix, arXiv 2023 | Few-shot UDA in 2D tasks | Per-frequency Wasserstein distance and adversarial spectral mixup | Future frequency diagnostic | Not an initial baseline |

FMM is the correct first comparison because it combines raw amplitude/phase manipulation, unlabeled target access, and brain-disease classification without biological pairs. DyMix is not a substitute baseline because its dynamic scheduler adds a target-selection risk.

## 3. Useful Diagnostics, Not Model Baselines

| Work | Reusable question | Why not a drop-in model |
|---|---|---|
| Gautheron et al., 2018, OT feature selection | Which source/target coordinates are similar or shifted? | Hand-crafted/vector feature selection, not a 3D CNN UDA architecture |
| SAMix, 2023 | Which raw frequencies have high source-target Wasserstein distance? | Few-shot setting and adversarial coefficient search |
| GenCA-MRI, 2025 | Are proposed invariant features structurally meaningful rather than background artifacts? | MRI reconstruction DG and multi-source causal alignment |
| SMEDL, 2025 | Does style mixing preserve task content? | 2D segmentation with pseudo-label consistency and adversarial learning |

OT and SAMix motivate label-blind diagnostics before a new improvement: compute raw-spectrum and feature-spectrum shifts on S_train and T_adapt, report bootstrap stability, and distinguish raw volumes from spatial feature maps and GAP embeddings. They do not justify retraining selected frequencies after target results are known.

## 4. Later Architectural Hypotheses

| Work | Relevant idea | Why it is not current evidence |
|---|---|---|
| Adaptogen-Perturbation, MedIA 2026 | Freeze backbone and optimize a small target-specific perturbation | Test-time/target adaptation for segmentation |
| SUDA, MedIA 2026 | Unlabeled target adaptation of a student representation | Pathology distillation and later target-label fine-tuning |
| Spectral-conditioned hypernetworks, 2026 PDF | Raw spectral embedding conditions normalization affine parameters | Multi-site meta-learning and five labeled target support subjects; DOI needs independent verification |
| Unified semantically grounded DA, TMI 2026 | Structural priors can avoid adversarial alignment and paired scans | 2D segmentation, reconstruction/deformation objectives, target validation subjects |

The spectral-hypernetwork paper is conceptually close to raw spectrum -> bounded gamma/beta, but it does not validate that operation in label-blind, single-source ADNI-to-NACC UDA. It is a future hypothesis after B1-fmm, not support for implementing a domain conditioner now.

## 5. Claim Boundaries

The following are testable in this worktree:

- unpaired source and target MRI batches can support an image-level frequency UDA algorithm;
- raw amplitude/phase operations are distinct from feature-map FFT and need their own audits;
- full-frequency rather than fixed-low-frequency mixing is a testable baseline choice;
- a frequency-distance map can be a label-blind descriptive quantity.

The following are not supported:

- ADNI/NACC spectral separability proves scanner or field-strength causality;
- a source-target domain classifier proves disease-preserving correction;
- an FMM result on the previously viewed target holdout is confirmatory;
- target-supervised, multi-source DG, reconstruction, or segmentation results rank directly against this UDA baseline.

## 6. Research Sequence

~~~
FMM reference baseline (B0-ref vs B1-fmm)
        -> FMM ablations (source FFT, attention, GRL)
        -> same-backbone FMM-core port
        -> frequency-difference weighting
        -> optional frozen spectral domain conditioner
~~~

Each arrow requires a locked configuration and source-selected checkpoint. The next arrow is not licensed by a single target point estimate.

## 7. Local Reading Set

- 978-3-031-47665-5_11.pdf: FMM, primary baseline.
- 2410.12827v1.pdf: DyMix, FMM extension.
- 2309.01207v1.pdf: SAMix.
- 1806.10861v1.pdf: OT feature selection for UDA.
- 1-s2.0-S1361841524003657-main.pdf: SMEDL.
- 1-s2.0-S1361841525000076-main.pdf: GenCA-MRI.
- 1-s2.0-S136184152600071X-main.pdf: Adaptogen-Perturbation.
- 1-s2.0-S136184152600246X-main.pdf: SUDA.
- article_an0262300241_cec045.pdf: spectral-conditioned hypernetworks.
- Unified_and_Semantically_Grounded_Domain_Adaptation_for_Medical_Image_Segmentation.pdf: semantic structure UDA.

## 8. Decision

GO for building the independent B0-ref/B1-fmm comparison path after inspecting the author implementation and locking unspecified hyperparameters. NO-GO for the low-rank domain conditioner, dynamic scheduler, or a new target-driven frequency selector before the FMM baseline exists.
