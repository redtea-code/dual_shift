# Documentation Map

| Folder | Purpose |
|---|---|
| `plans/` | Registered experiment plans and execution plans |
| `results/` | One result record and its detailed reports per experiment |
| `protocols/` | Frozen data and evaluation contracts |
| `analysis/` | Audits, literature synthesis, and mechanism analysis |
| `decisions/` | Project-wide data facts and claim boundaries |
| `templates/` | Templates for new plan and result records |

Start from [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md). New work must receive a
`DS-xxx` ID, a plan in `plans/`, and a result record in
`results/DS-xxx/README.md`.

See [EXPERIMENT_BRANCH_POLICY.md](EXPERIMENT_BRANCH_POLICY.md) for the rule
that plans, reports, maps, and text stay on `main`, while new experiment
branches contain only the implementation changes needed to run them.

Current registered experiments include DS-034 (scan-filtered CAPM), DS-035
(FMM baseline), DS-036 (target-style CAPM UDA probe), DS-037 (amplitude
transport audit), DS-038 (GRL factorial audit), DS-039 (CAPM-conditioned
residual adaptation), DS-040 (CAPM-conditioned frequency GRL), DS-041
(CAPM-conditioned source-free residual distribution alignment), DS-042
(MRI+table relation-preserving UDA), and DS-043 (CAPM-GRL multimodal concat).
See `EXPERIMENT_INDEX.md` for the current status of each experiment.

Historical `review/` records are preserved on the
`archive/pre-plan34-cleanup-20260809` Git ref. They are not active main-branch
documentation.
