# Model Variants Branch

This branch stores runnable model and experiment-code snapshots. Each experiment has its own folder under `model_variants/`, while reusable code is kept under `model_variants/shared/`.

## Layout

- `model_variants/shared/`: common backbones, DualShift modules, and shared training utilities
- `model_variants/DS-034/` through `model_variants/DS-044/`: experiment-specific model variants, runners, and configuration files
- The original project paths remain in place so the branch stays compatible with existing imports and entry points.

The folders are snapshots, not symlinks. Changes for a new experiment should be made in its experiment folder first, then promoted to shared code only when the interface is stable.

## Follow-up merges

The unresolved remote branches are kept available for later review. Merge their model code into the matching experiment folder rather than merging whole branches blindly:

- `codex/ds041-capm-residual-distribution` -> `model_variants/DS-041/`
- `codex/ds042-cmrp-uda` -> `model_variants/DS-042/`
- `codex/ds043-experiment-plan` -> `model_variants/DS-043/`
- `exp/ds043-run` -> `model_variants/DS-043/` and `model_variants/DS-044/`

Before any merge, compare the branch against `main`, resolve code conflicts by experiment scope, run the relevant tests, and record the source commit in that folder's `README.md`.

## Resolution status

| Experiment | Branch handling |
|---|---|
| DS-041 | Superseded by consolidated `main`; retained the current exports and recorded the older conflict. |
| DS-042 | Consolidated implementation retained; full `codex/ds042-cmrp-uda` source snapshot isolated under DS-042. |
| DS-043 | Planning branch handled as documentation; run-branch model files isolated under DS-043. |
| DS-044 | No finalized model implementation yet; run-branch context staged under DS-044. |

## Branch relationship

This branch starts from the consolidated `main` commit `606b246`. It is intended as a code-variant archive and integration staging area; documentation remains organized on `docs/experiment-records`.
