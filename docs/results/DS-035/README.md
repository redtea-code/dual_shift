# DS-035: Result record

Status: RUNNING

Plan: [DS-035 plan](../../plans/DS-035_FMM_BASELINE.md)

## Current state

The independent FMM runner and its smoke path are implemented. One exploratory
seed-42 run is recorded below; it is not a fresh confirmatory claim because the
target holdout was already used by historical C0-C4 reports.

## Verified evidence

- The comparison matrix and target-label access boundary are registered in the
  plan.
- The smoke command is an implementation check only.
- [Seed-42 exploratory ablation report](ABLATION_RESULTS_2026-08-18.md)

## Not verified

- Multi-seed real-data execution with a fresh frozen target holdout.
- Stable component attribution from the single seed.
- Faithful reproduction of every unspecified upstream FMM choice.

## Next action

Run the pre-registered baseline across multiple seeds with a fresh frozen
target holdout, then add the command, commit, configuration, manifest, metrics,
and GO/NO-GO decision here.
