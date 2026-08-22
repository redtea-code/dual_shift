# Experiment Index

This is the single status list for active experiments. Each row must link to
one plan and, once work is complete, one result record.

| ID | Experiment | Owner | Status | Plan | Result | Last updated |
|---|---|---|---|---|---|---|
| DS-034 | Scan-filtered CAPM / IE-CAPM / APIC v3_2 | cyh | RUNNING | [plan](plans/DS-034.md) | [results](results/DS-034/README.md) | 2026-08-19 |
| DS-035 | FMM raw-image UDA baseline | cyh | COMPLETED | [plan](plans/DS-035_FMM_BASELINE.md) | [results](results/DS-035/README.md) | 2026-08-19 |
| DS-036 | Target-style transport + CAPM UDA probe | cyh | COMPLETED | [plan](plans/DS-036_TARGET_STYLE_CAPM_UDA.md) | [results](results/DS-036/README.md) | 2026-08-19 |
| DS-037 | Amplitude transport mechanism audit | cyh | COMPLETED | [plan](plans/DS-037_AMPLITUDE_TRANSPORT_MECHANISM.md) | [results](results/DS-037/README.md) | 2026-08-20 |
| DS-038 | Domain/intensity GRL factorial audit | cyh | BLOCKED | [plan](plans/DS-038_GRL_FACTORIAL_MECHANISM.md) | [results](results/DS-038/README.md) | 2026-08-19 |

## Status values

`PROPOSED` -> `RUNNING` -> `COMPLETED`

`RUNNING` -> `BLOCKED`

`COMPLETED` requires a result record. `BLOCKED` must state the blocking reason
and the next decision needed.

## Record rules

- Use a new `DS-xxx` ID before starting work.
- Keep protocol, configuration, data/manifest, and code commit together in the
  plan or result record.
- Separate verified evidence from interpretation and future proposals.
- Update this table in the same change as the experiment record.
