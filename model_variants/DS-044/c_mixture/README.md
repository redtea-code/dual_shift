# DS-044 C1/C2 Reproduction

This folder contains the leakage-safe reproduction runner for the target-mixture hypothesis.

The runner expects an NPZ containing frozen P0 outputs. The current workspace
does not contain the historical P0 checkpoints or the C1/C2 output arrays, so
the real-data reproduction remains pending until those server artifacts are
restored.

- `source_train_embeddings`: source-train scan or subject embeddings;
- `source_train_subject_ids`: required when source embeddings are scan-level;
- `target_embeddings` and `target_subject_ids`;
- optional `target_probabilities` at subject level;
- optional `target_scan_probabilities` for scan-to-subject variance diagnostics.

Example:

```powershell
python run_ds044_c_mixture.py `
  --input outputs/ds044_followup/NACC_to_ADNI/C1_C2/frozen_p0_seed42.npz `
  --output outputs/ds044_followup/NACC_to_ADNI/C1_C2/seed42/summary.json `
  --seed 42
```

The script fits PCA on source-train embeddings only, aggregates target scans by subject, evaluates `K=2,3,4`, computes bootstrap ARI, and reports cluster-level support distance and prediction uncertainty without reading target labels.

The NPZ is an explicit handoff boundary. It prevents accidental mixing of a newly trained P0, a different subject split, or target labels into the historical reproduction.
