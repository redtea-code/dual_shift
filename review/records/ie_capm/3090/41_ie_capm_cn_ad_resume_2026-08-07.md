# IE-CAPM CN vs AD resumed on 3090 (not a copy of MCI formal protocol)

## Decision

- Host charter: **CN/NC vs AD only** on this 3090.
- Stopped the mistaken MCI_vs_AD restart.
- Resume CN with **shared experimental hygiene** from
  `docs/IE_CAPM_APIC_V3_2_ALIGNED_EXPERIMENT_PROTOCOL.md`, without copying the
  formal MCI task.

## What is shared vs not copied

| Item | This CN run | Formal protocol doc |
| --- | --- | --- |
| Task | **CN vs AD** (`1→0, 3→1`) | MCI vs AD (not used here) |
| Table | age/sex/education raw | same |
| Holdout | `subjects_all_paired` | same |
| Batch / epochs / lr | 4/2, 50, 1e-4 | same |
| Bootstrap | 200 | same |
| Target carve | false | same |
| Seeds | 42, 43 (3090 CN screening) | formal MCI seeds; not a claim that CN is the formal matrix |
| Matrix | B0/B1/P1 (+B2 eval) | same row IDs |

Superseded partial runs:

- `outputs/journal/ie_capm_cn_ad_demos/` (wrong batch/z-score/5-seed)
- `outputs/journal/ie_capm_mci_ad/` (wrong task for this host)

## Command

```text
python experiments/train_journal.py \
  --config_path journal_ie_capm_cn_ad.yaml \
  --study --directions ADNI_to_NACC NACC_to_ADNI \
  --seeds 42 43 \
  --variants ie_capm_img ie_capm_force ie_capm \
  --device cuda \
  --output-dir outputs/journal/ie_capm_cn_ad \
  --force-variants
```

Config: `journal_ie_capm_cn_ad.yaml`  
Stdout: `outputs/journal/ie_capm_cn_ad_stdout.txt`  
Claim status: still **no outcome claim**.
