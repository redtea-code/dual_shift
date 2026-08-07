# IE-CAPM protocol alignment check and restart — 3090

## Basic info

- Date: 2026-08-07
- Host: Windows · RTX 3090
- Branch: `run/ie-capm-cn-ad-3090`
- Authority: `docs/IE_CAPM_APIC_V3_2_ALIGNED_EXPERIMENT_PROTOCOL.md` (`main` @ `91d4771`)
- Plan: `review/plans/32_ie_capm_paired_experiment_plan_2026-08-07.md` (updated)

## Comparison: previous CN run vs protocol

| Item | Previous CN run | Protocol (main) | Match? |
| --- | --- | --- | --- |
| Task | CN/NC vs AD (`1→0, 3→1`) | **MCI vs AD** (`2→0, 3→1`) | **NO** |
| Table | age/sex/edu (z-scored) | age/sex/edu (**raw** 55–95 / 0–22) | **NO** |
| Seeds | 42–46 (five) | **42, 43** only | **NO** |
| Train batch | 2 | **4** | **NO** |
| Bootstrap | 1000 | **200** | **NO** |
| Target carve | `carve_e3_from_target: true` | **false** (no mechanism carve) | **NO** |
| Holdout key | `subjects_all_paired` | `subjects_all_paired` | yes |
| Directions | ADNI↔NACC | ADNI↔NACC | yes |
| Epochs / lr / wd | 50 / 1e-4 / 1e-4 | 50 / 1e-4 / 1e-4 | yes |
| Matrix rows | B0/B1/P1(+B2) | B0/B1/B2/P1 | yes |

Protocol also states: do **not** start CN_vs_AD until the MCI_vs_AD matrix and decision are finished.

## Action taken

1. Stopped the inconsistent CN full study (`outputs/journal/ie_capm_cn_ad_demos/`). That run is **superseded / not claimable**.
2. Checked out protocol docs from `origin/main`.
3. Added protocol-aligned config: `journal_ie_capm_mci_ad.yaml`.
4. Default IE-CAPM `var_specs` now use `protocol_table_var_specs()` (raw age/education).
5. Restarted formal matrix: MCI vs AD × seeds `{42,43}` × directions ADNI↔NACC × `{ie_capm_img, ie_capm_force, ie_capm}`.

## Restart command

```text
python experiments/train_journal.py \
  --config_path journal_ie_capm_mci_ad.yaml \
  --study --directions ADNI_to_NACC NACC_to_ADNI \
  --seeds 42 43 \
  --variants ie_capm_img ie_capm_force ie_capm \
  --device cuda \
  --output-dir outputs/journal/ie_capm_mci_ad \
  --force-variants
```

Stdout: `outputs/journal/ie_capm_mci_ad_stdout.txt`

## Claim status

Still **no outcome claim**. This restart only restores protocol compliance for the formal MCI_vs_AD screening gate.
