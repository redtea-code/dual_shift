# IE-CAPM CN vs AD (demos-only) run status — 3090

## Basic info

- Date: 2026-08-07
- Host: Windows · RTX 3090
- Branch: `run/ie-capm-cn-ad-3090` (from `main` @ `baeef7b`)
- Plan: `review/plans/32_ie_capm_paired_experiment_plan_2026-08-07.md`
- Analysis: `review/analysis/31_ie_capm_self_review_and_implementation_2026-08-07.md`
- Task: **NC/CN vs AD only** (`label_mapping: {1: 0, 3: 1}`)
- Table regime: **demographics-only** `[age, sex, education]` (journal z-scored cov3)
- Claim status: **no outcome claim authorized** (plan still `planned`)

## Scope decisions

1. This host runs **CN vs AD only**; MCI is out of scope here.
2. Complete clinical table (MMSE/APOE/CDRSB/ADAS/FAQ/site) is **not** jointly available on the current ADNI/NACC CSVs used by the journal path. Regime deferred; retained columns for the active regime are recorded below.
3. Legacy SCA-CAPM import is broken (`Model.causal` absent). Matrix mapping:
   - **B0** = `ie_capm_img` (same ResNet-18, `z=None`)
   - **B1** = `ie_capm_force` (train/eval with `force_capm=True`)
   - **B2** = `ie_capm_b2_force_eval` (P1 checkpoint evaluated with `force_capm=True`)
   - **P1** = `ie_capm` (gated training)
4. A1/A2 ablations are deferred until B1/P1 paired decision (per plan).

## Wiring delivered

- Variants registered in `experiments/train_journal.py`
- Pre-registered regularizer weights: gate_anchor/floor/modulation_preservation=0.01, basis_tv/orth=0.001
- Checkpoint selection: subject-level balanced accuracy for all IE-CAPM rows
- Artifacts: predictions CSVs, `journal_metrics.json`, `ie_capm_gate_summary.json`
- Configs:
  - `journal_ie_capm_cn_ad_demos.yaml` (full, 50 epochs, seeds 42–46)
  - `journal_ie_capm_cn_ad_demos_smoke.yaml` (2-epoch wiring smoke)

## Smoke (passed)

```text
python experiments/train_journal.py \
  --config_path journal_ie_capm_cn_ad_demos_smoke.yaml \
  --direction ADNI_to_NACC --seed 42 \
  --variants ie_capm_img ie_capm_force ie_capm \
  --device cuda \
  --output-dir outputs/journal/ie_capm_cn_ad_demos_smoke/cn_adni_to_nacc \
  --force-variants
# EXIT=0; B0/B1/P1/B2 artifacts written
```

## Full matrix (in progress)

```text
python experiments/train_journal.py \
  --config_path journal_ie_capm_cn_ad_demos.yaml \
  --study --directions ADNI_to_NACC NACC_to_ADNI \
  --seeds 42 43 44 45 46 \
  --variants ie_capm_img ie_capm_force ie_capm \
  --device cuda \
  --output-dir outputs/journal/ie_capm_cn_ad_demos \
  --force-variants
```

Cells: `{B0,B1,P1(+B2)} × {ADNI↔NACC} × seeds{42..46}` under demos-only.

Stdout: `outputs/journal/ie_capm_cn_ad_demos_stdout.txt`

## Next after completion

1. Summarize paired target BA (P1 vs B1) across seeds/directions.
2. Check B1 vs B2 agreement before interpreting P1.
3. Archive metrics tables under this `review/records/ie_capm/3090/` directory.
4. Only then consider A1/A2 or the complete-table regime.
