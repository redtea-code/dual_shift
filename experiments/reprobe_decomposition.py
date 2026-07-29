"""
Re-run Phase 1 decomposition diagnostics on saved causal CV checkpoints.

Uses Ridge-CV probes (config causal.probe) without retraining. Loads each fold's
best_model.pth, re-evaluates on the same held-out indices as train_causal, and
writes new decomposition reports / TSV.

Usage:
  # Single run (phase1 directory)
  python -m experiments.reprobe_decomposition \\
      --run_dir weights/classifier/causal_cv_summary/ADNI/resnet10_disentangled_task13_orth=0.01_age=0.0_test/phase1

  # Batch: all phase1 runs under a root
  python -m experiments.reprobe_decomposition \\
      --runs_root weights/classifier/causal_cv_summary

  # Legacy OLS probe (compare with old reports)
  python -m experiments.reprobe_decomposition --run_dir ... --probe_method ols

  # Compare a run against a fold-matched CE-only reference TSV
  python -m experiments.reprobe_decomposition --run_dir ... \\
      --reference_tsv weights/.../ce_only/phase1/summary/decomposition_summary_ridge.tsv

  # Overwrite decomposition.txt in place
  python -m experiments.reprobe_decomposition --run_dir ... --inplace
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime
from os.path import join as j

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.config import load_config
from experiments.model_registry import CAUSAL_MODELS, build_causal_model
from training.trainer_age import load_age_dataset, _metric_scalar
from training.trainer_v2 import _prepare_model_kwargs
from training.trainer_causal import (
    _resolve_causal_splits,
    _load_baseline_feature_model,
    _eval_baseline_classifier_auc,
)
from training.evaluator import evaluate_on_indices, format_metrics
from training.causal_diagnostics import (
    run_decomposition_analysis,
    save_decomposition_report,
    save_decomposition_tsv,
    decomp_to_tsv_row,
    summarize_cross_fold_pass,
    format_decomposition_report,
)
from Model.causal.probe_utils import resolve_probe_config


def _infer_model_name(run_parent: str) -> str | None:
    """Parse model name from ``.../{model}_task{task}_{descripe}/phaseN``."""
    base = os.path.basename(os.path.normpath(run_parent))
    marker = '_task'
    if marker not in base:
        return None
    candidate = base.split(marker, 1)[0]
    if candidate in CAUSAL_MODELS:
        return candidate
    return None


def _discover_phase_dirs(runs_root: str) -> list[str]:
    pattern = j(runs_root, '**', 'phase*')
    dirs = sorted({
        d for d in glob.glob(pattern, recursive=True)
        if os.path.isdir(d) and glob.glob(j(d, 'fold_*', 'model_best', 'best_model.pth'))
    })
    return dirs


def _build_fold_model_kwargs(cf, causal_cfg, phase, dataset, model_kwargs):
    """Mirror train_causal_cv model kwargs (age heads, etc.)."""
    loss_weights = dict(causal_cfg.get('loss_weights') or cf.get('loss_weights') or {})
    mk = _prepare_model_kwargs(model_kwargs, dataset)
    mk['causal_phase'] = phase

    age_pred_weight = float(loss_weights.get('age_prediction', 0.0) or 0.0)
    age_adv_weight = float(loss_weights.get('age_adversarial', 0.0) or 0.0)
    use_age_prediction = bool(causal_cfg.get('use_age_prediction', False)) or age_pred_weight > 0
    use_age_adversarial = bool(causal_cfg.get('use_age_adversarial', False)) or age_adv_weight > 0

    if use_age_prediction:
        mk['use_age_prediction'] = True
        if 'age_head_hidden' in causal_cfg:
            mk['age_head_hidden'] = causal_cfg['age_head_hidden']
    if use_age_adversarial:
        mk['use_age_adversarial'] = True
        if 'age_adv_hidden' in causal_cfg:
            mk['age_adv_hidden'] = causal_cfg['age_adv_hidden']
        if 'grl_lambda' in causal_cfg:
            mk['grl_lambda'] = causal_cfg['grl_lambda']
    return mk


def _to_float(value):
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_reference_tsv(path: str | None) -> dict[int, dict]:
    """Load a CE-only decomposition TSV keyed by fold number."""
    if not path:
        return {}
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Reference TSV not found: {path}")
    rows = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            fold = row.get('fold')
            if fold in (None, ''):
                continue
            rows[int(float(fold))] = row
    return rows


def _cls_metrics_from_eval(eval_results: dict | None) -> dict:
    """Extract Acc / AD-positive Sen/Spe / F1 / AUC / Loss from evaluate_on_indices."""
    empty = {
        'eval_acc': None, 'eval_sen': None, 'eval_spe': None,
        'eval_f1': None, 'eval_auc': None, 'eval_loss': None,
    }
    if not eval_results:
        return empty

    def _get(*keys):
        for key in keys:
            if eval_results.get(key) is not None:
                return _to_float(_metric_scalar(eval_results[key]))
        return None

    return {
        'eval_acc': _get('accuracy'),
        'eval_sen': _get('recall', 'sen'),
        'eval_spe': _get('specificity', 'spe'),
        'eval_f1': _get('f1'),
        'eval_auc': _get('auc'),
        'eval_loss': _get('loss'),
    }


def _extended_tsv_row(
        fold_idx: int,
        decomp: dict,
        eval_auc=None,
        reference_row: dict | None = None,
        eval_results: dict | None = None,
) -> dict:
    """TSV row with current probes plus optional CE-only reference deltas."""
    row = decomp_to_tsv_row(fold_idx, decomp, eval_auc=eval_auc)
    cls = _cls_metrics_from_eval(eval_results)
    # Prefer evaluate_on_indices AUC when available; keep eval_auc arg as fallback.
    if cls.get('eval_auc') is None and eval_auc is not None:
        cls['eval_auc'] = eval_auc
    row.update(cls)
    # Keep legacy column name used by older summaries.
    row['eval_auc'] = cls.get('eval_auc') if cls.get('eval_auc') is not None else eval_auc

    cond = decomp.get('conditional_age_probe') or {}
    fc_age = decomp.get('fc_age_probe') or {}
    leak = decomp.get('disease_leakage_probe') or {}
    uncond = cond.get('unconditional') or {}
    row['r2_age_fd'] = uncond.get('r2')
    row['r2_age_fc'] = fc_age.get('r2')
    row['r2_y_fc'] = (leak.get('y_from_fc') or {}).get('r2')
    probe_cfg = decomp.get('probe_config') or {}
    row['probe_method'] = uncond.get('probe_method') or probe_cfg.get('method')
    reference_fields = {
        'reference_fold': None,
        'reference_r2_age_fd': None,
        'reference_r2_age_fc': None,
        'reference_eval_auc': None,
        'reference_delta_auc_fc_given_age': None,
        'leakage_idx_vs_reference': None,
        'r2_age_fd_delta_vs_reference': None,
        'r2_age_fc_delta_vs_reference': None,
        'auc_delta_vs_reference': None,
        'delta_auc_fc_given_age_delta_vs_reference': None,
    }
    row.update(reference_fields)
    if reference_row:
        ref_r2_fd = _to_float(reference_row.get('r2_age_fd'))
        ref_r2_fc = _to_float(reference_row.get('r2_age_fc'))
        ref_auc = _to_float(reference_row.get('eval_auc'))
        ref_delta_auc = _to_float(reference_row.get('delta_auc_fc_given_age'))
        cur_r2_fd = _to_float(row.get('r2_age_fd'))
        cur_r2_fc = _to_float(row.get('r2_age_fc'))
        cur_delta_auc = _to_float(row.get('delta_auc_fc_given_age'))
        row['reference_fold'] = reference_row.get('fold')
        row['reference_r2_age_fd'] = ref_r2_fd
        row['reference_r2_age_fc'] = ref_r2_fc
        row['reference_eval_auc'] = ref_auc
        row['reference_delta_auc_fc_given_age'] = ref_delta_auc
        row['leakage_idx_vs_reference'] = (
            cur_r2_fd / ref_r2_fd
            if cur_r2_fd is not None and ref_r2_fd not in (None, 0)
            else None
        )
        row['r2_age_fd_delta_vs_reference'] = (
            cur_r2_fd - ref_r2_fd
            if cur_r2_fd is not None and ref_r2_fd is not None
            else None
        )
        row['r2_age_fc_delta_vs_reference'] = (
            cur_r2_fc - ref_r2_fc
            if cur_r2_fc is not None and ref_r2_fc is not None
            else None
        )
        row['auc_delta_vs_reference'] = (
            eval_auc - ref_auc
            if eval_auc is not None and ref_auc is not None
            else None
        )
        row['delta_auc_fc_given_age_delta_vs_reference'] = (
            cur_delta_auc - ref_delta_auc
            if cur_delta_auc is not None and ref_delta_auc is not None
            else None
        )
    return row


def reprobe_single_run(
        run_dir: str,
        model_name: str,
        device: str = 'cuda',
        config_path: str | None = None,
        out_suffix: str = 'ridge',
        inplace: bool = False,
        skip_baseline: bool = False,
        fold_filter: set[int] | None = None,
        probe_override: dict | None = None,
        reference_tsv: str | None = None,
) -> dict:
    run_dir = os.path.normpath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    cfg_path = config_path or j(run_dir, 'default.yaml')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"No config at {cfg_path}; pass --config_path explicitly."
        )

    cf = load_config(cfg_path)
    causal_cfg = dict(cf.get('causal') or {})
    if probe_override:
        causal_cfg['probe'] = {**(causal_cfg.get('probe') or {}), **probe_override}

    phase_m = re.search(r'phase(\d+)', os.path.basename(run_dir))
    phase = int(phase_m.group(1)) if phase_m else int(causal_cfg.get('phase', 1))

    cv_cfg = cf.get('cross_val') or {}
    age_cfg = cf.get('age_split') or {}
    n_splits = age_cfg.get('num_folds', cv_cfg.get('num_folds', 5))
    test_ratio = age_cfg.get('test_ratio', 0.2)
    random_seed = age_cfg.get('random_seed', cv_cfg.get('random_seed', 42))
    stratify = age_cfg.get('stratify', cv_cfg.get('stratify', True))
    use_standard_cv = test_ratio is not None and test_ratio <= 0
    split_mode = 'standard_cv' if use_standard_cv else 'age_holdout'
    eval_split_name = 'val' if use_standard_cv else 'test'
    auc_margin = float(causal_cfg.get('auc_pass_margin', 0.02))

    if model_name not in CAUSAL_MODELS:
        raise KeyError(f"Unknown causal model: {model_name}. Available: {sorted(CAUSAL_MODELS)}")

    full_dataset = load_age_dataset(cf)
    model_cls, model_kwargs = build_causal_model(model_name, cf, causal_phase=phase)
    fold_model_kwargs = _build_fold_model_kwargs(
        cf, causal_cfg, phase, full_dataset, model_kwargs,
    )
    num_classes = full_dataset.num_classes

    test_idx, folds, _ = _resolve_causal_splits(
        full_dataset, test_ratio, n_splits, random_seed, stratify,
    )

    baseline_feature_model = None
    baseline_classifier_model = None
    if not skip_baseline:
        baseline_feature_model = _load_baseline_feature_model(
            causal_cfg, cf, full_dataset, device,
        )
        baseline_cls_name = causal_cfg.get('baseline_classifier_model')
        if baseline_cls_name:
            try:
                from experiments.model_registry import build_model
                cls_cls, cls_kwargs, _, _ = build_model(baseline_cls_name, cf)
                cls_kwargs = dict(cls_kwargs)
                cls_kwargs['num_classes'] = num_classes
                baseline_classifier_model = cls_cls(**cls_kwargs).to(device)
                baseline_classifier_model.eval()
            except KeyError as exc:
                print(f"  [WARN] baseline classifier unavailable: {exc}")

    probe_label = resolve_probe_config(causal_cfg).get('method', 'ridge_cv')
    reference_rows = _load_reference_tsv(reference_tsv)
    report_prefix = 'decomposition' if inplace else f'decomposition_{out_suffix}'
    summary_dir = j(run_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Reprobe: {run_dir}")
    print(f"  model={model_name} | phase={phase} | probe={probe_label}")
    print(f"  split={split_mode} | eval={eval_split_name}")
    if reference_tsv:
        print(f"  CE-only reference TSV: {os.path.normpath(reference_tsv)}")
    print(f"  output prefix: {report_prefix}.txt")
    print(f"{'='*60}")

    all_decomp = []
    tsv_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_num = fold_idx + 1
        if fold_filter is not None and fold_num not in fold_filter:
            continue

        fold_path = j(run_dir, f'fold_{fold_num}')
        ckpt = j(fold_path, 'model_best', 'best_model.pth')
        if not os.path.isfile(ckpt):
            print(f"  [SKIP] fold {fold_num}: no checkpoint at {ckpt}")
            continue

        eval_idx = val_idx if split_mode == 'standard_cv' else test_idx

        model = model_cls(**fold_model_kwargs).to(device)
        state = torch.load(ckpt, map_location=device)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                f"  [WARN] fold {fold_num} load_state_dict: "
                f"missing={len(incompatible.missing_keys)} "
                f"unexpected={len(incompatible.unexpected_keys)}"
            )
        model.eval()

        eval_results = evaluate_on_indices(
            model, full_dataset, eval_idx, device,
            eval_bc=cf.get('eval_bc', 1), num_classes=num_classes,
            class_names=getattr(full_dataset, 'class_names', None),
        )
        eval_auc = _metric_scalar(eval_results.get('auc')) if eval_results.get('auc') is not None else None

        baseline_auc = None
        if baseline_classifier_model is not None:
            baseline_auc = _eval_baseline_classifier_auc(
                baseline_classifier_model, full_dataset, eval_idx, device,
                cf.get('eval_bc', 1), num_classes,
            )

        decomp = run_decomposition_analysis(
            model, full_dataset, eval_idx, device,
            eval_bc=cf.get('eval_bc', 1),
            num_classes=num_classes,
            baseline_model=baseline_feature_model,
            eval_auc=eval_auc,
            baseline_auc=baseline_auc,
            auc_margin=auc_margin,
            probe_cfg=causal_cfg,
        )
        all_decomp.append(decomp)
        tsv_rows.append(_extended_tsv_row(
            fold_num,
            decomp,
            eval_auc=eval_auc,
            reference_row=reference_rows.get(fold_num),
            eval_results=eval_results,
        ))

        out_eval_dir = j(fold_path, eval_split_name)
        os.makedirs(out_eval_dir, exist_ok=True)
        decomp_path = save_decomposition_report(decomp, out_eval_dir, prefix=report_prefix)
        # Always refresh metrics.txt with Acc/Sen/Spe/F1/AUC for fair comparison.
        metrics_path = j(out_eval_dir, 'metrics.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(format_metrics(eval_results, prefix='') + '\n')
            cm = eval_results.get('cm')
            if cm is not None:
                from training.evaluator import format_confusion_matrix
                f.write(format_confusion_matrix(
                    cm,
                    class_names=getattr(full_dataset, 'class_names', None),
                ) + '\n')
        cls = _cls_metrics_from_eval(eval_results)
        print(
            f"\n  Fold {fold_num}: Acc={cls.get('eval_acc')} "
            f"Sen={cls.get('eval_sen')} Spe={cls.get('eval_spe')} "
            f"F1={cls.get('eval_f1')} AUC={cls.get('eval_auc')} | "
            f"R2(age<-Fd)={(decomp.get('conditional_age_probe') or {}).get('unconditional', {}).get('r2')} | "
            f"R2(age<-Fc)={(decomp.get('fc_age_probe') or {}).get('r2')} | "
            f"dAUC(Fc|age)={(decomp.get('disease_leakage_probe') or {}).get('delta_auc_fc_given_age')}"
        )
        print(f"    -> {decomp_path}")
        print(f"    -> {metrics_path}")
        try:
            print(format_decomposition_report(decomp))
        except UnicodeEncodeError:
            print(format_decomposition_report(decomp).encode('ascii', 'replace').decode('ascii'))

        del model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

    cross_fold = summarize_cross_fold_pass(
        all_decomp,
        min_folds=int(causal_cfg.get('phase1_pass_min_folds', 4)),
    )

    tsv_name = 'decomposition_summary.tsv' if inplace else f'decomposition_summary_{out_suffix}.tsv'
    tsv_path = save_decomposition_tsv(tsv_rows, j(summary_dir, tsv_name))

    summary_name = 'reprobe_summary_inplace.txt' if inplace else f'reprobe_summary_{out_suffix}.txt'
    summary_path = j(summary_dir, summary_name)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Reprobe decomposition ({probe_label})\n")
        f.write(f"  run_dir: {run_dir}\n")
        f.write(f"  model: {model_name}\n")
        f.write(f"  timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  folds_reprobed: {len(all_decomp)}\n")
        if reference_tsv:
            f.write(f"  CE-only reference TSV: {os.path.normpath(reference_tsv)}\n")
        if cross_fold:
            f.write(f"  cross_fold: {cross_fold}\n")
        if tsv_path:
            f.write(f"  TSV: {tsv_path}\n")

    print(f"\n  Summary → {summary_path}")
    if tsv_path:
        print(f"  TSV     → {tsv_path}")

    return {
        'run_dir': run_dir,
        'n_folds': len(all_decomp),
        'cross_fold': cross_fold,
        'tsv_path': tsv_path,
        'summary_path': summary_path,
        'tsv_rows': tsv_rows,
        'model_name': model_name,
        'phase': phase,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Re-run Ridge-CV decomposition on saved causal checkpoints',
    )
    parser.add_argument(
        '--run_dir', type=str, default=None,
        help='Path to phaseN directory (contains fold_1/, summary/)',
    )
    parser.add_argument(
        '--runs_root', type=str, default=None,
        help='Batch mode: reprobe all phase* dirs with checkpoints under this root',
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Causal model name (auto-inferred from run dir name if omitted)',
    )
    parser.add_argument(
        '--config_path', type=str, default=None,
        help='YAML config (default: {run_dir}/default.yaml)',
    )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--out_suffix', type=str, default='ridge',
        help='Suffix for output files (decomposition_{suffix}.txt). Ignored if --inplace',
    )
    parser.add_argument(
        '--inplace', action='store_true',
        help='Overwrite decomposition.txt / decomposition_summary.tsv',
    )
    parser.add_argument(
        '--skip_baseline', action='store_true',
        help='Skip baseline feature/classifier models (faster)',
    )
    parser.add_argument(
        '--folds', type=str, default=None,
        help='Comma-separated fold numbers to reprobe, e.g. 1,3,5',
    )
    parser.add_argument(
        '--probe_method', type=str, default=None,
        choices=['ridge_cv', 'ols'],
        help='Override causal.probe.method from config',
    )
    parser.add_argument(
        '--reference_tsv', type=str, default=None,
        help=(
            'Optional CE-only decomposition_summary TSV for fold-matched '
            'relative leakage and AUC deltas.'
        ),
    )

    args = parser.parse_args()

    if not args.run_dir and not args.runs_root:
        parser.error('Provide --run_dir or --runs_root')

    probe_override = None
    if args.probe_method:
        probe_override = {'method': args.probe_method}

    fold_filter = None
    if args.folds:
        fold_filter = {int(x.strip()) for x in args.folds.split(',') if x.strip()}

    run_dirs = []
    if args.run_dir:
        run_dirs.append(os.path.normpath(args.run_dir))
    if args.runs_root:
        run_dirs.extend(_discover_phase_dirs(args.runs_root))

    if not run_dirs:
        print('No run directories found.')
        sys.exit(1)

    failures = []
    for run_dir in run_dirs:
        model_name = args.model or _infer_model_name(os.path.dirname(run_dir))
        if model_name is None:
            msg = f"Cannot infer model name for {run_dir}; pass --model"
            print(f"[ERROR] {msg}")
            failures.append(msg)
            continue
        try:
            reprobe_single_run(
                run_dir=run_dir,
                model_name=model_name,
                device=args.device,
                config_path=args.config_path,
                out_suffix=args.out_suffix,
                inplace=args.inplace,
                skip_baseline=args.skip_baseline,
                fold_filter=fold_filter,
                probe_override=probe_override,
                reference_tsv=args.reference_tsv,
            )
        except Exception as exc:
            print(f"[ERROR] {run_dir}: {exc}")
            failures.append(str(exc))

    if failures:
        print(f"\nCompleted with {len(failures)} error(s).")
        sys.exit(1)
    print(f"\nDone. Reprobed {len(run_dirs)} run(s).")


if __name__ == '__main__':
    main()
