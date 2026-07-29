"""
Re-evaluate dictionary CV checkpoints with comprehensive metrics.

Loads each fold's best_model.pth against the matching feature cache, rewrites
fold_*/test/dictionary_summary.{json,txt}, and rebuilds master TSV registries.

Usage:
  # Single run
  python -m experiments.reprobe_dictionary \\
      --run_root weights/classifier/dictionary_cv_summary/ADNI/<run_id>

  # All ADNI dictionary runs
  python -m experiments.reprobe_dictionary \\
      --runs_root weights/classifier/dictionary_cv_summary/ADNI \\
      --device cuda
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from os.path import join as j

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from Model.dictionary.model_factory import build_dictionary_model
from training.dictionary_diagnostics import (
    run_dictionary_diagnostics,
    save_dictionary_report,
)
from utils.dictionary_experiment_registry import (
    collect_fold_rows_from_run,
    save_dictionary_experiment_tables,
    summarize_run_rows,
    write_master_tables,
)
from utils.feature_cache import fold_cache_dir, load_feature_split

CACHE_BASE = 'resnet10_ce_only_task13_dim=64_orth=0.05_age=0.01_test'
POSTFIX_PRESETS = {
    'dual_dict_core',
    'dual_dict_no_age',
    'dual_dict_linear',
    'dual_dict_mlp',
    'single_dict_cls',
    'dual_branch_mlp',
}

# Canonical age-holdout run_id suffixes included in the postfix table.
POSTFIX_RUN_SUFFIXES = (
    '_joint_lista_kd16_kc16_dual_dict_core',
    '_joint_lista_kd16_kc16_dual_dict_no_age',
    '_linear_kd16_kc16_dual_dict_linear',
    '_mlp_kd16_kc16_dual_dict_mlp',
    '_single_kd16_kc0_single_dict_cls',
    '_dual_branch_mlp_kd16_kc16_dual_branch_mlp',
)


def _norm(path: str) -> str:
    return os.path.normpath(path).replace('\\', '/')


def parse_run_meta(run_id: str, cache_parent: str) -> dict:
    """Infer cache_root / encoder / dims / model_type / preset from run directory name."""
    full_cv = '_full_cv_' in run_id or run_id.endswith('_full_cv') or '_test_full_cv_' in run_id
    cache_name = f'{CACHE_BASE}_full_cv' if full_cv else CACHE_BASE
    cache_root = j(cache_parent, cache_name)

    kd_m = re.search(r'_kd(\d+)', run_id)
    kc_m = re.search(r'_kc(\d+)', run_id)
    k_d = int(kd_m.group(1)) if kd_m else 16
    k_c = int(kc_m.group(1)) if kc_m else 16

    model_type = 'dual'
    encoder_type = 'joint_lista'
    preset = 'dual_dict_core'

    if '_single_kd' in run_id or run_id.endswith('single_dict_cls'):
        model_type = 'single'
        preset = 'single_dict_cls'
        encoder_type = 'joint_lista'
    elif '_dual_branch_mlp_' in run_id or run_id.endswith('dual_branch_mlp'):
        model_type = 'dual_branch_mlp'
        preset = 'dual_branch_mlp'
        encoder_type = 'dual_branch_mlp'
    elif '_linear_ae_' in run_id or run_id.endswith('linear_ae'):
        model_type = 'linear_ae'
        preset = 'linear_ae'
        encoder_type = 'linear_ae'
    elif '_linear_kd' in run_id or 'dual_dict_linear' in run_id:
        encoder_type = 'linear'
        preset = 'dual_dict_linear'
    elif '_mlp_kd' in run_id or 'dual_dict_mlp' in run_id:
        encoder_type = 'mlp'
        preset = 'dual_dict_mlp'
    elif 'dual_dict_no_age' in run_id:
        encoder_type = 'joint_lista'
        preset = 'dual_dict_no_age'
    elif 'dual_dict_core' in run_id:
        encoder_type = 'joint_lista'
        preset = 'dual_dict_core'
    elif 'dims_8_8' in run_id:
        encoder_type = 'joint_lista'
        preset = 'dims_8_8'
    elif 'dims_24_16' in run_id:
        encoder_type = 'joint_lista'
        preset = 'dims_24_16'
    elif '_joint_lista_' in run_id:
        encoder_type = 'joint_lista'
        preset = 'dual_dict_core'

    if full_cv and preset in ('dual_dict_core', 'dual_dict_no_age'):
        # keep preset name; notes mark protocol
        pass

    return {
        'cache_root': cache_root,
        'model_type': model_type,
        'encoder_type': encoder_type,
        'k_d': k_d,
        'k_c': k_c,
        'preset': preset,
        'full_cv': full_cv,
        'feature_dim': 64,
        'lista_steps': 5,
    }


def discover_run_dirs(runs_root: str) -> list[str]:
    if not os.path.isdir(runs_root):
        return []
    dirs = []
    for name in sorted(os.listdir(runs_root)):
        path = j(runs_root, name)
        if not os.path.isdir(path):
            continue
        has_ckpt = any(
            os.path.isfile(j(path, d, 'model_best', 'best_model.pth'))
            for d in os.listdir(path)
            if d.startswith('fold_') and os.path.isdir(j(path, d))
        )
        if has_ckpt:
            dirs.append(path)
    return dirs


def _fold_ids(run_root: str) -> list[int]:
    ids = []
    for name in os.listdir(run_root):
        if name.startswith('fold_') and os.path.isdir(j(run_root, name)):
            ids.append(int(name.split('_', 1)[1]))
    return sorted(ids)


def reprobe_run(
        run_root: str,
        *,
        cache_parent: str,
        device: str = 'cuda',
        project_dir: str = 'weights/classifier',
        update_master: bool = False,
) -> dict:
    run_id = os.path.basename(run_root.rstrip('/\\'))
    meta = parse_run_meta(run_id, cache_parent)
    cache_root = meta['cache_root']
    if not os.path.isdir(cache_root):
        raise FileNotFoundError(f'cache_root missing: {cache_root}')

    device_obj = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    all_reports = []
    for fold_idx in _fold_ids(run_root):
        ckpt = j(run_root, f'fold_{fold_idx}', 'model_best', 'best_model.pth')
        if not os.path.isfile(ckpt):
            print(f'  [WARN] missing checkpoint fold {fold_idx}')
            continue
        fold_dir = fold_cache_dir(cache_root, fold_idx)
        train_npz = load_feature_split(j(fold_dir, 'features_train.npz'))
        num_classes = int(np_max_label(train_npz['labels'])) + 1

        enc = meta['encoder_type']
        if meta['model_type'] in ('dual_branch_mlp', 'linear_ae'):
            enc = 'joint_lista'  # unused by baselines; factory requires a string
        model = build_dictionary_model(
            model_type=meta['model_type'],
            feature_dim=int(meta['feature_dim']),
            k_d=int(meta['k_d']),
            k_c=int(meta['k_c']),
            num_classes=num_classes,
            encoder_type=enc,
            lista_steps=int(meta['lista_steps']),
        )
        state = torch.load(ckpt, map_location=device_obj)
        model.load_state_dict(state, strict=False)
        model.to(device_obj)

        report = run_dictionary_diagnostics(
            model, fold_dir, device=str(device_obj), num_classes=num_classes,
        )
        report['fold'] = fold_idx
        save_dictionary_report(report, j(run_root, f'fold_{fold_idx}', 'test'))
        all_reports.append(report)
        print(
            f'  fold {fold_idx}: acc={report["test_accuracy"]:.4f} '
            f'sen={report.get("test_sen"):.4f} spe={report.get("test_spe"):.4f} '
            f'f1={report.get("test_f1"):.4f} auc={report.get("test_auc"):.4f} '
            f'pr2={report.get("partial_r2_age")} dAUC={report.get("delta_auc_diag")}'
        )

    notes = 'full_cv' if meta['full_cv'] else 'age_holdout'
    fold_rows = collect_fold_rows_from_run(
        run_root,
        run_id=run_id,
        preset=meta['preset'],
        cache_root=_norm(cache_root),
        encoder_type=meta['encoder_type'],
        notes=notes,
    )
    for row in fold_rows:
        row['encoder_type'] = meta['encoder_type']
        row['k_d'] = meta['k_d']
        row['k_c'] = meta['k_c']
        row['run_root'] = _norm(run_root)
        row['cache_root'] = _norm(cache_root)

    paths = save_dictionary_experiment_tables(
        run_root,
        fold_rows,
        project_dir=project_dir,
        encoder_type=meta['encoder_type'],
        update_master=update_master,
    )
    return {
        'run_id': run_id,
        'meta': meta,
        'fold_rows': fold_rows,
        'run_row': paths.get('run_row') or summarize_run_rows(
            fold_rows, encoder_type=meta['encoder_type'],
        ),
        'n_folds': len(fold_rows),
    }


def np_max_label(labels) -> int:
    import numpy as np
    return int(np.max(labels))


def rebuild_master_tables(
        results: list[dict],
        project_dir: str,
):
    all_fold_rows = []
    all_run_rows = []
    postfix_fold_rows = []
    postfix_run_rows = []

    for res in results:
        fold_rows = res['fold_rows']
        run_row = res['run_row']
        all_fold_rows.extend(fold_rows)
        if run_row:
            all_run_rows.append(run_row)

        preset = (run_row or {}).get('preset') or res['meta']['preset']
        full_cv = res['meta']['full_cv']
        run_id = res['run_id']
        is_postfix_run = (not full_cv) and any(
            run_id.endswith(suf) for suf in POSTFIX_RUN_SUFFIXES
        )
        if is_postfix_run and preset in POSTFIX_PRESETS:
            tagged_folds = []
            for row in fold_rows:
                r = dict(row)
                if not str(r.get('preset', '')).endswith('_postfix'):
                    r['preset'] = f"{preset}_postfix"
                tagged_folds.append(r)
            tagged_run = dict(run_row) if run_row else {}
            if tagged_run:
                tagged_run['preset'] = f"{preset}_postfix"
            postfix_fold_rows.extend(tagged_folds)
            if tagged_run:
                postfix_run_rows.append(tagged_run)

    write_master_tables(
        project_dir, all_fold_rows, all_run_rows,
        fold_name='dictionary_experiments_per_fold.tsv',
        run_name='dictionary_experiments.tsv',
    )
    write_master_tables(
        project_dir, postfix_fold_rows, postfix_run_rows,
        fold_name='dictionary_experiments_postfix_per_fold.tsv',
        run_name='dictionary_experiments_postfix.tsv',
    )
    print(
        f'Wrote master tables: {len(all_run_rows)} runs / {len(all_fold_rows)} folds; '
        f'postfix {len(postfix_run_rows)} runs / {len(postfix_fold_rows)} folds'
    )


def main():
    parser = argparse.ArgumentParser(description='Reprobe dictionary checkpoints')
    parser.add_argument('--run_root', type=str, default=None)
    parser.add_argument(
        '--runs_root', type=str,
        default='weights/classifier/dictionary_cv_summary/ADNI',
    )
    parser.add_argument(
        '--cache_parent', type=str,
        default='weights/classifier/dictionary_feature_cache/ADNI',
    )
    parser.add_argument('--project_dir', type=str, default='weights/classifier')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fold', type=int, default=None, help='unused; all folds')
    args = parser.parse_args()

    if args.run_root:
        run_dirs = [args.run_root]
    else:
        run_dirs = discover_run_dirs(args.runs_root)

    if not run_dirs:
        print('No dictionary runs found.')
        return

    results = []
    for run_root in run_dirs:
        print(f'\n=== Reprobe {os.path.basename(run_root)} ===')
        try:
            res = reprobe_run(
                run_root,
                cache_parent=args.cache_parent,
                device=args.device,
                project_dir=args.project_dir,
                update_master=False,
            )
            results.append(res)
        except Exception as exc:
            print(f'  [ERROR] {exc}')
            raise

    rebuild_master_tables(results, args.project_dir)
    print(f'\nDone. Reprobed {len(results)} runs.')


if __name__ == '__main__':
    main()
