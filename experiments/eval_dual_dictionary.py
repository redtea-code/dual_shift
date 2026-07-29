"""Evaluate dual-dictionary checkpoints and aggregate baseline comparisons."""
from __future__ import annotations

import argparse
import json
import os
import sys
from os.path import join as j

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from Model.dictionary.model_factory import build_dictionary_model
from training.dictionary_diagnostics import (
    collect_dictionary_outputs,
    run_dictionary_diagnostics,
    save_dictionary_report,
)
from training.dictionary_stability import dictionary_stability_report, atom_activation_stats
from utils.config import load_config
from utils.dictionary_splits import load_split_manifest
from utils.feature_cache import fold_cache_dir, load_feature_split


def _aggregate_reports(reports: list) -> dict:
    if not reports:
        return {}
    keys = ['test_accuracy', 'test_auc', 'partial_r2_age', 'delta_auc_diag']
    agg = {}
    for key in keys:
        vals = [r.get(key) for r in reports if r.get(key) is not None]
        if vals:
            agg[f'{key}_mean'] = float(np.mean(vals))
            agg[f'{key}_std'] = float(np.std(vals))
    agg['n_folds'] = len(reports)
    return agg


def _cross_fold_stability(run_root: str, cache_root: str, fold_ids: list, dict_cfg: dict, device: str):
    """Document §9: principal angles / Hungarian matching across folds."""
    refs = {}
    for fold_idx in fold_ids:
        ckpt = j(run_root, f'fold_{fold_idx}', 'model_best', 'best_model.pth')
        if not os.path.isfile(ckpt):
            continue
        fold_dir = fold_cache_dir(cache_root, fold_idx)
        train_npz = load_feature_split(j(fold_dir, 'features_train.npz'))
        num_classes = int(np.max(train_npz['labels'])) + 1
        model = build_dictionary_model(
            model_type=dict_cfg.get('model_type', 'dual'),
            feature_dim=int(dict_cfg.get('feature_dim', 64)),
            k_d=int(dict_cfg.get('k_d', 16)),
            k_c=int(dict_cfg.get('k_c', 16)),
            num_classes=num_classes,
            encoder_type=dict_cfg.get('encoder_type', 'joint_lista'),
            lista_steps=int(dict_cfg.get('lista_steps', 5)),
        )
        state = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state, strict=False)
        D = model.D.detach().cpu().numpy() if hasattr(model, 'D') else None
        C = model.C.detach().cpu().numpy() if hasattr(model, 'C') else None
        outs = collect_dictionary_outputs(
            model, j(fold_dir, 'features_test.npz'), device=device,
        )
        refs[fold_idx] = {
            'D': D, 'C': C,
            'Q_d': outs['Q_d'], 'Q_c': outs['Q_c'],
        }

    if len(refs) < 2:
        return {'n_folds_compared': len(refs), 'pairs': []}

    fold_list = sorted(refs.keys())
    pairs = []
    ref = fold_list[0]
    for other in fold_list[1:]:
        Da, Ca = refs[ref]['D'], refs[ref]['C']
        Db, Cb = refs[other]['D'], refs[other]['C']
        if Da is None or Db is None:
            continue
        if Ca is not None and Ca.size == 0:
            Ca = Da[:, :0]
        if Cb is not None and Cb.size == 0:
            Cb = Db[:, :0]
        rep = dictionary_stability_report(
            Da, Ca if Ca is not None else Da[:, :0],
            Db, Cb if Cb is not None else Db[:, :0],
            refs[ref]['Q_d'], refs[other]['Q_d'],
        )
        rep['fold_a'] = ref
        rep['fold_b'] = other
        pairs.append(rep)

    activation = {
        f'fold_{fid}': {
            'Q_d': atom_activation_stats(refs[fid]['Q_d']),
            'Q_c': atom_activation_stats(refs[fid]['Q_c']),
        }
        for fid in fold_list
    }
    return {
        'n_folds_compared': len(refs),
        'reference_fold': ref,
        'pairs': pairs,
        'activation': activation,
    }


def eval_dictionary_run(
        run_root: str,
        cache_root: str,
        config_path: str = 'config/default.yaml',
        device: str = 'cuda',
        fold: int = None,
):
    cf = load_config(config_path)
    dict_cfg = cf.get('dictionary') or {}
    manifest = load_split_manifest(cache_root)
    fold_ids = [fold] if fold else [f['fold_idx'] for f in manifest['folds']]

    all_reports = []
    for fold_idx in fold_ids:
        ckpt = j(run_root, f'fold_{fold_idx}', 'model_best', 'best_model.pth')
        if not os.path.isfile(ckpt):
            print(f"  [WARN] missing checkpoint for fold {fold_idx}")
            continue

        fold_dir = fold_cache_dir(cache_root, fold_idx)
        train_npz = load_feature_split(j(fold_dir, 'features_train.npz'))
        num_classes = int(np.max(train_npz['labels'])) + 1

        model = build_dictionary_model(
            model_type=dict_cfg.get('model_type', 'dual'),
            feature_dim=int(dict_cfg.get('feature_dim', 64)),
            k_d=int(dict_cfg.get('k_d', 16)),
            k_c=int(dict_cfg.get('k_c', 16)),
            num_classes=num_classes,
            encoder_type=dict_cfg.get('encoder_type', 'joint_lista'),
            lista_steps=int(dict_cfg.get('lista_steps', 5)),
        )
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        report = run_dictionary_diagnostics(
            model, fold_dir, device=device, num_classes=num_classes,
        )
        report['fold'] = fold_idx
        out_dir = j(run_root, f'fold_{fold_idx}', 'test')
        save_dictionary_report(report, out_dir)
        all_reports.append(report)
        print(
            f"Fold {fold_idx}: test_acc={report['test_accuracy']:.4f} "
            f"test_auc={report.get('test_auc')} "
            f"partial_r2_age={report.get('partial_r2_age')}"
        )

    summary_dir = j(run_root, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    agg = _aggregate_reports(all_reports)

    stability = {}
    if fold is None and len(fold_ids) > 1:
        try:
            stability = _cross_fold_stability(
                run_root, cache_root, fold_ids, dict_cfg, device,
            )
            stab_path = j(summary_dir, 'dictionary_stability.json')
            with open(stab_path, 'w', encoding='utf-8') as f:
                json.dump(stability, f, indent=2, default=str)
            print(f"Saved cross-fold stability → {stab_path}")
        except Exception as exc:
            print(f"  [WARN] Stability report failed: {exc}")
            stability = {'error': str(exc)}

    agg_path = j(summary_dir, 'dictionary_eval_summary.txt')
    with open(agg_path, 'w', encoding='utf-8') as f:
        f.write('Dictionary held-out evaluation\n')
        for r in all_reports:
            f.write(
                f"fold_{r['fold']}: acc={r['test_accuracy']:.4f} "
                f"auc={r.get('test_auc')} "
                f"partial_r2={r.get('partial_r2_age')} "
                f"delta_auc_diag={r.get('delta_auc_diag')}\n"
            )
        f.write('\nCross-fold aggregate:\n')
        for k, v in agg.items():
            f.write(f"  {k}: {v}\n")
        if stability.get('pairs'):
            f.write('\nCross-fold dictionary stability (vs fold 1):\n')
            for p in stability['pairs']:
                d_ang = (p.get('D') or {}).get('principal_angles_deg_mean')
                c_ang = (p.get('C') or {}).get('principal_angles_deg_mean')
                f.write(
                    f"  fold_{p.get('fold_a')}_vs_{p.get('fold_b')}: "
                    f"D_angle_mean={d_ang} C_angle_mean={c_ang}\n"
                )

    json_path = j(summary_dir, 'dictionary_eval_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(
            {'folds': all_reports, 'aggregate': agg, 'stability': stability},
            f, indent=2, default=str,
        )

    print(f"Saved aggregate summary → {agg_path}")
    return all_reports


def main():
    parser = argparse.ArgumentParser(description='Evaluate dual-dictionary checkpoints')
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument('--run_root', type=str, required=True)
    parser.add_argument('--cache_root', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fold', type=int, default=None)
    args = parser.parse_args()

    eval_dictionary_run(
        run_root=args.run_root,
        cache_root=args.cache_root,
        config_path=args.config_path,
        device=args.device,
        fold=args.fold,
    )


if __name__ == '__main__':
    main()
