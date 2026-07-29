"""Update gamma_mechanism_5fold_review.md table from gamma_mech_5fold_summary.tsv."""
from __future__ import annotations

import csv
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

REVIEW = os.path.join(PROJECT_ROOT, 'utils', 'review', 'gamma_mechanism_5fold_review.md')
TSV = os.path.join(PROJECT_ROOT, 'weights', 'classifier', 'gamma_mech_5fold_summary.tsv')

ID_PATTERNS = [
    ('W0_P1', 'mech_W0_P1'),
    ('W0_CE', 'mech_W0_CE'),
    ('W0_FiLM', 'mech_W0_FiLM'),
    ('W0_Main', 'mech_W0_Main'),
    ('A1', 'mech_A1'),
    ('A3', 'mech_A3'),
    ('A5a', 'mech_A5a'),
    ('A5b', 'mech_A5b'),
    ('A6', 'mech_A6'),
    ('A8', 'mech_A8'),
    ('A10', 'mech_A10'),
]


def _fmt(mean, std):
    if mean is None or mean == '':
        return '_pending_'
    try:
        m = float(mean)
        s = float(std) if std not in (None, '') else 0.0
        return f'{m:.3f} ± {s:.3f}'
    except (TypeError, ValueError):
        return '_pending_'


def load_rows():
    if not os.path.isfile(TSV):
        return {}
    rows = {}
    with open(TSV, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            tag = r.get('tag') or r.get('run_dir') or ''
            for eid, key in ID_PATTERNS:
                if key in tag:
                    rows[eid] = r
    return rows


def judge(rows):
    def auc(eid):
        r = rows.get(eid)
        if not r or r.get('test_auc_mean') in (None, ''):
            return None
        return float(r['test_auc_mean'])

    main, a1, a3, a8, a10 = map(auc, ['W0_Main', 'A1', 'A3', 'A8', 'A10'])
    if None in (main, a1, a3, a8, a10):
        return '_pending_ (need W0_Main + Wave1)'
    bits = []
    if abs(a1 - main) <= 0.01 and (main - a3) <= 0.01:
        bits.append('H1 (capacity/constant gate)')
    if (main - a3) > 0.02 and (main - a1) > 0.02:
        bits.append('H2 (age-conditioned gate)')
    if (main - a10) > 0.03:
        bits.append('H3 (Phase1 init)')
    if (main - a8) > 0.015:
        bits.append('H4 (confounder supervision)')
    return '; '.join(bits) if bits else 'inconclusive on pre-registered thresholds'


def main():
    rows = load_rows()
    lines = [
        '| Run | test AUC (mean±std) | test Acc | γ mean | corr(γ,age) | 判定线索 |',
        '|-----|---------------------|----------|--------|-------------|---------|',
    ]
    clues = {
        'W0_Main': '主方法', 'A1': 'H1', 'A3': 'H2', 'A8': 'H4', 'A10': 'H3',
    }
    for eid, _ in ID_PATTERNS:
        r = rows.get(eid, {})
        auc = _fmt(r.get('test_auc_mean'), r.get('test_auc_std'))
        acc = _fmt(r.get('test_acc_mean'), r.get('test_acc_std'))
        gm = r.get('gamma_mean_mean')
        gm_s = f"{float(gm):.3f}" if gm not in (None, '') else '—'
        corr = r.get('corr_gamma_age_mean')
        corr_s = f"{float(corr):.3f}" if corr not in (None, '') else '—'
        lines.append(
            f"| {eid} | {auc} | {acc} | {gm_s} | {corr_s} | {clues.get(eid, '')} |"
        )
    table = '\n'.join(lines)
    verdict = judge(rows)

    text = open(REVIEW, 'r', encoding='utf-8').read()
    # Replace section 3 table between markers
    pattern = (
        r'(## 3\. 主结果表.*?\n\n)'
        r'(?:\| Run \|.*?\n(?:\|.*?\n)*)'
        r'(\n机制判定（待填）：\*\*).*?(\*\*)'
    )
    repl = rf'\g<1>{table}\n\g<2>{verdict}\g<3>'
    new_text, n = re.subn(pattern, repl, text, count=1, flags=re.S)
    if n == 0:
        # fallback: replace simple pending table
        new_text = text.replace('_pending_', table.split('\n', 2)[-1] if False else text)
        # hard replace from "| Run |" block
        start = text.find('| Run | test AUC')
        end = text.find('机制判定')
        if start != -1 and end != -1:
            new_text = text[:start] + table + '\n\n' + text[end:]
            new_text = re.sub(
                r'机制判定（待填）：\*\*_pending_\*\*',
                f'机制判定（待填）：**{verdict}**',
                new_text,
            )
    open(REVIEW, 'w', encoding='utf-8').write(new_text)
    print(f'Updated {REVIEW}')
    print('Verdict:', verdict)


if __name__ == '__main__':
    main()
