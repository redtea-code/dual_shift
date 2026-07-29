"""Fill gamma_vs_fusion_5fold_review.md from gamma_fusion_5fold_summary.tsv."""
from __future__ import annotations

import csv
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVIEW = os.path.join(PROJECT_ROOT, 'utils', 'review', 'gamma_vs_fusion_5fold_review.md')
TSV = os.path.join(PROJECT_ROOT, 'weights', 'classifier', 'gamma_fusion_5fold_summary.tsv')

ORDER = [
    'B0', 'B_gamma', 'B_film', 'B_daft', 'B_hyper', 'B_concat',
    'L0', 'L1', 'L2', 'A_global', 'A_noshare', 'A_shuffle',
]
NOTES = {
    'B0': 'CE-only', 'B_gamma': '主方法', 'B_film': 'FiLM last',
    'B_daft': '', 'B_hyper': '', 'B_concat': '',
    'L0': 'CE-only γ', 'L1': '+sparsity', 'L2': '+smooth',
    'A_global': '', 'A_noshare': '', 'A_shuffle': '',
}


def main():
    rows = {}
    if os.path.isfile(TSV):
        with open(TSV, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f, delimiter='\t'):
                eid = r.get('exp_id') or ''
                if eid:
                    rows[eid] = r

    lines = [
        '| Run | test AUC | test Acc | γ mean | corr(γ,age) | 备注 |',
        '|-----|----------|----------|--------|-------------|------|',
    ]
    for eid in ORDER:
        r = rows.get(eid, {})
        def fmt(m, s):
            if m in (None, ''):
                return '_pending_'
            try:
                return f'{float(m):.3f} ± {float(s or 0):.3f}'
            except ValueError:
                return '_pending_'
        auc = fmt(r.get('test_auc_mean'), r.get('test_auc_std'))
        acc = fmt(r.get('test_acc_mean'), r.get('test_acc_std'))
        gm = r.get('gamma_mean_mean')
        gm_s = f'{float(gm):.3f}' if gm not in (None, '') else '—'
        cr = r.get('corr_gamma_age_mean')
        cr_s = f'{float(cr):.3f}' if cr not in (None, '') else '—'
        lines.append(
            f"| {eid} | {auc} | {acc} | {gm_s} | {cr_s} | {NOTES.get(eid, '')} |"
        )
    table = '\n'.join(lines)

    # Verdict
    def auc(eid):
        r = rows.get(eid)
        if not r or r.get('test_auc_mean') in (None, ''):
            return None
        return float(r['test_auc_mean'])

    def acc(eid):
        r = rows.get(eid)
        if not r or r.get('test_acc_mean') in (None, ''):
            return None
        return float(r['test_acc_mean'])

    g, film, concat, shuffle, l0 = map(
        auc, ['B_gamma', 'B_film', 'B_concat', 'A_shuffle', 'L0'],
    )
    bits = []
    if None not in (g, shuffle) and (g - shuffle) > 0.02:
        bits.append('A_shuffle 掉点 → γ 依赖表格')
    elif None not in (g, shuffle) and abs(g - shuffle) <= 0.01:
        bits.append('A_shuffle≈B_gamma(AUC) → 表格条件信号弱/不稳定')
    if None not in (g, l0) and abs(g - l0) <= 0.01:
        bits.append('L0≈B_gamma → 优势主要来自门结构')
    elif None not in (g, l0) and (l0 - g) > 0.015:
        bits.append('L0>B_gamma → 默认正则伤害；优势来自γ门而非 TV/L1')
    elif None not in (g, l0) and (g - l0) > 0.015:
        bits.append('B_gamma≫L0 → 正则损失有贡献')
    baselines = [x for x in (film, concat, auc('B_daft'), auc('B_hyper')) if x is not None]
    if g is not None and baselines and g > max(baselines) + 0.005:
        bits.append('B_gamma 高于 FiLM/DAFT/Hyper/Concat')
    if auc('A_global') is not None and g is not None and (auc('A_global') - g) > 0.01:
        bits.append('A_global≥patch → 空间门非必要')
    verdict = '; '.join(bits) if bits else '_pending_'

    text = open(REVIEW, 'r', encoding='utf-8').read()
    start = text.find('| Run | test AUC')
    end = text.find('机制判定')
    if start != -1 and end != -1:
        text = text[:start] + table + '\n\n' + text[end:]
    text = re.sub(
        r'机制判定（待填）：\*\*.*?\*\*',
        f'机制判定：**{verdict}**',
        text,
        count=1,
    )
    # Refresh one-liner with numbers if table present
    one = (
        f'在 std5cv 下 B_gamma test AUC={g:.3f}；'
        f'相对融合基线最高≈{max(baselines):.3f}；'
        f'L0={l0:.3f}（正则未帮助）；'
        f'A_shuffle={shuffle:.3f}。'
        if None not in (g, l0, shuffle) and baselines else
        '本线只比较 ResNet+patch-γ 与同协议表格融合基线；不把 Phase1 解耦写进方法贡献。'
    )
    text = re.sub(
        r'## 5\. 一句话\n\n\*\*.*?\*\*',
        f'## 5. 一句话\n\n**{one}**',
        text,
        count=1,
        flags=re.DOTALL,
    )
    open(REVIEW, 'w', encoding='utf-8').write(text)
    print('Updated', REVIEW)
    print('Verdict:', verdict)


if __name__ == '__main__':
    main()
