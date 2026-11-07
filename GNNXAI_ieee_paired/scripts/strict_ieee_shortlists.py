"""Short-list sensitivity for the paired IEEE-CIS SHAP comparisons."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.instance_level_refitlevel import _topk_overlap

SEEDS = range(201, 221)
SPLITS = ('random', 'temporal')
LENGTHS = (3, 5, 10)


def from_checkpoints(directory):
    rows = []
    temporal_ids = None
    for seed in SEEDS:
        for split in SPLITS:
            with np.load(directory / f'{seed}_{split}.npz') as fit:
                ids = fit['ids']
                if ids.shape != (5000,) or len(np.unique(ids)) != 5000 or int(fit['p']) != 486:
                    raise ValueError(f'Unexpected evaluation sample at {seed}/{split}')
                if split == 'temporal':
                    if temporal_ids is None:
                        temporal_ids = ids
                    np.testing.assert_array_equal(ids, temporal_ids)
                tab, graph, control = (fit[name] for name in ('tab', 'hyb', 'cov'))
                for arm, width in [(tab, 486), (graph, 550), (control, 550)]:
                    if arm.shape != (5000, width) or not np.isfinite(arm).all():
                        raise ValueError(f'Invalid SHAP array at {seed}/{split}')
                for k in LENGTHS:
                    g = _topk_overlap(tab, graph[:, :486], k)
                    c = _topk_overlap(tab, control[:, :486], k)
                    rows.append((seed, split, k, len(ids), g, c))
    return pd.DataFrame(rows, columns=['seed', 'split', 'k', 'payments', 'graph', 'gaussian'])


def summarize(rows, directory):
    detail = rows.set_index(['seed', 'split', 'k']).sort_index()
    expected = pd.MultiIndex.from_product([SEEDS, SPLITS, LENGTHS], names=detail.index.names)
    if not detail.index.equals(expected) or not detail.payments.eq(5000).all():
        raise ValueError('Incomplete or duplicated evaluation samples')
    for arm in ('graph', 'gaussian'):
        if not detail[arm].between(0, 1).all():
            raise ValueError(f'Invalid overlap for {arm}')
    original = pd.read_csv(directory / 'plot_comparisons_by_seed.csv').set_index('seed')
    paired = pd.read_csv(directory / 'paired.csv').set_index('seed')
    summaries = []
    for split in SPLITS:
        for k in LENGTHS:
            sample = detail.xs((split, k), level=('split', 'k'))
            delta = 100 * (sample.graph - sample.gaussian)
            if k == 10:
                for col, arm in [('graph', 'hyb'), ('gaussian', 'cov')]:
                    np.testing.assert_allclose(10 * sample[col], original.loc[sample.index, f'{split}_{arm}'], rtol=0, atol=1e-10)
                np.testing.assert_allclose(delta, 100 * paired.loc[sample.index, f'top10_{split}'], rtol=0, atol=1e-9)
            # The four new contrasts are the two short lists in each regime.
            comparisons = 1 if k == 10 else 4
            half = stats.t.ppf(1 - .05 / (2 * comparisons), len(delta) - 1) * stats.sem(delta)
            summaries.append((split, k, len(delta), k * sample.graph.mean(), k * sample.gaussian.mean(),
                              delta.mean(), delta.mean() - half, delta.mean() + half, comparisons))
    return pd.DataFrame(summaries, columns=['split', 'k', 'pairs', 'graph_fields', 'gaussian_fields',
                                            'difference_pp', 'lower_pp', 'upper_pp', 'family_size'])


def write_table(summary, path):
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Validation & List length & Graph & Gaussian & Difference in pp (95\% interval) \\',
             r'\midrule']
    for row in summary.itertuples():
        if row.k == 10:
            continue
        label = 'Random' if row.split == 'random' else 'Forward-time'
        lines.append(f'{label} & {row.k} & {row.graph_fields:.2f} & {row.gaussian_fields:.2f} & '
                     f'\u0024{row.difference_pp:.2f}~[{row.lower_pp:.2f}, {row.upper_pp:.2f}]\u0024'
                     + r' \\')
    path.write_text('\n'.join(lines + [r'\bottomrule', r'\end{tabular}', '']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'results/strict_ieee/primary')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()
    out = args.output or args.directory
    out.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        rows = pd.read_csv(args.directory / 'shortlists_by_seed.csv')
    else:
        rows = from_checkpoints(args.directory)
    summary = summarize(rows, args.directory)
    if not args.summary_only:
        rows.to_csv(out / 'shortlists_by_seed.csv', index=False)
    summary.to_csv(out / 'shortlists_summary.csv', index=False)
    write_table(summary, out / 'strict_ieee_shortlists.tex')
    print(summary.to_string(index=False))
