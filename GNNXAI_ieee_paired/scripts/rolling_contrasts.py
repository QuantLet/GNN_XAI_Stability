"""Audit the fixed panel and compute paired temporal contrasts from saved fits."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from report_chronological import ARMS, compute, top_three


def concordant_pairs(scores, y, day, n_days):
    """Count correctly ordered positive-negative pairs by their two calendar days."""
    matrix = np.zeros((n_days, n_days))
    for j in range(n_days):
        negatives = np.sort(scores[(day == j) & (y == 0)])
        positive = scores[y == 1]
        wins = (negatives.searchsorted(positive, 'left') + negatives.searchsorted(positive, 'right')) / 2
        matrix[:, j] = np.bincount(day[y == 1], weights=wins, minlength=n_days)
    return matrix


def audit(results, raw, output):
    frame = compute(results, raw)
    times = pd.read_csv(raw, usecols=['TransactionID', 'TransactionDT']).set_index('TransactionID').TransactionDT
    rng = np.random.default_rng(20261107)
    updates, contrasts, rewiring = [], [], []
    last = {}
    for cutoff, group in frame.groupby('cutoff'):
        matrices = {}
        with np.load(results / f'c{cutoff}_tab.npz') as base:
            ids, y = base['wave_ids'], base['y']
        day = times.loc[ids].to_numpy() // 86400
        day -= day.min()
        nd = int(day.max()) + 1
        positives, negatives = (np.bincount(day[y == k], minlength=nd) for k in (1, 0))
        for row in group.itertuples():
            with np.load(results / f'{row.model}.npz') as fit:
                np.testing.assert_array_equal(fit['wave_ids'], ids)
                matrix = concordant_pairs(fit['scores'], y, day, nd)
                np.testing.assert_allclose(matrix.sum() / (positives.sum() * negatives.sum()), row.auc, atol=1e-12)
                matrices.setdefault(row.arm, []).append(matrix)
                change = json.loads(fit['edges_changed'].item())
                if row.arm == 'separate_rewired':
                    assert set(change) == {'addr1', 'DeviceType', 'P_emaildomain', 'id_31'}
                    assert all(0 < value <= (times < times.quantile(cutoff / 200)).sum() for value in change.values())
                    rewiring.append(dict(model=row.model, **change))
                else:
                    assert change == {}
                ranks = top_three(fit['phi'])
                key = row.model.split('_', 1)[1]
                if key in last:
                    old = last[key]
                    common = (old[:, :, None] == ranks[:, None, :]).any(axis=2).sum(axis=1)
                    updates.append(dict(before=previous, after=cutoff, arm=key, n_panel=len(ranks),
                                        n_reference=len(fit['bg_ids']), top1_retained=(old[:, 0] == ranks[:, 0]).mean(),
                                        top3_retained=common.mean()))
                last[key] = ranks
        previous = cutoff
        # Average the metric for the three frozen fits; resample the same days for every arm.
        starts = rng.integers(nd, size=(12000, (nd+6)//7))
        days = ((starts[:, :, None] + np.arange(7)) % nd).reshape(len(starts), -1)[:, :nd]
        w = np.zeros((len(days), nd), dtype=int)
        np.add.at(w, (np.arange(len(days))[:, None], days), 1)
        den = (w @ positives) * (w @ negatives)
        assert den.min() > 0
        for other in ['same', 'tab', 'separate_rewired']:
            delta = np.mean(matrices['separate'],axis=0) - np.mean(matrices[other],axis=0)
            observed = delta.sum()/(positives.sum()*negatives.sum())
            draws = np.einsum('bi,ij,bj->b', w, delta, w, optimize=True)/den
            lo, hi = np.quantile(draws, [.025/9, 1-.025/9])
            contrasts.append(dict(cutoff=cutoff, comparator=other, days=nd, delta_auc=observed,
                                  low=lo, high=hi, draws=len(w)))
    updated = pd.DataFrame(updates).set_index(['before','after','arm']).sort_index()
    saved = pd.read_csv(results/'updates.csv').set_index(['before','after','arm']).sort_index()
    pd.testing.assert_frame_equal(updated, saved)
    output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'rolling_fits.csv',index=False)
    updated.to_csv(output/'rolling_updates.csv')
    pd.DataFrame(rewiring).to_csv(output/'rolling_edge_counts.csv',index=False)
    pd.DataFrame(contrasts).to_csv(output/'rolling_auc_contrasts.csv',index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('results', 'raw', 'output'):
        p.add_argument(name,type=Path)
    args = p.parse_args()
    audit(args.results, args.raw, args.output)
