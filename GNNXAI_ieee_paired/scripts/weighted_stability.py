"""Exploratory importance-weighting sensitivity on the original 20 saved pairs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

def run(source, output):
    old = pd.read_csv(source / 'wasserstein_by_seed.csv').set_index('seed')
    if sorted(old.index) != list(range(201, 221)):
        raise ValueError('Expected the original twenty distinct paired seeds')
    rows = []
    for seed in old.index:
        with np.load(source / f'{seed}_random.npz') as a, np.load(source / f'{seed}_temporal.npz') as b:
            metrics = {}
            for arm in ('tab', 'hyb', 'cov'):
                x, y = (s[arm][:, :486] for s in (a, b))
                if x.shape != (5000, 486) or y.shape != x.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
                    raise ValueError(f'Invalid data for {seed}, {arm}')
                gap = np.abs(np.sort(x, axis=0) - np.sort(y, axis=0)).mean(axis=0)
                scale = (np.abs(x).mean(axis=0) + np.abs(y).mean(axis=0)) / 2
                relative = np.divide(gap, scale, out=np.zeros_like(gap), where=scale > 0)
                np.testing.assert_allclose([gap.mean(), relative.mean()], old.loc[seed,[f'raw_{arm}',f'normalized_{arm}']].to_numpy(dtype=float), rtol=1e-6, atol=1e-10)
                # Common weights avoid rewarding models for shifting importance away from tabular fields.
                if arm == 'tab':
                    weights = scale.astype(float) / scale.sum(dtype=float)
                metrics[arm] = np.average(relative, weights=weights)
            rows.append(dict(seed=seed, **metrics))
    table = pd.DataFrame(rows).set_index('seed')
    table.to_csv(output / 'weighted_stability_by_seed.csv')
    summaries = []
    for comparator in ('tab','cov'):
        d = table.hyb - table[comparator]
        half = t.ppf(.975, len(d)-1)*d.sem()
        summaries.append(dict(comparator=comparator,n_seeds=len(d),positive=int((d>0).sum()),mean=d.mean(),low=d.mean()-half,high=d.mean()+half))
    summary=pd.DataFrame(summaries)
    summary.to_csv(output / 'weighted_stability_summary.csv',index=False)
    print(summary.to_string(index=False))

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    args=p.parse_args()
    run(args.source,Path(__file__).resolve().parent)
