"""Paired time-block intervals for frozen models on the later IEEE-CIS cohort."""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C, splits

ARMS = ('tabular', 'same_block', 'separated')


def score_groups(scores):
    # Descending tied-score groups reproduce sklearn's thresholding for both metrics.
    _, ascending = np.unique(scores, return_inverse=True)
    return ascending.max() - ascending


def weighted_metrics(y, groups, weights):
    pos = np.bincount(groups, weights=y * weights)
    neg = np.bincount(groups, weights=(1-y) * weights)
    tp = pos.cumsum()
    fp = neg.cumsum()
    if tp[-1] == 0 or fp[-1] == 0:
        raise ValueError('One outcome class absent in a block draw')
    auc = np.sum(neg * (tp - pos / 2)) / (tp[-1] * fp[-1])
    precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp)>0)
    return np.array([auc, np.dot(precision, pos) / tp[-1]])


def estimate(directory):
    frame = pd.read_csv(C.RAW_DIR/'train_transaction.csv', usecols=[C.TIME_COL,C.ID_COL,C.TARGET])
    frame = frame.sort_values(C.TIME_COL).reset_index(drop=True)
    later = frame.loc[~splits.temporal_split_mask(frame)].reset_index(drop=True)
    # Validate ID order against the actual outcomes. Stable input order is needed at ties.
    y = later[C.TARGET].to_numpy()
    day = later[C.TIME_COL].to_numpy() // 86400
    day -= day.min()
    days = int(day.max())+1
    test_ids = later[C.ID_COL].to_numpy()
    all_groups = []
    scores_expected = pd.read_csv(directory/'stage_results.csv').set_index('model')
    for name in ARMS:
        with np.load(directory/f'{name}.npz') as saved:
            scores = saved['scores']
            if 'test_ids' in saved.files:
                np.testing.assert_array_equal(saved['test_ids'],test_ids)
            else:
                from src.shap_engine import stratified_sample
                check = stratified_sample(y,1000,C.SHAP_EVAL_SEED)
                np.testing.assert_array_equal(saved['ids'],test_ids[check])
        groups = score_groups(scores)
        ones=np.ones(len(y))
        direct=np.array([roc_auc_score(y,scores),average_precision_score(y,scores)])
        np.testing.assert_allclose(direct,scores_expected.loc[name,['auc','ap']],atol=1e-12)
        np.testing.assert_allclose(weighted_metrics(y,groups,ones),direct,atol=1e-12)
        all_groups.append(groups)

    # Use identical calendar-day weights across models and keep payments together within day.
    rng = np.random.default_rng(20261107)
    draws = np.empty((1000,len(ARMS),2))
    for i in range(len(draws)):
        starts = rng.integers(days, size=(days+6)//7)
        sampled = ((starts[:,None]+np.arange(7))%days).ravel()[:days]
        weights = np.bincount(sampled,minlength=days)[day]
        for j, groups in enumerate(all_groups):
            draws[i,j] = weighted_metrics(y,groups,weights)
    columns = [arm + '_' + metric for arm in ARMS for metric in ('auc', 'ap')]
    table = pd.DataFrame(draws.reshape(len(draws), -1), columns=columns)
    table['days'] = days
    table.to_csv(directory/'stage_block_draws.csv', index=False)
    return intervals_from_draws(directory)


def intervals_from_draws(directory):
    scores_expected = pd.read_csv(directory/'stage_results.csv').set_index('model')
    draws = pd.read_csv(directory/'stage_block_draws.csv')
    if len(draws) != 1000 or draws.days.nunique() != 1:
        raise ValueError('Unexpected block bootstrap design')
    days = int(draws.days.iloc[0])
    rows=[]
    for arm, ref in [('separated','tabular'),('separated','same_block'),('same_block','tabular')]:
        for metric in ('auc','ap'):
            delta = draws[arm+'_'+metric] - draws[ref+'_'+metric]
            if not np.isfinite(delta).all():
                raise ValueError('Invalid paired block differences')
            lo,hi=np.quantile(delta,[.025,.975])
            point=scores_expected.loc[arm,metric]-scores_expected.loc[ref,metric]
            rows.append(dict(model=arm,reference=ref,metric=metric,difference=point,
                             lower=lo,upper=hi,days=days,block_days=7,draws=len(draws)))
    return pd.DataFrame(rows)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory',type=Path,default=ROOT/'results/strict_ieee/stages')
    ap.add_argument('--from-draws',action='store_true')
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    table=intervals_from_draws(args.directory) if args.from_draws else estimate(args.directory)
    table.to_csv(args.output or args.directory/'stage_intervals.csv',index=False)
    print(table.to_string(index=False))
