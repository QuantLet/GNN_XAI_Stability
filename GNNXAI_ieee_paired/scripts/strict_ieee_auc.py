"""Paired AUC inference for exploratory forward-time model comparisons."""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.strict_ieee_stage_intervals import score_groups, weighted_metrics
from src import config as C, splits


def cat_draws(primary, stages):
    frame = pd.read_csv(C.RAW_DIR/'train_transaction.csv', usecols=[C.TIME_COL, C.ID_COL, C.TARGET])
    frame = frame.sort_values(C.TIME_COL).reset_index(drop=True)
    holdout = frame.loc[~splits.temporal_split_mask(frame)]
    y = holdout[C.TARGET].to_numpy()
    if len(y) != 177162 or y.sum() != 6125:
        raise ValueError('Unexpected evaluation cohort')
    days = holdout[C.TIME_COL].to_numpy() // 86400
    days -= days.min()
    n_days = int(days.max()) + 1
    with np.load(primary/'catboost_temporal.npz') as saved:
        np.testing.assert_array_equal(saved['test_ids'], holdout[C.ID_COL])
        scores = [saved[arm+'_prob'] for arm in ('hyb','tab')]
    reference = pd.read_csv(primary/'catboost_temporal.csv').set_index('arm')
    for name, score in zip(('hyb','tab'), scores):
        np.testing.assert_allclose(roc_auc_score(y,score),reference.loc[name,'auc'],rtol=0,atol=1e-12)
    # Recalculate one stage arm to verify draw-by-draw alignment with the published bootstrap.
    with np.load(stages/'tabular.npz') as stage:
        scores.append(stage['scores'])
    groups = [score_groups(score) for score in scores]
    archived = pd.read_csv(stages/'stage_block_draws.csv')
    if len(archived) != 1000 or not archived.days.eq(n_days).all():
        raise ValueError('Unexpected bootstrap design')
    rng = np.random.default_rng(20261107)
    values = []
    for stage_auc in archived.tabular_auc:
        starts = rng.integers(n_days, size=(n_days+6)//7)
        sampled = ((starts[:, None] + np.arange(7)) % n_days).ravel()[:n_days]
        weights = np.bincount(sampled,minlength=n_days)[days]
        hybrid, tabular, check = [weighted_metrics(y,g,weights)[0] for g in groups]
        np.testing.assert_allclose(check,stage_auc,rtol=0,atol=1e-12)
        values.append(hybrid-tabular)
    return pd.DataFrame({'cat_graph_minus_tabular':values})


def summarize(primary, stages, cats):
    data = pd.read_csv(primary/'context_by_seed.csv').set_index('seed')
    if data.index.tolist() != list(range(201,221)) or data.auc_tab.nunique() != 1:
        raise ValueError('Unexpected refits or moving temporal baseline')
    rows=[]
    for model, ref in [('hyb','tab'),('hyb','cov'),('cov','tab')]:
        delta=data['auc_'+model]-data['auc_'+ref]
        mean=delta.mean()
        err=delta.sem()
        point=t.ppf(.975,19)*err
        multiple=t.ppf(1-.05/(2*3),19)*err
        rows.append(('seeds',model+'-'+ref,mean,mean-point,mean+point,mean-multiple,mean+multiple))
    stage=pd.read_csv(stages/'stage_results.csv').set_index('model')
    boot=pd.read_csv(stages/'stage_block_draws.csv')
    cat=pd.read_csv(primary/'catboost_temporal.csv').set_index('arm')
    means={'cat_graph_minus_tabular':cat.loc['hyb','auc']-cat.loc['tab','auc']}
    draws=cats.copy()
    if len(draws)!=len(boot):
        raise ValueError('Draw counts disagree')
    for m,r in [('separated','same_block'),('separated','tabular'),('same_block','tabular')]:
        key=m+'-'+r
        means[key]=stage.loc[m,'auc']-stage.loc[r,'auc']
        draws[key]=boot[m+'_auc']-boot[r+'_auc']
    points=pd.Series(means)
    se=draws.std(ddof=1)
    if not np.isfinite(draws).all().all() or not (se>0).all():
        raise ValueError('Invalid bootstrap sample')
    # One common critical value covers all four frozen-model contrasts on the same days.
    critical=((draws-points)/se).abs().max(axis=1).quantile(.95)
    for key,mean in points.items():
        lo,hi=draws[key].quantile([.025,.975])
        rows.append(('days',key,mean,lo,hi,mean-critical*se[key],mean+critical*se[key]))
    return pd.DataFrame(rows,columns=['unit','contrast','difference','point_lower','point_upper','family_lower','family_upper'])


def write_table(result, path):
    labels = ['Graph minus tabular', 'Graph minus Gaussian', 'Gaussian minus tabular',
              'CatBoost: graph minus tabular', 'Earlier minus same-period graph',
              'Earlier graph minus tabular', 'Same-period graph minus tabular']
    lines = [r'\begin{tabular}{lrr}', r'\toprule',
             r'AUC contrast & Difference & Simultaneous 95\% interval \\', r'\midrule']
    for idx, (row, label) in enumerate(zip(result.itertuples(), labels)):
        if idx in (0, 3):
            heading = 'Refits on the fixed cohort' if idx == 0 else 'Day blocks for fixed models'
            lines.append(r'\multicolumn{3}{l}{\emph{' + heading + r'}} \\')
        lines.append(f'{label} & \u0024{row.difference:+.5f}\u0024 & '
                     f'\u0024[{row.family_lower:+.5f}, {row.family_upper:+.5f}]\u0024' + r' \\')
    path.write_text('\n'.join(lines + [r'\bottomrule', r'\end{tabular}', '']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,help='Use the public aggregate CSV files only')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    primary=args.reference or ROOT/'results/strict_ieee/primary'
    stages=args.reference or ROOT/'results/strict_ieee/stages'
    out=args.output or stages
    out.mkdir(exist_ok=True,parents=True)
    if args.reference:
        cats=pd.read_csv(primary/'cat_auc_block_draws.csv')
    else:
        cats=cat_draws(primary,stages)
        cats.to_csv(out/'cat_auc_block_draws.csv',index=False)
    result=summarize(primary,stages,cats)
    result.to_csv(out/'paired_auc_inference.csv',index=False)
    write_table(result,out/'strict_ieee_auc.tex')
    print(result.to_string(index=False))
