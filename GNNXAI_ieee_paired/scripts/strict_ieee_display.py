"""Measure changes to the first displayed field and absolute attribution mass."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('tab', 'hyb', 'cov')


def recover(directory):
    reference = pd.read_csv(directory/'context_by_seed.csv').set_index('seed')
    paired = pd.read_csv(directory/'paired.csv').set_index('seed')
    rows=[]
    temporal_ids=None
    for seed in range(201,221):
        row={'seed':seed}
        masses={a:np.zeros(2) for a in ARMS}
        for split in ('random','temporal'):
            with np.load(directory/f'{seed}_{split}.npz') as f:
                ids=f['ids']
                if ids.shape!=(5000,) or len(np.unique(ids))!=5000 or int(f['p'])!=486:
                    raise ValueError('Unexpected denominator')
                if split=='temporal':
                    if temporal_ids is None:
                        temporal_ids=ids
                    np.testing.assert_array_equal(ids,temporal_ids)
                for a in ARMS:
                    ab=np.abs(f[a])
                    if ab.shape!=(5000,486 if a=='tab' else 550) or not np.isfinite(ab).all():
                        raise ValueError('Invalid SHAP array')
                    shared=ab[:,:486]
                    masses[a]+=np.array([shared.sum(dtype=np.float64),ab[:,486:].sum(dtype=np.float64)])
                    if split=='temporal':
                        # A fixed input order supplies the tie-break a one-slot display needs.
                        winner=shared.argmax(axis=1)
                        row['ties_'+a]=int(((shared==shared.max(axis=1,keepdims=True)).sum(axis=1)>1).sum())
                        if a=='tab':
                            baseline=winner
                        else:
                            row['changed_'+a]=int((winner!=baseline).sum())
        for a,(shared,added) in masses.items():
            row['shared_mass_'+a]=shared/10000
            row['added_mass_'+a]=added/10000
            if a!='tab':
                np.testing.assert_allclose(added/(shared+added),reference.loc[seed,'share_'+a],rtol=1e-5)
        row['js_difference']=paired.loc[seed,'js_hyb_minus_cov']
        row['n_display']=len(temporal_ids)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(table):
    if table.seed.tolist()!=list(range(201,221)) or not table.n_display.eq(5000).all():
        raise ValueError('Expected 20 ordered pairs with 5,000 payments each')
    v=pd.DataFrame()
    for a in ('hyb','cov'):
        if not table['changed_'+a].between(0,table.n_display).all():
            raise ValueError('Invalid change count')
        v['change_pct_'+a]=100*table['changed_'+a]/table.n_display
        v['share_'+a]=100*table['added_mass_'+a]/(table['shared_mass_'+a]+table['added_mass_'+a])
    v['change_gap_pp']=v.change_pct_hyb-v.change_pct_cov
    for a in ARMS:
        v['shared_mass_'+a]=table['shared_mass_'+a]
    if not np.isfinite(v).all().all():
        raise ValueError('Nonfinite summary')
    mean=v.mean()
    half=t.ppf(.975,len(v)-1)*v.sem()
    return pd.DataFrame({'mean':mean,'lower':mean-half,'upper':mean+half})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'results/strict_ieee/primary')
    parser.add_argument('--summary-only',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    out=args.output or args.directory
    out.mkdir(exist_ok=True,parents=True)
    table=pd.read_csv(args.directory/'display_by_seed.csv') if args.summary_only else recover(args.directory)
    result=summarize(table)
    if not args.summary_only:
        table.to_csv(out/'display_by_seed.csv',index=False)
    result.to_csv(out/'display_summary.csv',index_label='metric')
    print(result.to_string())
    print('Maximum tied first fields per fit:',table[['ties_tab','ties_hyb','ties_cov']].max().to_dict())
