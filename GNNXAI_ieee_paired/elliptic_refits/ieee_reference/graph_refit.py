"""Local explanation agreement across graph initializations on the fixed R8.6 forward fold."""
from itertools import combinations
from pathlib import Path
import argparse
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import t


def rankings(matrix):
    if matrix.shape != (5000,486) or not np.isfinite(matrix).all():
        raise ValueError('Invalid shared attribution matrix')
    return np.argsort(-np.abs(matrix),axis=1,kind='stable')[:, :10]


def agreement(a,b):
    changed = np.count_nonzero(a[:,0]!=b[:,0])
    counts={}
    for k in (3,10):
        overlap=(a[:,:k,None]==b[:,None,:k]).any(axis=2).sum()
        counts[f'top{k}_overlap']=overlap/len(a)
    return dict(leading_changed=int(changed),fraction=changed/len(a),**counts)


def run(root, out):
    seeds=list(range(201,221))
    ranks, gaussian, refs, graphs={}, {}, {}, set()
    display=pd.read_csv(root/'display_by_seed.csv').set_index('seed')
    for seed in seeds:
        with np.load(root/f'{seed}_temporal.npz') as saved:
            ids=saved['ids']; tab=saved['tab']
            if seed==201:
                ref_ids=ids.copy()
                assert len(set(ids))==5000
                fingerprint=hashlib.sha256(tab.tobytes()).digest()
                base=rankings(tab)
            np.testing.assert_array_equal(ids,ref_ids)
            assert hashlib.sha256(tab.tobytes()).digest()==fingerprint
            graphs.add(str(saved['graph_hash']))
            ranks[seed]=rankings(saved['hyb'][:,:486])
            gaussian[seed]=rankings(saved['cov'][:,:486])
            compared=agreement(ranks[seed],base)
            row=display.loc[seed]
            assert (row.n_display,row.changed_hyb,row.ties_tab,row.ties_hyb)==(len(ids),compared['leading_changed'],0,0)
            refs[seed]=compared['fraction']
    assert len(graphs)==len(seeds)
    frame=pd.DataFrame([dict(a=a,b=b,**agreement(ranks[a],ranks[b]),
                             gaussian_fraction=np.mean(gaussian[a][:,0]!=gaussian[b][:,0]),
                             against_tabular=(refs[a]+refs[b])/2)
                        for a,b in combinations(seeds,2)])
    out.mkdir(exist_ok=True,parents=True)
    frame['excess_over_tabular_comparison']=frame.fraction-frame.against_tabular
    frame['graph_minus_gaussian_reseed']=frame.fraction-frame.gaussian_fraction
    frame.to_csv(out/'graph_pairs.csv',index=False)
    summary=[]
    for metric in ('fraction','top3_overlap','top10_overlap','gaussian_fraction','graph_minus_gaussian_reseed','against_tabular','excess_over_tabular_comparison'):
        value=frame[metric].mean()
        # Delete a graph initialization, hence all 19 associated contrasts. Payments remain fixed.
        loo=np.array([frame.loc[(frame.a!=s)&(frame.b!=s),metric].mean() for s in seeds])
        se=np.sqrt((len(seeds)-1)*np.mean((loo-loo.mean())**2))
        h=t.ppf(.975,len(seeds)-1)*se
        disjoint=frame[(frame.a%2==1)&(frame.b==frame.a+1)][metric]
        assert len(disjoint)==10
        summary.append(dict(metric=metric,estimate=value,lower=value-h,upper=value+h,disjoint_mean=disjoint.mean()))
    summary=pd.DataFrame(summary)
    summary.to_csv(out/'graph_summary.csv',index=False)
    print(summary.to_string(index=False))
    print('Per seed baseline range',min(refs.values()),max(refs.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('primary',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    run(a.primary,a.output)
