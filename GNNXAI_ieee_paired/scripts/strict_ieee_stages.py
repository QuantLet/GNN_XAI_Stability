"""Chronological stage separation. All tree learners fit the same middle block."""
from pathlib import Path
import argparse
import gc
import fcntl
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.strict_ieee import C, data_prep, models, representation, shap_engine, splits
from scripts.instance_level_refitlevel import _topk_overlap


def evaluate(name, x, z, df, middle, test, directory):
    path = directory / f'{name}.npz'
    if path.exists():
        return
    y = df[C.TARGET].to_numpy()
    train_x = models.build_input(x[middle], None if z is None else z[middle])
    test_x = models.build_input(x[test], None if z is None else z[test])
    print('Fitting', name, flush=True)
    fit = XGBClassifier(**dict(C.XGB_PARAMS, device='cpu'))
    fit.fit(train_x, y[middle], verbose=False)
    scores = fit.predict_proba(test_x)[:, 1]
    print('Explaining', name, flush=True)
    values, idx = shap_engine.compute_shap(fit, train_x, y[middle], test_x, y[test], eval_n=1000)
    if not np.isfinite(scores).all() or not np.isfinite(values).all():
        raise ValueError('Nonfinite scores or SHAP')
    temp = path.with_suffix('.partial.npz')
    np.savez(temp, scores=scores, phi=values, ids=df[C.ID_COL].to_numpy()[test][idx])
    temp.replace(path)


def run(directory):
    df = data_prep.load_raw()
    outer = splits.temporal_split_mask(df)
    time = df[C.TIME_COL].to_numpy()
    early = time < np.median(time[outer])
    middle = outer & ~early
    test = ~outer
    if not (time[early].max() < time[middle].min() <= time[middle].max() < time[test].min()):
        raise ValueError('Overlapping chronological blocks')
    y = df[C.TARGET].to_numpy()
    meta = pd.DataFrame([dict(block=k, n=int(m.sum()), frauds=int(y[m].sum()),
                              start=int(time[m].min()), end=int(time[m].max()))
                         for k,m in [('early',early),('middle',middle),('test',test)]])
    print(meta.to_string(index=False), flush=True)
    # A single common set of middle-block tabular encodings enters every tree.
    # The early graph has its own frozen early-block encodings and entity degrees.
    x, z = representation(df, middle, C.SEED, dict(C.SAGE_PARAMS))
    evaluate('same_block', x, z, df, middle, test, directory)
    del z
    gc.collect()
    evaluate('tabular', x, None, df, middle, test, directory)
    if not (directory/'separated.npz').exists():
        graph_x, early_z = representation(df, early, C.SEED, dict(C.SAGE_PARAMS))
        del graph_x
        evaluate('separated', x, early_z, df, middle, test, directory)
    expected_ids = df[C.ID_COL].to_numpy()[test][shap_engine.stratified_sample(y[test],1000,C.SHAP_EVAL_SEED)]
    with np.load(directory/'tabular.npz') as tab:
        tab_phi = tab['phi']
    results=[]
    for name in ('tabular','same_block','separated'):
        with np.load(directory/f'{name}.npz') as fit:
            np.testing.assert_array_equal(fit['ids'],expected_ids)
            phi, scores = fit['phi'], fit['scores']
            if phi.shape != (1000, 486 if name=='tabular' else 550) or scores.shape!=(test.sum(),):
                raise ValueError(f'Invalid {name} denominator')
            results.append(dict(model=name, auc=roc_auc_score(y[test],scores),
                                ap=average_precision_score(y[test],scores),
                                retained_three=3*_topk_overlap(tab_phi,phi[:,:486],3)))
    meta.to_csv(directory/'stage_blocks.csv',index=False)
    result=pd.DataFrame(results)
    result.to_csv(directory/'stage_results.csv',index=False)
    print(result.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'results/strict_ieee/stages')
    args=parser.parse_args()
    args.directory.mkdir(exist_ok=True,parents=True)
    with (args.directory/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.directory)
