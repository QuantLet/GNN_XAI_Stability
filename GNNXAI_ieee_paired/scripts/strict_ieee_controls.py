"""Secondary checks for the strict IEEE-CIS experiment, canonical seed 42.

Select CatBoost iterations on a chronological holdout *inside* the outer training
period, including a fresh inner graph. Audit raw and standardized Wasserstein
contrasts at the previously published surrogate width; report attained scale.
"""
import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.strict_ieee import C, ROOT, control, data_prep, models, representation, shap_engine, splits
from src.metrics import expected_calibration_error
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def catboost(df, directory, pilot):
    path = directory/'catboost_temporal.csv'
    if path.exists():
        return
    mask = splits.temporal_split_mask(df)
    train = df.loc[mask].reset_index(drop=True)
    inner = splits.temporal_split_mask(train, cutoff_pct=.85)
    sage = dict(C.SAGE_PARAMS)
    params = dict(C.CATB_PARAMS, thread_count=4, allow_writing_files=False, task_type='CPU')
    params.pop('devices', None)
    if pilot:
        params.update(iterations=32, depth=4, early_stopping_rounds=5)
        sage.update(max_epochs=3)
    x, z = representation(train, inner, C.SEED, sage)
    iterations = {}
    for name, extra in [('tab', None), ('hyb', z)]:
        inputs = models.build_input(x, extra)
        model = models.fit_catb(inputs[inner], train[C.TARGET][inner], inputs[~inner], train[C.TARGET][~inner], params)
        iterations[name] = model.get_best_iteration()+1
        if iterations[name] < 1:
            raise ValueError('Inner validation selected no trees')
    del inputs, x, z, train, model
    x, z = representation(df, mask, C.SEED, sage)
    y = df[C.TARGET].to_numpy()
    params.pop('early_stopping_rounds')
    rows = []
    attribution = {}
    for name, extra in [('tab', None), ('hyb', z)]:
        inputs = models.build_input(x, extra)
        model = CatBoostClassifier(**dict(params, iterations=iterations[name], use_best_model=False))
        model.fit(inputs[mask], y[mask], verbose=False)
        probs = model.predict_proba(inputs[~mask])[:,1]
        sv, idx = shap_engine.compute_shap(model, inputs[mask], y[mask], inputs[~mask], y[~mask],
                                            background_n=100 if pilot else C.SHAP_BACKGROUND_N,
                                            eval_n=250 if pilot else C.SHAP_EVAL_N)
        attribution[name] = sv
        attribution[name+'_ids'] = df.loc[~mask,C.ID_COL].to_numpy()[idx]
        attribution[name+'_prob'] = probs
        rows.append(dict(arm=name, seed=C.SEED, inner_train_n=inner.sum(), inner_valid_n=(~inner).sum(),
                         train_n=mask.sum(), test_n=(~mask).sum(), test_frauds=y[~mask].sum(),
                         iterations=iterations[name], auc=roc_auc_score(y[~mask],probs),
                         average_precision=average_precision_score(y[~mask],probs),
                         brier=brier_score_loss(y[~mask],probs), ece=expected_calibration_error(y[~mask],probs)))
    attribution['test_ids'] = df.loc[~mask,C.ID_COL].to_numpy()
    np.savez(directory/'catboost_temporal.npz', **attribution)
    tmp = path.with_suffix('.partial.csv')
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def scale(df, directory, pilot):
    # Width fixed from the historical experiment, so its attained scale must be
    # measured anew. Generate independent covariance blocks using training rows only.
    for split in ('temporal', 'random'):
        path = directory/f'scale_{split}.npz'
        if path.exists():
            continue
        mask = splits.temporal_split_mask(df) if split=='temporal' else splits.random_split_mask(df,seed=C.SEED)
        sage = dict(C.SAGE_PARAMS)
        tree = dict(C.XGB_PARAMS,device='cpu')
        if pilot:
            sage.update(max_epochs=3)
            tree.update(n_estimators=24,max_depth=4)
        x,z = representation(df,mask,C.SEED,sage)
        y = df[C.TARGET].to_numpy()
        width = 703
        wide = np.empty((len(z),width),dtype=np.float32)
        for start in range(0,width,z.shape[1]):
            size = min(z.shape[1],width-start)
            wide[:,start:start+size] = control(z,mask,C.SEED+start//z.shape[1])[:,:size]
        saved = dict(p=x.shape[1])
        for name,extra in [('tab',None),('hyb',z),('wide',wide)]:
            inputs = models.build_input(x,extra)
            model = models.fit_xgb(inputs[mask],y[mask],inputs[~mask],y[~mask],tree)
            saved[name], idx = shap_engine.compute_shap(model,inputs[mask],y[mask],inputs[~mask],y[~mask],
                                                        background_n=100 if pilot else C.SHAP_BACKGROUND_N,
                                                        eval_n=250 if pilot else C.SHAP_EVAL_N)
            saved[name+'_ids'] = df.loc[~mask,C.ID_COL].to_numpy()[idx]
        tmp = path.with_suffix('.partial.npz')
        np.savez(tmp,**saved)
        tmp.replace(path)
    with np.load(directory/'scale_random.npz') as r, np.load(directory/'scale_temporal.npz') as t:
        p = int(t['p'])
        baseline = np.abs(np.concatenate([r['tab'],t['tab']])).mean(0)
        rows = []
        for name in ('tab','hyb','wide'):
            ar,at = r[name],t[name]
            mass = np.abs(np.concatenate([ar,at])).mean(0)
            live = (baseline>0) & (mass[:p]>0)
            w = np.array([wasserstein_distance(ar[:,k],at[:,k]) for k in range(p)])
            # A column with zero pooled scale has identically zero distance.
            norm = np.divide(w,mass[:p],out=np.zeros_like(w),where=mass[:p]>0)
            rows.append(dict(arm=name,width=ar.shape[1]-p,p=p,active_for_ratio=live.sum(),
                             scale_ratio=np.median(mass[:p][live]/baseline[live]),
                             share=mass[p:].sum()/mass.sum(),wasserstein=w.mean(),normalized=norm.mean()))
        pd.DataFrame(rows).to_csv(directory/'scale.csv',index=False)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot',action='store_true')
    opt=ap.parse_args()
    out=ROOT/'results'/'strict_ieee'/('pilot' if opt.pilot else 'primary')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.controls.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        data=data_prep.load_raw()
        if opt.pilot:
            data=data.sample(n=6000,random_state=42).sort_values(C.TIME_COL).reset_index(drop=True)
        catboost(data,out,opt.pilot)
        scale(data,out,opt.pilot)
