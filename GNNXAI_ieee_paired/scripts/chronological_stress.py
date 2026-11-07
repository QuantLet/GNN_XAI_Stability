"""Exploratory temporal stage and edge controls, compared on fixed later payments.

Run in a fresh output directory when the design or upstream training code changes.
Only the paired observed and re-wired historical training graphs differ in the
edge control. All held-out edges retain the observed links at inference.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

# The macOS wheels bring different OpenMP runtimes. Keep each fit single-threaded.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import torch
import shap
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('project', type=Path)
p.add_argument('output', type=Path)
args = p.parse_args()
sys.path.insert(0, str(args.project.resolve()))
from scripts.strict_ieee import C, data_prep, embeddings, gnn, graph, models

SEEDS = (201, 202, 203)
CUTOFFS = (60, 70, 80)
TYPES = ('same', 'separate', 'separate_rewired')
OUT = args.output


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(df, mask):
    table, _ = data_prep.build_features(df, mask, historical_frequencies=False)
    x = table.to_numpy(dtype=np.float32)
    del table
    if x.shape != (len(df), 486):
        raise ValueError('Unexpected transaction schema')
    return graph.standardize(x[mask], x)


def graph_fit(df, x, mask, seed, rewire):
    full = graph.build_hetero_graph(df, x, mask, inductive=True)
    train_ids = np.flatnonzero(mask)
    changed = {}
    if rewire:
        # The degrees at both endpoints, the sentinel positions and the time
        # profile of each relation are preserved. No held-out edge is moved.
        rng = np.random.default_rng(10000 + seed)
        days = df[C.TIME_COL].to_numpy() // 86400
        for col in C.ENTITY_COLS:
            forward = full['transaction', 'to', col].edge_index
            reverse = full[col, 'rev_to', 'transaction'].edge_index
            ids = reverse[0].numpy().copy()
            original = ids.copy()
            sentinel = len(full[col].x) - 1
            np.testing.assert_array_equal(forward[0].numpy(), train_ids)
            for day in np.unique(days[mask]):
                selected = np.flatnonzero(mask & (days == day) & (original != sentinel))
                ids[selected] = rng.permutation(original[selected])
            np.testing.assert_array_equal(np.bincount(ids[mask]), np.bincount(original[mask]))
            np.testing.assert_array_equal(ids[~mask], original[~mask])
            np.testing.assert_array_equal(ids == sentinel, original == sentinel)
            reverse[0] = torch.from_numpy(ids)
            forward[1] = torch.from_numpy(ids[mask])
            changed[col] = int((ids != original).sum())
        if not any(changed.values()):
            raise ValueError('Rewiring did not change any training edges')
    sub = full.subgraph({'transaction':torch.from_numpy(train_ids)})
    fitted = gnn.train_graphsage(sub, seed=seed)
    z, _ = embeddings.extract_embeddings(fitted, full)
    return z, changed


def top(x, k):
    return np.argsort(-np.abs(x), axis=1, kind='stable')[:, :k]


def retained(reference, values, k):
    a, b = top(reference, k), top(values, k)
    return float((a[:, :, None] == b[:, None, :]).any(axis=2).sum(axis=1).mean())


def checkpoint(name, x, z, y, middle, wave, reference, panel, ids, changed):
    path = OUT / (name + '.npz')
    if path.exists():
        return
    train = models.build_input(x[middle], None if z is None else z[middle])
    print('Fitting tree', name, flush=True)
    fit = XGBClassifier(**dict(C.XGB_PARAMS, device='cpu', n_jobs=1))
    fit.fit(train, y[middle], verbose=False)
    del train
    def inputs(rows):
        return models.build_input(x[rows], None if z is None else z[rows])
    scores = fit.predict_proba(inputs(wave))[:, 1]
    # Supply exactly 100 rows to avoid an implicit SHAP background subsample.
    bg = inputs(reference)
    ex = shap.TreeExplainer(fit, data=bg, feature_perturbation='interventional', model_output='raw')
    np.testing.assert_array_equal(ex.data, bg)
    phi = np.asarray(ex.shap_values(inputs(panel), check_additivity=False))
    if phi.shape != (len(panel), bg.shape[1]) or not np.isfinite(phi).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid explanations or scores')
    temp = path.with_suffix('.partial.npz')
    np.savez_compressed(temp, phi=phi, panel_ids=ids[panel], bg_ids=ids[reference],
                        wave_ids=ids[wave], scores=scores, y=y[wave],
                        edges_changed=json.dumps(changed, sort_keys=True))
    temp.replace(path)


def update_summary():
    rows = []
    for cutoff in CUTOFFS:
        base = OUT / f'c{cutoff}_tab.npz'
        if not base.exists():
            continue
        with np.load(base) as t:
            for path in sorted(OUT.glob(f'c{cutoff}_*.npz')):
                if path.name.endswith('.partial.npz'):
                    continue
                with np.load(path) as a:
                    for field in ('panel_ids','bg_ids','wave_ids','y'):
                        np.testing.assert_array_equal(t[field],a[field])
                    rows.append(dict(model=path.stem, n_wave=len(a['y']), n_fraud=int(a['y'].sum()),
                        auc=roc_auc_score(a['y'],a['scores']), ap=average_precision_score(a['y'],a['scores']),
                        n_panel=len(a['panel_ids']), top3_vs_tab=retained(t['phi'],a['phi'][:,:486],3)))
    tmp = OUT/'descriptive.partial.csv'
    pd.DataFrame(rows).to_csv(tmp,index=False)
    tmp.replace(OUT/'descriptive.csv')
    updates = []
    for before, after in zip(CUTOFFS, CUTOFFS[1:]):
        for earlier in sorted(OUT.glob(f'c{before}_*.npz')):
            if earlier.name.endswith('.partial.npz'):
                continue
            suffix = earlier.name.split('_',1)[1]
            later = OUT / f'c{after}_{suffix}'
            if not later.exists():
                continue
            with np.load(earlier) as a, np.load(later) as b:
                for field in ('bg_ids','panel_ids'):
                    np.testing.assert_array_equal(a[field],b[field])
                updates.append(dict(before=before,after=after,arm=suffix[:-4],
                    n_panel=len(a['panel_ids']),n_reference=len(a['bg_ids']),
                    top1_retained=retained(a['phi'][:,:486],b['phi'][:,:486],1),
                    top3_retained=retained(a['phi'][:,:486],b['phi'][:,:486],3)))
    if updates:
        tmp = OUT/'updates.partial.csv'
        pd.DataFrame(updates).to_csv(tmp,index=False)
        tmp.replace(OUT/'updates.csv')


def run():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    df = data_prep.load_raw()
    time = df[C.TIME_COL].to_numpy()
    ids = df[C.ID_COL].to_numpy()
    y = df[C.TARGET].to_numpy()
    if not df[C.ID_COL].is_unique or np.any(np.diff(time)<0):
        raise ValueError('Nonunique IDs or unsorted times')
    q = lambda pct: np.quantile(time, pct/100)
    rng = np.random.default_rng(314159)
    reference = np.sort(rng.choice(np.flatnonzero(time<q(20)),100,replace=False))
    panel = np.sort(rng.choice(np.flatnonzero(time>=q(80)),1000,replace=False))
    paths=[Path(__file__),*(args.project/'src'/f for f in ('data_prep.py','graph.py','gnn.py','embeddings.py','models.py','config.py','feature_columns_486.json'))]
    paths += [C.RAW_DIR / file for file in ('train_identity.csv','train_transaction.csv')]
    manifest = dict(seeds=SEEDS,cutoffs=CUTOFFS,bg_ids=ids[reference].tolist(),panel_ids=ids[panel].tolist(),
                    design='exploratory',versions=dict(numpy=np.__version__,torch=torch.__version__,shap=shap.__version__),
                    hashes={path.name:digest(path) for path in paths})
    blob=json.dumps(manifest,sort_keys=True)
    manifest_path=OUT/'design.json'
    if manifest_path.exists() and manifest_path.read_text()!=blob:
        raise ValueError('Design or inputs changed since previous run')
    manifest_path.write_text(blob)
    for cut in CUTOFFS:
        early = time < q(cut/2)
        middle = (time >= q(cut/2)) & (time < q(cut))
        wave = (time >= q(cut)) & (time < q(cut+10))
        assert time[early].max()<time[middle].min() and time[middle].max()<time[wave].min()
        if middle[panel].any() or not early[reference].all():
            raise ValueError('Reference/panel chronology violated')
        x = encode(df,middle)
        checkpoint(f'c{cut}_tab',x,None,y,middle,wave,reference,panel,ids,{})
        update_summary()
        early_x = encode(df,early)
        for seed in SEEDS:
            for arm in TYPES:
                name=f'c{cut}_{seed}_{arm}'
                if (OUT/(name+'.npz')).exists():
                    continue
                print(f'Fitting {name}',flush=True)
                past = arm != 'same'
                z, changed = graph_fit(df, early_x if past else x,early if past else middle,seed,arm=='separate_rewired')
                checkpoint(name,x,z,y,middle,wave,reference,panel,ids,changed)
                del z
                gc.collect()
                update_summary()
        del early_x,x
        gc.collect()

if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
