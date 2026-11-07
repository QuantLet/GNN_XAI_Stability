"""Strict forward-time IEEE-CIS rerun. Run with --pilot for a reduced smoke test.

Each completed split is an atomic checkpoint. The old entry points reproduce the
archived specification. This entry point always rebuilds representations from raw data.
"""
import argparse
import fcntl
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C, data_prep, embeddings, gnn, graph, models, shap_engine, splits, stability
from scripts.instance_level_refitlevel import _spearman, _topk_overlap
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score


def representation(df, mask, seed, sage_params):
    table, _ = data_prep.build_features(df, mask, historical_frequencies=False)
    x = table.to_numpy(dtype=np.float32)
    del table
    x = graph.standardize(x[mask], x)
    full = graph.build_hetero_graph(df, x, mask, inductive=True)
    # Physically remove all held-out nodes during optimization. At inference
    # they receive historical messages but cannot update any entity.
    train = full.subgraph({'transaction': torch.from_numpy(np.flatnonzero(mask))})
    fitted = gnn.train_graphsage(train, params=sage_params, seed=seed)
    z, _ = embeddings.extract_embeddings(fitted, full)
    return x, z


def control(z, mask, seed):
    cov = np.cov(z[mask], rowvar=False)
    factor = np.linalg.cholesky(cov + 1e-6 * np.eye(z.shape[1]))
    return (np.random.default_rng(seed).standard_normal(z.shape) @ factor.T).astype(np.float32)


def run_split(df, seed, split, directory, pilot):
    path = directory / f'{seed}_{split}.npz'
    if path.exists():
        return
    print(f'Starting {seed} {split} ({len(df)} transactions)', flush=True)
    mask = splits.random_split_mask(df, seed=seed) if split == 'random' else splits.temporal_split_mask(df)
    params = dict(C.SAGE_PARAMS)
    tree = dict(C.XGB_PARAMS, device='cpu')
    if pilot:
        params.update(max_epochs=3)
        tree.update(n_estimators=24, max_depth=4)
    x, z = representation(df, mask, seed, params)
    y = df[C.TARGET].to_numpy()
    cv = control(z, mask, seed)
    saved = dict(train_n=mask.sum(), test_n=(~mask).sum(), p=x.shape[1],
                 graph_hash=hashlib.sha256(z.tobytes()).hexdigest())
    for name, extra in [('tab', None), ('hyb', z), ('cov', cv)]:
        checkpoint = directory / f'{seed}_{split}_{name}.npz'
        if checkpoint.exists():
            with np.load(checkpoint) as existing:
                if (str(existing['graph_hash']) != saved['graph_hash'] or
                    ('ids' in saved and not np.array_equal(saved['ids'], existing['ids']))):
                    raise ValueError(f'Incompatible arm checkpoint: {checkpoint}')
                saved.update({key: existing[key] for key in ('ids', name, f'auc_{name}', f'ap_{name}')})
            continue
        print(f'Fitting {seed} {split} {name}', flush=True)
        inputs = models.build_input(x, extra)
        fit = models.fit_xgb(inputs[mask], y[mask], inputs[~mask], y[~mask], tree)
        probs = fit.predict_proba(inputs[~mask])[:, 1]
        print(f'Explaining {seed} {split} {name}', flush=True)
        phi, idx = shap_engine.compute_shap(fit, inputs[mask], y[mask], inputs[~mask], y[~mask],
                                            background_n=100 if pilot else C.SHAP_BACKGROUND_N,
                                            eval_n=250 if pilot else C.SHAP_EVAL_N)
        ids = df.loc[~mask, C.ID_COL].to_numpy()[idx]
        if 'ids' in saved and not np.array_equal(saved['ids'], ids):
            raise ValueError('Unpaired SHAP evaluation rows')
        if not np.isfinite(phi).all():
            raise ValueError('Non-finite SHAP')
        saved['ids'] = ids
        saved[name] = phi
        saved[f'auc_{name}'] = roc_auc_score(y[~mask], probs)
        saved[f'ap_{name}'] = average_precision_score(y[~mask], probs)
        partial = checkpoint.with_suffix('.partial.npz')
        np.savez(partial, **{key: saved[key] for key in ('graph_hash','ids',name,f'auc_{name}',f'ap_{name}')})
        partial.replace(checkpoint)
        del inputs, fit, probs
        print(f'Saved arm {name}', flush=True)
    tmp = path.with_suffix('.partial.npz')
    np.savez(tmp, **saved)
    tmp.replace(path)
    for name in ('tab','hyb','cov'):
        (directory / f'{seed}_{split}_{name}.npz').unlink(missing_ok=True)
    print(f'Saved {path}', flush=True)


def summarize(directory, seeds):
    rows = []
    graphs, baselines = set(), set()
    for seed in seeds:
        paths = [directory / f'{seed}_{split}.npz' for split in ('random', 'temporal')]
        if not all(path.exists() for path in paths):
            continue
        sides = []
        for path in paths:
            with np.load(path) as data:
                sides.append(dict(data))
        r, t = sides
        graphs.add(str(t['graph_hash']))
        baselines.add(hashlib.sha256(t['tab'].tobytes()).hexdigest())
        p = int(r['p'])
        row = dict(seed=seed, p=p, n_shap_random=len(r['ids']), n_shap_temporal=len(t['ids']))
        for split, side in zip(('random', 'temporal'), sides):
            tab, hyb, cov = [side[a][:, :p] for a in ('tab', 'hyb', 'cov')]
            row['top10_'+split] = _topk_overlap(tab, hyb) - _topk_overlap(tab, cov)
            row['spearman_'+split] = _spearman(tab, hyb) - _spearman(tab, cov)
        for a in ('tab', 'cov'):
            ar, at, br, bt = r[a][:,:p], t[a][:,:p], r['hyb'][:,:p], t['hyb'][:,:p]
            deltas = []
            for k in range(p):
                pool = np.concatenate([v[:, k] for v in (ar, at, br, bt)])
                edges = stability.bin_edges(pool, C.N_BINS, C.BIN_TYPE)
                deltas.append(stability.js_divergence(br[:,k], bt[:,k], edges) -
                              stability.js_divergence(ar[:,k], at[:,k], edges))
            row['js_hyb_minus_'+a] = np.mean(deltas)
        for name in ('hyb', 'cov'):
            aug = np.abs(np.concatenate([r[name], t[name]]))
            tab_scale = np.abs(np.concatenate([r['tab'], t['tab']])).mean(axis=0)
            shared = aug[:,:p].mean(axis=0)
            live = (shared > 0) & (tab_scale > 0)
            row['scale_'+name] = np.median(shared[live]/tab_scale[live])
            row['absorbed_'+name] = aug[:,p:].sum()/aug.sum()
            row['wasserstein_'+name+'_minus_tab'] = np.mean([
                stats.wasserstein_distance(r[name][:,k],t[name][:,k]) -
                stats.wasserstein_distance(r['tab'][:,k],t['tab'][:,k]) for k in range(p)])
        rows.append(row)
    if len(graphs) != len(rows) or len(baselines) > 1:
        raise ValueError('Temporal graph or baseline seed invariance failed')
    d = pd.DataFrame(rows)
    d.to_csv(directory/'paired.csv', index=False)
    if len(rows) == len(seeds):
        out = []
        for col in d.columns[4:]:
            v = d[col]
            h = stats.t.ppf(.975,len(v)-1)*v.sem()
            out.append(dict(metric=col,n=len(v),mean=v.mean(),lower=v.mean()-h,upper=v.mean()+h))
        pd.DataFrame(out).to_csv(directory/'intervals.csv',index=False)


def main(pilot):
    directory = ROOT/'results'/'strict_ieee'/('pilot' if pilot else 'primary')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        df = data_prep.load_raw()
        if pilot:
            df = df.sample(n=6000, random_state=42).sort_values(C.TIME_COL).reset_index(drop=True)
        seeds = list(range(201, 204 if pilot else 221))
        for seed in seeds:
            for split in ('random','temporal'):
                run_split(df, seed, split, directory, pilot)
            summarize(directory, seeds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    main(parser.parse_args().pilot)
