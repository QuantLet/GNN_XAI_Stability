"""Matched reseeding experiment on the labelled Elliptic transactions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np
import pandas as pd


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def atomic_npz(path, **arrays):
    tmp=path.with_suffix('.partial.npz')
    np.savez_compressed(tmp,**arrays)
    tmp.replace(path)


def prepare(project, out):
    sys.path.insert(0,str(project))
    from src import elliptic, config as C, shap_engine
    import shap, torch, xgboost
    from sklearn.model_selection import train_test_split
    df,x,y=elliptic.prepare_elliptic()
    ids=df.txId.to_numpy()
    steps=df.time_step.to_numpy()
    assert len(set(ids))==len(ids) and np.isfinite(x).all() and x.shape==(len(y),165)
    temporal=elliptic.temporal_mask(df)
    train,_=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=201)
    random=np.zeros(len(y),bool)
    random[train]=True
    edges=elliptic.load_elliptic()[2]
    mapper=pd.Series(np.arange(len(y)),index=ids)
    ends=edges[['txId1','txId2']].apply(lambda col: col.map(mapper)).dropna().to_numpy(dtype=int).T
    src,dst=ends
    train_edges=ends[:,temporal[src]&temporal[dst]]
    crossing=int((temporal[src]&~temporal[dst]).sum())
    if crossing:
        raise ValueError('Eligible historical-to-future edges require a directed inference graph')
    scans=[*sorted((C.RAW_DIR/'elliptic').glob('*.csv')),project/'src/elliptic.py',project/'scripts/05_elliptic.py',
           project/'src/config.py',project/'src/models.py',project/'src/shap_engine.py',Path(__file__)]
    meta=dict(seeds=list(range(201,221)),primary='graph minus Gaussian leading-field change rate',
              unit='delete one of twenty paired seed fits, fixed chronological explanation panel',
              cut=int(steps[~temporal].min()),features=x.shape[1],labeled=len(y), illicit=int(y.sum()),
              labeled_edges=int(src.size),training_edges=int(train_edges.shape[1]),history_to_future=crossing,
              versions=dict(torch=torch.__version__,shap=shap.__version__,xgboost=xgboost.__version__,numpy=np.__version__),
              hashes={p.name:digest(p) for p in scans},splits={})
    arrays=dict(x=x,y=y,ids=ids,steps=steps,edge=train_edges)
    for tag,mask in [('temporal',temporal),('random',random)]:
        tr,te=np.flatnonzero(mask),np.flatnonzero(~mask)
        ev=te[shap_engine.stratified_sample(y[te],C.SHAP_EVAL_N,C.SHAP_EVAL_SEED)]
        candidates=tr[shap_engine.stratified_sample(y[tr],C.SHAP_BACKGROUND_N,C.SHAP_BACKGROUND_SEED)]
        bg=shap.utils.sample(candidates,100,random_state=0)
        np.testing.assert_array_equal(shap.maskers.Independent(x[candidates]).data,x[bg])
        assert len(set(ev))==5000 and len(set(bg))==100 and not np.intersect1d(bg,ev).size
        arrays.update({tag:mask,tag+'_bg':bg,tag+'_ev':ev})
        meta['splits'][tag]=dict(train=len(tr),test=len(te),test_illicit=int(y[te].sum()),explained=len(ev),explained_illicit=int(y[ev].sum()),background=len(bg))
    path=out/'design.json'
    content=json.dumps(meta,indent=2,sort_keys=True)+'\n'
    if path.exists() and path.read_text()!=content:
        raise ValueError('Inputs/design changed; use a fresh directory')
    npz=out/'inputs.npz'
    if not npz.exists():
        atomic_npz(npz,**arrays)
    else:
        with np.load(npz) as old:
            for name, val in arrays.items(): np.testing.assert_array_equal(old[name],val)
    path.write_text(content)
    print('Locked design:',meta['splits'], 'historical edges to test:',crossing,flush=True)


def work(job):
    project,out,key=job
    sys.path.insert(0,str(project))
    from src import config as C
    import shap, torch
    from torch_geometric.data import Data
    from xgboost import XGBClassifier
    torch.set_num_threads(1)
    with np.load(out/'inputs.npz') as d:
        tag=key if isinstance(key,str) else 'temporal'
        mask,bg,ev=d[tag],d[tag+'_bg'],d[tag+'_ev']
        x,y,ids=d['x'],d['y'],d['ids']
        edge=d['edge']
    def fit(xin,seed):
        m=XGBClassifier(**dict(C.XGB_PARAMS,device='cpu',n_jobs=1,random_state=seed))
        m.fit(xin[mask],y[mask])
        return m
    def explain(m,xin):
        ex=shap.TreeExplainer(m,data=xin[bg],feature_perturbation='interventional',model_output='raw')
        np.testing.assert_array_equal(ex.data,xin[bg])
        phi=np.asarray(ex.shap_values(xin[ev],check_additivity=False))
        assert phi.shape==(len(ev),xin.shape[1]) and np.isfinite(phi).all()
        # Numerical additivity check on a fixed small subset.
        np.testing.assert_allclose(phi[:16].sum(axis=1)+ex.expected_value,m.predict(xin[ev[:16]],output_margin=True),rtol=2e-4,atol=2e-4)
        return phi[:,:x.shape[1]]
    if isinstance(key,str):
        p=out/f'tab_{tag}.npz'
        if p.exists(): return
        a,b=(fit(x,s) for s in (42,43))
        attrs=lambda m:json.loads(bytes(m.get_booster().save_raw(raw_format='json')))['learner']
        same=attrs(a)==attrs(b)
        score42=a.predict_proba(x[~mask])[:,1]
        score43=b.predict_proba(x[~mask])[:,1]
        phi=explain(a,x)
        if same and np.array_equal(score42,score43):
            max_delta=0.
            changed=0
        else:
            other=explain(b,x)
            max_delta=float(np.abs(phi-other).max())
            changed=int((np.abs(phi).argmax(1)!=np.abs(other).argmax(1)).sum())
        atomic_npz(p,ids=ids[ev],bg_ids=ids[bg],phi=phi,same_trees=same,
                   same_predictions=np.array_equal(score42,score43),max_delta=max_delta,changed=changed)
        print(p.stem, 'identical trees', same,flush=True)
        return
    files={arm:out/f'{key}_{arm}.npz' for arm in ('hyb','cov')}
    if all(f.exists() for f in files.values()): return
    e5=importlib.import_module('scripts.05_elliptic')
    data=Data(x=torch.from_numpy(x),edge_index=torch.from_numpy(edge),y=torch.from_numpy(np.where(mask,y,-1)),train_mask=torch.from_numpy(mask))
    z,_=e5.train_extract(data,device='cpu',seed=key)
    factor=np.linalg.cholesky(np.cov(z[mask],rowvar=False)+1e-6*np.eye(z.shape[1]))
    noise=(np.random.default_rng(key).standard_normal(z.shape)@factor.T).astype(np.float32)
    for arm,extra in [('hyb',z),('cov',noise)]:
        if files[arm].exists(): continue
        inp=np.column_stack((x,extra))
        model=fit(inp,42)
        phi=explain(model,inp)
        atomic_npz(files[arm],ids=ids[ev],bg_ids=ids[bg],phi=phi,seed=key,
                   graph_hash=hashlib.sha256(z.tobytes()).hexdigest())
        print(files[arm].stem,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('project',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    project,out=a.project.resolve(),a.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(project,out)
        jobs=[(project,out,k) for k in ['temporal','random',*range(201,221)]]
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            list(pool.map(work,jobs))
        print('Completed',flush=True)
