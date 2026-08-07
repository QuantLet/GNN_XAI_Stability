"""Leave-one-out PAIRWISE re-seeded null bands per (dataset, learner, binning) cell.

The single mandated null construction: reseed ONLY the tabular baseline learner's
random_state on a FIXED split / feature matrix / eval subsample, then read the
diagnostic on (seed_i tab, seed_j tab) pairs (i<j), which differ by training-RNG
noise alone -> every flagged feature is a Type I error. The 95% band of the
per-pair flagged fraction is the null band. Seed 42 (the project reference) is
EXCLUDED from the pair pool, matching scripts/null_analysis.run_t5.

This module computes the band for an arbitrary {seed: (sv_random, sv_temporal)}
cache under a chosen bin_type/n_bins, so the same code serves every cell:
  - IEEE-CIS XGBoost   (caches: stability/null_refits/nullshap_{split}_seed*.npy)
  - IEEE-CIS CatBoost  (caches: stability/null_refits/nullshap_catb_{split}_seed*.npy)
  - Elliptic XGBoost   (caches: stability/null_refits_elliptic/nullshap_xgb_{split}_seed*.npy)
  - Elliptic CatBoost  (caches: stability/null_refits_elliptic/nullshap_catb_{split}_seed*.npy)
each under quantile (primary) and equalwidth (robustness) binning.

Perm diagnostic is faithful to scripts/null_analysis.perm_diagnostic (same _js_fast,
same daily/time-step block permutation, same BH) but takes bin_type explicitly.
"""
from __future__ import annotations

import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src import stability as S
from scripts.null_analysis import _js_fast

ALPHA = C.ALPHA
N_PERM = 1000
N_PAIRS = 66          # matches the firmed XGBoost band (dcc074e)
SUBSET_SEED = 2026


def perm_diagnostic(a_r, a_t, b_r, b_t, blk_r, blk_t, bin_type, n_bins, seed=0):
    """Per-feature (delta_k, raw block-permutation p) under an explicit bin_type."""
    p = a_r.shape[1]
    rng = np.random.default_rng(seed)
    _, br_idx = np.unique(blk_r, return_inverse=True)
    _, bt_idx = np.unique(blk_t, return_inverse=True)
    nBr, nBt = int(br_idx.max()) + 1, int(bt_idx.max()) + 1
    deltas = np.empty(p)
    praw = np.empty(p)
    for k in range(p):
        ar, at, br, bt = a_r[:, k], a_t[:, k], b_r[:, k], b_t[:, k]
        edges = S.bin_edges(np.concatenate([ar, at, br, bt]), n_bins, bin_type)
        delta = _js_fast(br, bt, edges) - _js_fast(ar, at, edges)
        ge = 0
        for _ in range(N_PERM):
            mr = (rng.random(nBr) < 0.5)[br_idx]
            mt = (rng.random(nBt) < 0.5)[bt_idx]
            ar2 = np.where(mr, br, ar); br2 = np.where(mr, ar, br)
            at2 = np.where(mt, bt, at); bt2 = np.where(mt, at, bt)
            d = _js_fast(br2, bt2, edges) - _js_fast(ar2, at2, edges)
            if d <= delta:
                ge += 1
        deltas[k] = delta
        praw[k] = (1 + ge) / (N_PERM + 1)
    return deltas, praw


def flagged_fraction(deltas, praw):
    p_bh = S.bh_adjust(praw)
    return float(((p_bh < ALPHA) & (deltas < 0)).mean())


def _load_ckpt(ckpt_path):
    """Return {(i,j): (frac, md)} of pairs already computed (resumable run)."""
    done = {}
    if ckpt_path and ckpt_path.exists():
        for _, r in pd.read_csv(ckpt_path).iterrows():
            done[(int(r["i"]), int(r["j"]))] = (float(r["frac"]), float(r["md"]))
    return done


def _append_ckpt(ckpt_path, i, j, frac, md):
    """Append one pair's result and flush -- a crash loses at most the in-flight pair."""
    if not ckpt_path:
        return
    header = not ckpt_path.exists()
    with open(ckpt_path, "a", encoding="utf-8") as fh:
        if header:
            fh.write("i,j,frac,md\n")
        fh.write(f"{i},{j},{frac:.10f},{md:.10f}\n")
        fh.flush()


def pairwise_band(shap_by_seed, blk_r, blk_t, bin_type, n_bins=C.N_BINS,
                  n_pairs=N_PAIRS, subset_seed=SUBSET_SEED, ref_seed=C.SEED,
                  ckpt_path=None):
    """Return (fracs, mean_deltas, band, md_band) for the pairwise null.

    Per-pair checkpointing (ckpt_path): each pair is independently seeded
    (seed=1000*i+j), so skipping already-computed pairs and recomputing the rest
    yields bit-identical aggregation -- the run is resumable after a crash.
    """
    seeds = sorted(s for s in shap_by_seed if s != ref_seed)
    all_pairs = list(combinations(seeds, 2))
    if n_pairs and n_pairs < len(all_pairs):
        rng = np.random.default_rng(subset_seed)
        idx = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        pairs = [all_pairs[i] for i in sorted(idx)]
    else:
        pairs = all_pairs
    done = _load_ckpt(ckpt_path)
    if done:
        print(f"    [resume] {len(done)}/{len(pairs)} pairs already checkpointed", flush=True)
    fracs, md = [], []
    for n, (i, j) in enumerate(pairs):
        if (i, j) in done:
            f_ij, md_ij = done[(i, j)]
        else:
            (ar, at), (br, bt) = shap_by_seed[i], shap_by_seed[j]
            m = min(ar.shape[0], br.shape[0])  # guard tiny row-count drift
            d, praw = perm_diagnostic(ar[:m], at[:m], br[:m], bt[:m],
                                      blk_r[:m], blk_t[:m], bin_type, n_bins, seed=1000 * i + j)
            f_ij, md_ij = flagged_fraction(d, praw), float(d.mean())
            _append_ckpt(ckpt_path, i, j, f_ij, md_ij)
        fracs.append(f_ij)
        md.append(md_ij)
        if (n + 1) % 10 == 0:
            print(f"    {n+1}/{len(pairs)} pairs", flush=True)
    fracs, md = np.array(fracs), np.array(md)
    band = dict(mean=float(fracs.mean()), sd=float(fracs.std(ddof=1)),
                lo=float(np.percentile(fracs, 2.5)), hi=float(np.percentile(fracs, 97.5)),
                n_pairs=len(pairs))
    md_band = dict(md_mean=float(md.mean()),
                   md_lo=float(np.percentile(md, 2.5)), md_hi=float(np.percentile(md, 97.5)))
    return fracs, md, band, md_band


# --------------------------------------------------------------------------- #
# cache loaders                                                               #
# --------------------------------------------------------------------------- #
def _load_ieee(prefix, cache):
    """prefix='' -> xgb (nullshap_{split}_seed*.npy); 'catb_' -> catboost."""
    seeds = sorted({int(re.search(r"seed(\d+)", f.name).group(1))
                    for f in cache.glob(f"nullshap_{prefix}random_seed*.npy")})
    shap = {s: (np.load(cache / f"nullshap_{prefix}random_seed{s}.npy"),
                np.load(cache / f"nullshap_{prefix}temporal_seed{s}.npy")) for s in seeds}
    blk_r = np.load(cache / "evalblocks_random.npy")
    blk_t = np.load(cache / "evalblocks_temporal.npy")
    return shap, blk_r, blk_t


def _load_elliptic(learner, cache):
    seeds = sorted({int(re.search(r"seed(\d+)", f.name).group(1))
                    for f in cache.glob(f"nullshap_{learner}_random_seed*.npy")})
    shap = {s: (np.load(cache / f"nullshap_{learner}_random_seed{s}.npy"),
                np.load(cache / f"nullshap_{learner}_temporal_seed{s}.npy")) for s in seeds}
    blk_r = np.load(cache / "evalblocks_random.npy")
    blk_t = np.load(cache / "evalblocks_temporal.npy")
    return shap, blk_r, blk_t


CELLS = {
    "ieee_xgb":  lambda: _load_ieee("", C.STABILITY_DIR / "null_refits"),
    "ieee_catb": lambda: _load_ieee("catb_", C.STABILITY_DIR / "null_refits"),
    "elliptic_xgb":  lambda: _load_elliptic("xgb", C.STABILITY_DIR / "null_refits_elliptic"),
    "elliptic_catb": lambda: _load_elliptic("catb", C.STABILITY_DIR / "null_refits_elliptic"),
}
DATASET_LEARNER = {
    "ieee_xgb": ("IEEE-CIS", "XGBoost"), "ieee_catb": ("IEEE-CIS", "CatBoost"),
    "elliptic_xgb": ("Elliptic", "XGBoost"), "elliptic_catb": ("Elliptic", "CatBoost"),
}
OUT = ROOT / "results" / "tables" / "null_bands_all.csv"


def main(cells=None, binnings=("quantile", "equalwidth")):
    cells = cells or list(CELLS)
    rows = []
    if OUT.exists():
        rows = pd.read_csv(OUT).to_dict("records")
    for cell in cells:
        try:
            shap, blk_r, blk_t = CELLS[cell]()
        except (FileNotFoundError, ValueError) as e:
            print(f"[skip] {cell}: caches absent ({e})", flush=True)
            continue
        ds, lr = DATASET_LEARNER[cell]
        n_seeds = len(shap)
        for bt in binnings:
            print(f"[{cell}] {bt}: {n_seeds} seeds -> pairwise band", flush=True)
            ckpt = C.STABILITY_DIR / "null_refits" / f"ckpt_band_{cell}_{bt}.csv"
            fracs, md, band, md_band = pairwise_band(shap, blk_r, blk_t, bt, ckpt_path=ckpt)
            rows = [r for r in rows
                    if not (r["dataset"] == ds and r["learner"] == lr and r["binning"] == bt)]
            rows.append(dict(dataset=ds, learner=lr, binning=bt, n_seeds=n_seeds,
                             **band, **md_band))
            print(f"    band [{band['lo']*100:.1f}%, {band['hi']*100:.1f}%] "
                  f"mean {band['mean']*100:.1f}%  delta null [{md_band['md_lo']:+.4f}, "
                  f"{md_band['md_hi']:+.4f}]", flush=True)
    df = pd.DataFrame(rows).sort_values(["dataset", "learner", "binning"])
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT} ({len(df)} cells)")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None, choices=list(CELLS))
    ap.add_argument("--binnings", nargs="*", default=["quantile", "equalwidth"])
    a = ap.parse_args()
    raise SystemExit(main(cells=a.cells, binnings=a.binnings))
