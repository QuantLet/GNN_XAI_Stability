"""Instance-level agreement with the REFIT as the unit.

Why. sec:results-instance is the paper's only positive result, and until now it was the
one claim read against a band built from twelve dependent fits -- exactly the object
sec:results-refitlevel declines to use for the negative. Judging the positive by a
weaker standard than the negative is not defensible, so this recomputes the per-row
statistics with the refit as the unit and no band involved.

Construction. sec:results-instance asks whether the real embedding moves per-row
explanations FURTHER FROM THE BASELINE than a graph-free control does. At refit level
that is a paired difference, and all three arms exist at seeds 10-21 with SHAP
persisted -- baseline (null_refits/), real hybrid (null_refits_hybrid/) and
covariance-matched control (null_refits_covmatched/). For each seed s and statistic T
we compute

    Delta_s = T(baseline_s, hybrid_s) - T(baseline_s, covmatched_s)

over the common tabular block on the same evaluation rows. T is agreement, so
Delta_s < 0 means the real embedding agrees with the baseline LESS than the control
does -- the direction sec:results-instance reports. Each seed contributes one Delta,
the seeds are independent, and an ordinary t-interval applies with no band involved.

Two sources, ONE estimator. --source selects which set of refits supplies the arms;
everything downstream of the file paths is shared code, so the two readings are
comparable by construction rather than by inspection.

  stochastic (default)  seeds 10-21, stability/null_refits{,_hybrid,_covmatched}/.
                        The learner runs subsample = 0.8, so these are refits under the
                        SECONDARY specification -- for a long time the only arms whose
                        SHAP was persisted, hence the only refit-level reading available.
  pinned                seeds 201-220, the --persist-shap arms of
                        seed_sensitivity_unconditional.py. Whole-pipeline unconditional
                        replicates under the paper's PRIMARY pinned learner: both sides
                        of the contrast are re-drawn per seed and all three arms are kept.

The pinned source reads the seed list from seed_sensitivity_unconditional_shap.csv
rather than from the cache directory. That csv row is appended only after BOTH sides of
a seed finish, so it is the completion marker -- globbing the cache directly would pick
up a seed the run is still writing.

Writes results/tables/instance_level_refitlevel.csv (stochastic) or
instance_level_refitlevel_pinned.csv (pinned).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402

BASE = C.STABILITY_DIR / "null_refits"
HYB = C.STABILITY_DIR / "null_refits_hybrid"
COV = C.STABILITY_DIR / "null_refits_covmatched"
# Mirrors seed_sensitivity_unconditional.SHAP_DIR: the pinned arms are a recomputable
# cache kept off the Drive-synced tree, and the same override names it.
UNCOND = Path(os.environ.get("UNCOND_SHAP_DIR",
                             Path.home() / ".cache" / "gnnxai" / "uncond_refits"))
UNCOND_CSV = ROOT / "results" / "tables" / "seed_sensitivity_unconditional_shap.csv"
N_TAB = 486
TOPK = 10
OUT = {"stochastic": ROOT / "results" / "tables" / "instance_level_refitlevel.csv",
       "pinned": ROOT / "results" / "tables" / "instance_level_refitlevel_pinned.csv"}


def _topk_overlap(A: np.ndarray, B: np.ndarray, k: int = TOPK) -> float:
    """Mean over rows of |top-k(A_i) ∩ top-k(B_i)| / k, on |phi|."""
    ia = np.argpartition(-np.abs(A), k - 1, axis=1)[:, :k]
    ib = np.argpartition(-np.abs(B), k - 1, axis=1)[:, :k]
    out = np.empty(len(A))
    for i in range(len(A)):
        out[i] = len(np.intersect1d(ia[i], ib[i], assume_unique=True)) / k
    return float(out.mean())


def _spearman(A: np.ndarray, B: np.ndarray) -> float:
    """Mean over rows of the Spearman correlation of |phi| across features."""
    ra = stats.rankdata(np.abs(A), axis=1)
    rb = stats.rankdata(np.abs(B), axis=1)
    ra = ra - ra.mean(1, keepdims=True)
    rb = rb - rb.mean(1, keepdims=True)
    num = (ra * rb).sum(1)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    ok = den > 0
    return float((num[ok] / den[ok]).mean())


def _paths(source: str, seed: int, split: str) -> tuple[Path, Path, Path]:
    """(baseline, hybrid, covmatched) SHAP paths for one (seed, split)."""
    if source == "stochastic":
        return (BASE / f"nullshap_{split}_seed{seed}.npy",
                HYB / f"nullshap_sage_l1_{split}_seed{seed}.npy",
                COV / f"nullshap_covmatched_{split}_seed{seed}.npy")
    return (UNCOND / f"uncond_tab_{split}_seed{seed}.npy",
            UNCOND / f"uncond_hyb_{split}_seed{seed}.npy",
            UNCOND / f"uncond_cov_{split}_seed{seed}.npy")


def seeds(source: str) -> list[int]:
    if source == "stochastic":
        h = {int(re.search(r"seed(\d+)", f.name).group(1))
             for f in HYB.glob("nullshap_sage_l1_random_seed*.npy")}
        c = {int(re.search(r"seed(\d+)", f.name).group(1))
             for f in COV.glob("nullshap_covmatched_random_seed*.npy")}
        return sorted(h & c)
    # Completion marker, not a glob: a seed still being written has arrays on disk but
    # no csv row (the row is appended after both sides finish).
    if not UNCOND_CSV.exists():
        return []
    done = sorted(int(s) for s in pd.read_csv(UNCOND_CSV)["seed"])
    out = []
    for s in done:
        missing = [p.name for split in ("random", "temporal")
                   for p in _paths("pinned", s, split) if not p.exists()]
        if missing:
            # A csv row without its arrays means the run wrote the row in plain mode.
            # Loud, because silently dropping seeds would shrink n without saying so.
            raise SystemExit(f"seed {s} is in {UNCOND_CSV.name} but missing "
                             f"{len(missing)} cached arm(s): {missing[0]} ... "
                             f"was it run without --persist-shap?")
        out.append(s)
    return out


def main(source: str = "stochastic") -> int:
    rows = []
    for s in seeds(source):
        for split in ("random", "temporal"):
            p0, ph, pc = _paths(source, s, split)
            t0, hy, cv = np.load(p0), np.load(ph), np.load(pc)
            for name, a in (("baseline", t0), ("hybrid", hy), ("covmatched", cv)):
                if a.ndim != 2 or a.shape[1] < N_TAB:
                    raise SystemExit(f"seed {s} {split} {name}: expected >= {N_TAB} "
                                     f"columns, got shape {a.shape}")
            # The tabular block is the first N_TAB columns of every arm by construction
            # (build_input concatenates the embedding AFTER the standardized tabular
            # matrix), so this slice compares like with like across arms.
            t0, hy, cv = t0[:, :N_TAB], hy[:, :N_TAB], cv[:, :N_TAB]
            m = min(len(t0), len(hy), len(cv))
            t0, hy, cv = t0[:m], hy[:m], cv[:m]
            oh, oc = _topk_overlap(t0, hy), _topk_overlap(t0, cv)
            sh, sc = _spearman(t0, hy), _spearman(t0, cv)
            rows.append(dict(seed=s, split=split,
                             overlap_hyb=oh, overlap_cov=oc, d_overlap=oh - oc,
                             spearman_hyb=sh, spearman_cov=sc, d_spearman=sh - sc))
            print(f"  seed {s} {split}: d_top{TOPK} {oh-oc:+.4f}  "
                  f"d_spearman {sh-sc:+.4f}", flush=True)
    if not rows:
        print(f"no paired refit caches found for source={source}")
        return 1
    d = pd.DataFrame(rows)
    out = OUT[source]
    d.to_csv(out, index=False)

    print(f"\nREFIT-LEVEL PAIRED DIFFERENCE  (hybrid - covmatched, both vs baseline)")
    print(f"  source: {source} "
          f"({'pinned learner, unconditional whole-pipeline replicates' if source == 'pinned' else 'stochastic learner, subsample = 0.8'})")
    print("  Delta < 0  =>  the real embedding agrees with the baseline LESS than the control")
    for split in ("random", "temporal"):
        sub = d[d.split == split]
        for col in ("d_overlap", "d_spearman"):
            x = sub[col].values
            n = len(x)
            mu, sd = x.mean(), x.std(ddof=1)
            se = sd / np.sqrt(n)
            t = stats.t.ppf(0.975, n - 1)
            lo, hi = mu - t * se, mu + t * se
            print(f"  {split:9s} {col:11s} n={n}  mean {mu:+.4f} (sd {sd:.4f})  "
                  f"95% t [{lo:+.4f}, {hi:+.4f}]  -> "
                  f"{'EXCLUDES' if lo > 0 or hi < 0 else 'INCLUDES'} zero")

    other = OUT["stochastic" if source == "pinned" else "pinned"]
    if other.exists():
        # Same estimator, different learner specification. Report the two side by side
        # rather than letting the reader assume the positive transfers.
        o = pd.read_csv(other)
        print(f"\n  against {other.name}:")
        for split in ("random", "temporal"):
            for col in ("d_overlap", "d_spearman"):
                a = d[d.split == split][col].mean()
                b = o[o.split == split][col].mean()
                print(f"    {split:9s} {col:11s} this {a:+.4f}   other {b:+.4f}   "
                      f"diff {a - b:+.4f}")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="stochastic", choices=["stochastic", "pinned"],
                    help="which refits supply the arms (see module docstring)")
    raise SystemExit(main(ap.parse_args().source))
