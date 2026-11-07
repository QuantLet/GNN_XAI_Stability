"""Freeze-and-extract GraphSAGE embeddings and the control variants.

extract_embeddings(): single frozen forward pass on the leakage-controlled
inference graph, returning z^(1), z^(2) in R^64 per transaction (Section 3.3).

The control constructors (Section 3.5) operate on the real layer-one embedding:
  random      -> independent Gaussian matched to per-column std
  permuted    -> row-shuffle (destroys tx-entity alignment, keeps marginals)
  noise       -> real z^(1) + N(0, (0.1*sigma)^2) at inference
  covmatched  -> N(0, Sigma_hat) per row, full 64x64 covariance, no graph signal
"""
from __future__ import annotations

import numpy as np
import torch

from . import config as C


@torch.no_grad()
def extract_embeddings(model, data, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Frozen forward pass -> (z1, z2) transaction embeddings (rule 4)."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    data = data.to(device)
    z1, z2 = model.encode(data.x_dict, data.edge_index_dict)
    return (z1["transaction"].cpu().numpy().astype(np.float32),
            z2["transaction"].cpu().numpy().astype(np.float32))


def make_control(kind: str, z1: np.ndarray, seed: int = C.SEED) -> np.ndarray:
    """Return a control embedding derived from the real layer-one embedding z1."""
    rng = np.random.default_rng(seed)
    if kind == "random":
        sd = z1.std(axis=0, keepdims=True)
        return rng.standard_normal(z1.shape).astype(np.float32) * sd
    if kind == "permuted":
        perm = rng.permutation(len(z1))
        return z1[perm].copy()
    if kind == "noise":
        sd = z1.std(axis=0, keepdims=True)
        return (z1 + rng.standard_normal(z1.shape).astype(np.float32) * C.NOISE_INJ_SIGMA * sd)
    if kind == "covmatched":
        cov = np.cov(z1, rowvar=False)
        L = np.linalg.cholesky(cov + 1e-6 * np.eye(cov.shape[0]))
        g = rng.standard_normal(z1.shape).astype(np.float64)
        return (g @ L.T).astype(np.float32)
    raise ValueError(f"unknown control kind {kind!r}")


def embedding_for_variant(embed: str | None, z1: np.ndarray, z2: np.ndarray,
                          seed: int = C.SEED) -> np.ndarray | None:
    """Resolve a Variant.embed tag to the concrete embedding columns."""
    if embed is None:
        return None
    if embed == "l1":
        return z1
    if embed == "l2":
        return z2
    return make_control(embed, z1, seed=seed)


def main(device: str | None = None) -> None:
    """Stage 1 (original API): train GraphSAGE per split, cache embeddings + split
    data, and write sage_only predictions from the GNN head.

    Uses the auto-detected device (GPU when available) and falls back to CPU for a
    split if the full-batch graph exceeds GPU memory. Control embeddings are
    deterministic functions of z1 (make_control), regenerated on demand.
    """
    import pandas as pd
    import torch

    from . import data_prep, splits, graph, gnn, pipeline, utils

    log = utils.get_logger()
    device = device or C.DEVICE
    log.info("device: %s", device)
    df = data_prep.load_raw()
    for split in C.SPLITS:
        C.seed_everything()
        log.info("preparing split %s", split)
        mask = splits.get_split_mask(df, split)
        Xdf, ent = data_prep.build_features(df, mask)
        Xraw = Xdf.values.astype(np.float32)
        Xstd = graph.standardize(Xraw[mask], Xraw).astype(np.float32)

        data = graph.build_hetero_graph(df, Xstd, mask)
        dev = device
        try:
            model = gnn.train_graphsage(data, device=dev)
            z1, z2 = extract_embeddings(model, data, device=dev)
        except torch.cuda.OutOfMemoryError:
            log.warning("    GPU OOM on split %s; falling back to CPU for the GNN", split)
            torch.cuda.empty_cache()
            dev = "cpu"
            data = graph.build_hetero_graph(df, Xstd, mask)
            model = gnn.train_graphsage(data, device=dev)
            z1, z2 = extract_embeddings(model, data, device=dev)

        sd = pipeline.SplitData(
            split=split, df=df, X_tab=Xstd, tab_names=list(Xdf.columns),
            entity_names=ent, train_mask=mask, z1=z1, z2=z2,
            blocks=splits.block_ids(df),
        )
        pipeline.save_split(sd)

        # sage_only: probabilities from the trained GNN classification head
        with torch.no_grad():
            d = data.to(dev)
            logits, _, _ = model(d.x_dict, d.edge_index_dict)
            proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        val = ~mask
        pd.DataFrame({"y_true": df[C.TARGET].values[val], "y_proba": proba[val]}).to_parquet(
            C.STABILITY_DIR / f"preds_sage_only_{split}.parquet")
        log.info("  cached embeddings + split data + sage_only preds for %s", split)
