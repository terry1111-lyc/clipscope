"""Retrieval metrics with binary relevance and cluster-aware uncertainty."""
from __future__ import annotations

import numpy as np


def retrieval_metrics(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    k = 10
    if not relevant or len(ranked) != len(set(ranked)):
        raise ValueError("Nonempty relevance and a unique ranking are required")
    hits = np.array([x in relevant for x in ranked], dtype=float)
    cut = hits[:k]
    positives = np.flatnonzero(hits)
    discounts = 1 / np.log2(np.arange(len(cut)) + 2)
    ideal = (1 / np.log2(np.arange(min(len(relevant), k)) + 2)).sum()
    return {
        "hit@1": float(hits[:1].sum() > 0),
        "hit@5": float(hits[:5].sum() > 0),
        "recall@10": float(cut.sum() / len(relevant)),
        "mrr": float(1 / (positives[0] + 1)) if len(positives) else 0.0,
        "map@10": float((np.cumsum(cut) / (np.arange(len(cut)) + 1) * cut).sum() / min(len(relevant), k)),
        "ndcg@10": float((cut * discounts).sum() / ideal),
    }


def cluster_interval(values: list[float], groups: list[str], seed: int = 42, draws: int = 2000) -> list[float]:
    """Percentile bootstrap; resample groups, keep all queries of each group."""
    if len(values) != len(groups) or not values:
        raise ValueError("Values and groups must be nonempty and aligned")
    unique = sorted(set(groups))
    arrays = [np.asarray([v for v, g in zip(values, groups) if g == key]) for key in unique]
    totals, counts = np.array([x.sum() for x in arrays]), np.array([len(x) for x in arrays])
    samples = np.random.default_rng(seed).integers(0, len(unique), (draws, len(unique)))
    boot = totals[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return np.quantile(boot, [0.025, 0.975]).tolist()


def summarize(rows: list[dict], seed: int = 42) -> dict:
    groups = [r["group"] for r in rows]
    return {key: {
        "mean": float(np.mean([r["metrics"][key] for r in rows])),
        "ci95": cluster_interval([r["metrics"][key] for r in rows], groups, seed),
    } for key in rows[0]["metrics"]}


def paired_comparison(a: dict, b: dict) -> dict:
    for key in ("dataset_sha256", "queries_sha256", "split", "direction"):
        if a[key] != b[key]:
            raise ValueError(f"Cannot compare reports with different {key}")
    left = {r["id"]: r for r in a["queries"]}
    right = {r["id"]: r for r in b["queries"]}
    if left.keys() != right.keys():
        raise ValueError("Reports must contain the same query IDs")
    keys = sorted(left)
    groups = [left[q]["group"] for q in keys]
    return {metric: {
        "delta_b_minus_a": float(np.mean(delta := [right[q]["metrics"][metric] - left[q]["metrics"][metric] for q in keys])),
        "paired_ci95": cluster_interval(delta, groups),
    } for metric in left[keys[0]]["metrics"]}
