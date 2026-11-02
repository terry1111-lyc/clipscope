"""Complete rankings, per-query evidence, slices and reproducibility metadata."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import read_jsonl
from .encoder import PROMPTS
from .metrics import retrieval_metrics, summarize


def package_versions() -> dict:
    versions = {}
    for name in ("numpy", "Pillow", "torch", "open_clip_torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def source_digest() -> str:
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def evaluate(index, encoder, queries_path: Path, split: str = "test", ensemble: bool = False,
             direction: str = "text-to-image", seed: int = 42) -> dict:
    if index.meta["fingerprint"] != encoder.fingerprint:
        raise ValueError("Encoder/index mismatch")
    queries = read_jsonl(queries_path)
    seen = set()
    group_splits = {}
    source_groups = {}
    gallery = set(index.ids)
    for q in queries:
        if q["id"] in seen or not isinstance(q.get("group"), str) or not q["group"]:
            raise ValueError("Queries require unique IDs and nonempty bootstrap groups")
        seen.add(q["id"])
        if q["group"] in group_splits and group_splits[q["group"]] != q["split"]:
            raise ValueError("A query group crosses dataset splits")
        group_splits[q["group"]] = q["split"]
        source = q.get("source_id")
        if source in source_groups and source_groups[source] != q["group"]:
            raise ValueError("Queries from the same source image must share a bootstrap group")
        source_groups[source] = q["group"]
        if not q.get("relevant") or not set(q["relevant"]).issubset(gallery):
            raise ValueError("Every query must have relevant IDs in the gallery")
        if q.get("source_id") not in gallery or not q.get("text", "").strip():
            raise ValueError("Queries require an indexed source_id and nonempty text")
    selected = [q for q in queries if q["split"] == split]
    if not selected:
        raise ValueError(f"No queries for split {split}")
    query_hash = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    if direction == "image-to-text":
        # One caption per image; evaluate each source image only once.
        selected = list({q["source_id"]: q for q in selected}.values())
        selected = [{**q, "id": q["source_id"] + "-image-query", "slice": "image-caption",
                     "text": index.by_id[q["source_id"]]["caption"]} for q in selected]
        candidates = encoder.texts([r["caption"] for r in index.records], ensemble)
        source_positions = {key: i for i, key in enumerate(index.ids)}
    elif direction != "text-to-image":
        raise ValueError("Unknown retrieval direction")
    encoder.texts([selected[0]["text"]], ensemble)  # Explicit warmup, excluded from latency.
    rows, elapsed = [], []
    for q in selected:
        start = time.perf_counter()
        if direction == "text-to-image":
            vector = encoder.texts([q["text"]], ensemble)[0]
            ranking = index.search(vector, len(index.ids))
            relevant = set(q["relevant"])
        else:
            vector = index.vectors[source_positions[q["source_id"]]]
            scores = candidates @ vector
            ranking = [{"id": index.ids[i], "score": float(scores[i])} for i in np.argsort(-scores, kind="stable")]
            relevant = {q["source_id"]}
        elapsed.append((time.perf_counter() - start) * 1000)
        rows.append({
            "id": q["id"], "text": q["text"], "source_id": q["source_id"], "group": q["group"],
            "slice": q.get("slice", "unspecified"), "relevant": sorted(relevant),
            "metrics": retrieval_metrics([r["id"] for r in ranking], relevant),
            "ranking": [{"id": r["id"], "score": r["score"]} for r in ranking],
        })
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "uncommitted"
    return {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "source_sha256": source_digest(),
        "dataset_sha256": index.meta["dataset_sha256"], "queries_sha256": query_hash,
        "encoder": encoder.info, "encoder_fingerprint": encoder.fingerprint,
        "device": encoder.device, "split": split, "direction": direction, "seed": seed,
        "ensemble": ensemble, "prompts": list(PROMPTS) if ensemble else ["{}"],
        "count": len(rows), "bootstrap_groups": len({r["group"] for r in rows}),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": package_versions()},
        "latency_ms": {"p50": float(np.median(elapsed)), "p95": float(np.percentile(elapsed, 95)),
                       "scope": "warm text encoding + exact ranking" if direction == "text-to-image" else "ranking only; image/caption embeddings cached"},
        "metrics": summarize(rows, seed),
        "slices": {s: summarize([r for r in rows if r["slice"] == s], seed) for s in sorted({r["slice"] for r in rows})},
        "queries": rows,
    }


def write_report(report: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Measured retrieval run", "", f"Model: `{report['encoder']['model']}` · device: `{report['device']}` · split: `{report['split']}`",
             f"Direction: {report['direction']} · ensemble: {report['ensemble']} · queries: {report['count']} · bootstrap groups: {report['bootstrap_groups']}", "",
             "When using the bundled shape dataset, these results are a compositional diagnostic, not a natural-image benchmark. Dataset scope must be considered before generalizing.", "",
             "| Metric | Mean | Cluster bootstrap 95% CI |", "|---|---:|---:|"]
    for metric, value in report["metrics"].items():
        lines.append(f"| {metric} | {value['mean']:.4f} | [{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}] |")
    lines += ["", f"Warm latency p50/p95: {report['latency_ms']['p50']:.2f}/{report['latency_ms']['p95']:.2f} ms. Scope: {report['latency_ms']['scope']}.", "", "## Lowest-scoring queries", ""]
    for q in sorted(report["queries"], key=lambda x: x["metrics"]["ndcg@10"])[:8]:
        lines.append(f"- `{q['id']}`: {q['text']} — top result `{q['ranking'][0]['id']}`, relevant `{', '.join(q['relevant'])}`; nDCG@10={q['metrics']['ndcg@10']:.3f}")
    lines += ["", "Exact rankings, package versions, dataset/query hashes and prompt templates are in the adjacent JSON file.", ""]
    destination.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
