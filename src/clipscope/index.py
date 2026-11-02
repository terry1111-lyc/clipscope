"""Atomic SQLite snapshot + exact cosine search. No approximate-index confounder."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

from .data import load_manifest, safe_image
from .encoder import normalize


def build_index(manifest: Path, destination: Path, encoder, batch_size: int = 8) -> dict:
    rows = load_manifest(manifest)
    vectors = normalize(encoder.images([safe_image(manifest.parent, row["path"]) for row in rows], batch_size))
    if len(vectors) != len(rows):
        raise ValueError("Encoder returned an incorrect number of vectors")
    meta = {
        "schema": 1, "fingerprint": encoder.fingerprint, "encoder": encoder.info,
        "dimension": int(vectors.shape[1]), "count": len(rows),
        "dataset_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=destination.parent, suffix=".sqlite")
    os.close(fd)
    try:
        with sqlite3.connect(tmp) as db:
            db.execute("CREATE TABLE metadata (value TEXT NOT NULL)")
            db.execute("CREATE TABLE images (id TEXT PRIMARY KEY, record TEXT NOT NULL, vector BLOB NOT NULL)")
            db.execute("INSERT INTO metadata VALUES (?)", (json.dumps(meta),))
            db.executemany("INSERT INTO images VALUES (?, ?, ?)", [
                (r["id"], json.dumps(r), v.astype("<f4").tobytes()) for r, v in zip(rows, vectors)
            ])
        os.replace(tmp, destination)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return meta


class VectorIndex:
    def __init__(self, path: Path, fingerprint: str | None = None):
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            self.meta = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
            rows = db.execute("SELECT record, vector FROM images ORDER BY id").fetchall()
        if self.meta["schema"] != 1:
            raise ValueError("Unsupported index schema; rebuild the index")
        if fingerprint is not None and self.meta["fingerprint"] != fingerprint:
            raise ValueError("Model/preprocessing mismatch. Rebuild this index with the selected model")
        self.records = [json.loads(r[0]) for r in rows]
        self.vectors = normalize(np.stack([np.frombuffer(r[1], dtype="<f4") for r in rows]))
        if self.vectors.shape != (self.meta["count"], self.meta["dimension"]):
            raise ValueError("Corrupt index dimensions")
        self.ids = [r["id"] for r in self.records]
        self.by_id = dict(zip(self.ids, self.records))

    def search(self, query: np.ndarray, k: int = 12, diversity: float = 0) -> list[dict]:
        if not 1 <= k <= len(self.ids) or not 0 <= diversity <= 1:
            raise ValueError("Invalid k or diversity")
        query = normalize(query).reshape(-1)
        if query.shape != (self.vectors.shape[1],):
            raise ValueError("Query embedding dimension mismatch")
        scores = self.vectors @ query
        order = np.argsort(-scores, kind="stable")
        if diversity:
            # MMR on a bounded candidate pool; score remains original cosine.
            candidates = list(order[:max(k, min(len(order), 5 * k))])
            selected = [candidates.pop(0)]
            while len(selected) < k:
                redundancy = (self.vectors[candidates] @ self.vectors[selected].T).max(axis=1)
                utility = (1 - diversity) * scores[candidates] - diversity * redundancy
                selected.append(candidates.pop(int(np.argmax(utility))))
            order = selected
        return [{**self.records[i], "score": float(scores[i]), "rank": rank + 1} for rank, i in enumerate(order[:k])]
