import numpy as np
import pytest
from clipscope.data import load_manifest, safe_image, read_jsonl, write_jsonl
from clipscope.encoder import normalize
from clipscope.index import VectorIndex, build_index


def test_snapshot_reloads_and_refuses_wrong_model(dataset):
    root, encoder, index, path = dataset
    assert len(index.ids) == 48
    hit = index.search(index.vectors[0], 1)[0]
    assert hit["score"] == pytest.approx(1, abs=1e-6)
    assert VectorIndex(path, encoder.fingerprint).ids == index.ids
    with pytest.raises(ValueError, match="mismatch"): VectorIndex(path, "another-model")


def test_mmr_unique_and_bounds(dataset):
    _, _, index, _ = dataset
    result = index.search(index.vectors[0], 12, .5)
    assert len({r["id"] for r in result}) == 12
    with pytest.raises(ValueError): index.search(index.vectors[0], 0)
    with pytest.raises(ValueError): index.search(np.ones(7), 1)
    with pytest.raises(ValueError): normalize(np.array([0, 0]))
    with pytest.raises(ValueError): normalize(np.array([np.nan, 1]))


def test_safe_path_and_duplicates(dataset, tmp_path):
    root, _, _, _ = dataset
    outsider = tmp_path / "outside.png"; outsider.write_bytes(b"x")
    with pytest.raises(ValueError): safe_image(root, "../outside.png")
    (root / "escape.png").symlink_to(outsider)
    with pytest.raises(ValueError): safe_image(root, "escape.png")
    rows = read_jsonl(root / "manifest.jsonl")
    rows[1]["path"] = rows[0]["path"]
    write_jsonl(root / "bad.jsonl", rows)
    with pytest.raises(ValueError, match="Duplicate"): load_manifest(root / "bad.jsonl")


def test_failed_rebuild_preserves_prior_snapshot(dataset):
    root, encoder, index, path = dataset
    original = path.read_bytes()
    def fail(*args): raise RuntimeError("out of memory")
    encoder.images = fail
    with pytest.raises(RuntimeError): build_index(root / "manifest.jsonl", path, encoder)
    assert path.read_bytes() == original
