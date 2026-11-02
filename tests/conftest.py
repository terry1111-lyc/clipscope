"""Small analytical encoder used only by unit tests; never exposed as a model."""
from pathlib import Path
import numpy as np
import pytest
from clipscope.encoder import normalize
from clipscope.index import VectorIndex, build_index
from clipscope.synthetic import generate


class FakeEncoder:
    fingerprint = "test-only"
    info = {"model": "test-only", "architecture": "fixture", "pretrained": "none"}
    device = "cpu"

    def images(self, images, batch_size=8):
        from clipscope.data import load_image
        return normalize(np.array([np.asarray(load_image(x) if isinstance(x, Path) else x).mean(axis=(0, 1)) for x in images]))

    def texts(self, texts, ensemble=False):
        return normalize(np.tile([1., 2., 3.], (len(texts), 1)))


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "shapes"
    generate(root)
    encoder = FakeEncoder()
    path = tmp_path / "index.sqlite"
    build_index(root / "manifest.jsonl", path, encoder)
    return root, encoder, VectorIndex(path, encoder.fingerprint), path
