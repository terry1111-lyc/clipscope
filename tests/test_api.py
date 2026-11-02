import pytest
from fastapi.testclient import TestClient
from clipscope.api import create_app


def test_search_upload_errors_and_safe_assets(dataset):
    root, encoder, index, _ = dataset
    with TestClient(create_app(encoder, index, root)) as client:
        assert client.get("/healthz").json()["indexed"] == 48
        assert client.get("/").status_code == 200
        assert client.get("/api/status").json()["model"]["model"] == "test-only"
        result = client.post("/api/search", json={"text": "red circle", "k": 4}).json()
        assert len(result["results"]) == 4
        assert client.get("/api/images/" + result["results"][0]["id"]).status_code == 200
        assert client.get("/api/images/not-found").status_code == 404
        assert client.post("/api/search", json={"text": "  "}).status_code == 422
        assert client.post("/api/search", json={"text": "x", "k": 0}).status_code == 422
        image = next((root / "images").glob("*.png")).read_bytes()
        assert client.post("/api/search/image", files={"image": ("x.png", image, "image/png")}, data={"text": "red", "alpha": .4}).status_code == 200
        assert client.post("/api/search/image", files={"image": ("x.png", b"invalid")}).status_code == 422
        assert client.post("/api/search", content=b"x" * (8 * 1024 * 1024 + 1)).status_code == 413


def test_token_protects_queries_images_and_reports(dataset, monkeypatch):
    root, encoder, index, _ = dataset
    monkeypatch.setenv("CLIPSCOPE_API_TOKEN", "test-token")
    with TestClient(create_app(encoder, index, root)) as client:
        for path in ("/api/status", "/api/reports", "/api/images/red-circle-left-1"):
            assert client.get(path).status_code == 401
        assert client.get("/api/status", headers={"Authorization": "Bearer test-token"}).status_code == 200
        assert client.get("/healthz").status_code == 200


def test_changed_dataset_refused_at_startup(dataset):
    root, encoder, index, _ = dataset
    (root / index.records[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        with TestClient(create_app(encoder, index, root)): pass
