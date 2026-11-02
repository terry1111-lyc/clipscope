"""Read-only serving API. Index mutation is an operator-only CLI operation."""
from __future__ import annotations

import io
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import UnidentifiedImageError, Image
from starlette.concurrency import run_in_threadpool

from .data import digest, load_image, safe_image
from .encoder import OpenCLIPEncoder, normalize
from .index import VectorIndex

MAX_BODY = 8 * 1024 * 1024


class BodyLimit:
    """Bound actual bytes as well as Content-Length, including chunked uploads."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > MAX_BODY:
                return await JSONResponse({"detail": "Request exceeds 8 MiB"}, status_code=413)(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break
        async def replay():
            return chunks.pop(0) if chunks else await receive()
        await self.app(scope, replay, send)


class SearchRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    k: int = Field(default=12, ge=1, le=100)
    ensemble: bool = False
    diversity: float = Field(default=0, ge=0, le=1)


def create_app(encoder=None, index=None, data_root: Path | None = None, report_dir: Path | None = None) -> FastAPI:
    root = (data_root or Path(os.getenv("CLIPSCOPE_DATA", "data/shapes"))).resolve()
    reports = (report_dir or Path(os.getenv("CLIPSCOPE_REPORTS", "experiments/results"))).resolve()

    @asynccontextmanager
    async def lifespan(app):
        app.state.encoder = encoder or OpenCLIPEncoder(
            os.getenv("CLIPSCOPE_MODEL", "b32-openai"), os.getenv("CLIPSCOPE_DEVICE", "auto"),
            os.getenv("CLIPSCOPE_CACHE", ".cache/models"))
        app.state.index = index or VectorIndex(Path(os.getenv("CLIPSCOPE_INDEX", "data/index.sqlite")), app.state.encoder.fingerprint)
        if app.state.index.meta["fingerprint"] != app.state.encoder.fingerprint:
            raise ValueError("Encoder/index mismatch")
        # Fail before serving when assets moved or changed after indexing.
        for row in app.state.index.records:
            if digest(safe_image(root, row["path"])) != row["sha256"]:
                raise ValueError(f"Indexed image changed: {row['id']}")
        yield

    app = FastAPI(title="CLIPScope", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimit)
    token = os.getenv("CLIPSCOPE_API_TOKEN", "")

    @app.middleware("http")
    async def authorization(request: Request, call_next):
        if token and request.url.path.startswith("/api/"):
            supplied = request.headers.get("authorization", "")
            if not secrets.compare_digest(supplied, "Bearer " + token):
                return JSONResponse({"detail": "Bearer token required"}, status_code=401)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/healthz")
    def health():
        return {"status": "ok", "indexed": len(app.state.index.ids)}

    @app.get("/api/status")
    def status():
        return {"model": app.state.encoder.info, "device": app.state.encoder.device,
                "index": app.state.index.meta, "max_upload_mib": 8}

    def results(vector, k, diversity, start):
        found = app.state.index.search(vector, min(k, len(app.state.index.ids)), diversity)
        return {"results": [{"id": r["id"], "caption": r["caption"], "score": r["score"],
                             "rank": r["rank"], "tags": r.get("tags", [])} for r in found],
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
                "score_kind": "cosine similarity; not a probability"}

    @app.post("/api/search")
    def search(query: SearchRequest):
        if not query.text.strip():
            raise HTTPException(422, "Text must not be blank")
        start = time.perf_counter()
        vector = app.state.encoder.texts([query.text], query.ensemble)[0]
        return results(vector, query.k, query.diversity, start)

    @app.post("/api/search/image")
    async def search_image(image: UploadFile = File(...), text: str = Form("", max_length=500),
                           alpha: float = Form(0.5, ge=0, le=1), k: int = Form(12, ge=1, le=100)):
        raw = await image.read(MAX_BODY + 1)
        await image.close()
        try:
            im = load_image(io.BytesIO(raw))
        except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise HTTPException(422, "Invalid image or image exceeds 16 megapixels") from exc
        def infer():
            start = time.perf_counter()
            vector = app.state.encoder.images([im])[0]
            if text.strip():
                vector = normalize((1 - alpha) * vector + alpha * app.state.encoder.texts([text])[0])
            return results(vector, k, 0, start)
        return await run_in_threadpool(infer)

    @app.get("/api/images/{image_id}")
    def get_image(image_id: str):
        row = app.state.index.by_id.get(image_id)
        if row is None:
            raise HTTPException(404, "Unknown image ID")
        return FileResponse(safe_image(root, row["path"]))

    @app.get("/api/reports")
    def get_reports():
        result = []
        for path in sorted(reports.glob("*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            if "metrics" in report and "encoder" in report:
                result.append({"name": path.stem, **{k: report[k] for k in
                    ("encoder", "device", "split", "direction", "ensemble", "count", "metrics", "latency_ms", "slices")},
                    "failures": sorted(report["queries"], key=lambda q: q["metrics"]["ndcg@10"])[:5]})
        return result

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def home():
        return FileResponse(static / "index.html")

    return app
