"""Validated local datasets. No network fetching or arbitrary path serving."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 16_000_000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def safe_image(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Image must be an existing file inside the dataset directory")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("Only PNG, JPEG and WebP images are supported")
    return path


def load_image(source) -> Image.Image:
    with Image.open(source) as im:
        if im.width * im.height > Image.MAX_IMAGE_PIXELS:
            raise ValueError("Image exceeds the 16 megapixel limit")
        return ImageOps.exif_transpose(im).convert("RGB")


def load_manifest(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    if not rows or len(rows) > 20_000:
        raise ValueError("Dataset must contain 1–20,000 images")
    seen_ids, seen_hashes = set(), set()
    for row in rows:
        if not isinstance(row.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", row["id"]) or row["id"] in seen_ids:
            raise ValueError("Image IDs must be unique, URL-safe identifiers of at most 128 characters")
        if not isinstance(row.get("caption"), str) or not row["caption"].strip():
            raise ValueError("Every image requires a caption for audit and image-to-text evaluation")
        image = safe_image(path.parent, row["path"])
        sha = digest(image)
        if sha in seen_hashes:
            raise ValueError("Duplicate image bytes detected; deduplicate before evaluation")
        seen_ids.add(row["id"])
        seen_hashes.add(sha)
        row["sha256"] = sha
    return rows
