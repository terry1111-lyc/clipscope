"""Import an already-downloaded, licensed COCO-format caption corpus.

No image download. The resulting subset is not the official COCO benchmark split.
"""
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from clipscope.data import safe_image, write_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.out.exists():
        p.error("Use a new output directory; existing datasets are never overwritten")
    if not 10 <= args.limit <= 20_000:
        p.error("limit must be 10–20,000")
    raw = json.loads(args.annotations.read_text())
    captions = defaultdict(list)
    for a in raw["annotations"]:
        if a["caption"].strip():
            captions[a["image_id"]].append(a["caption"].strip())
    images = sorted([x for x in raw["images"] if captions[x["id"]]], key=lambda x: x["id"])
    random.Random(args.seed).shuffle(images)
    images = images[:args.limit]
    if len(images) < 10:
        p.error("At least ten captioned images are required")
    # Validate all paths before creating the destination.
    sources = [safe_image(args.images, x["file_name"]) for x in images]
    (args.out / "images").mkdir(parents=True)
    records, queries = [], []
    for i, (image, source) in enumerate(zip(images, sources)):
        key = str(image["id"])
        name = key + source.suffix.lower()
        shutil.copyfile(source, args.out / "images" / name)
        records.append({"id": key, "path": "images/" + name, "caption": captions[image["id"]][0],
                        "license": image.get("license", "check source corpus terms")})
        split = "dev" if i < max(1, len(images) // 5) else "test"
        for j, text in enumerate(captions[image["id"]]):
            queries.append({"id": f"{key}-{j}", "source_id": key, "text": text,
                            "relevant": [key], "group": key, "split": split, "slice": "natural-caption"})
    write_jsonl(args.out / "manifest.jsonl", records)
    write_jsonl(args.out / "queries.jsonl", queries)
    (args.out / "DATASET_CARD.md").write_text(
        f"# Local COCO-format subset\n\nImages: {len(records)}; seed: {args.seed}.\n"
        "20% dev / 80% test source-image groups; a shared fixed gallery. Not official COCO protocol.\n"
        "Image-to-text uses only the first caption per image. Text-to-image uses all captions.\n"
        "Verify source image and annotation licenses before redistribution. Images are not relicensed by this project.\n")
    print(f"Imported {len(records)} images and {len(queries)} queries into {args.out}")


if __name__ == "__main__":
    main()
