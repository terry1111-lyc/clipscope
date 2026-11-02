"""Original, reproducible shape diagnostic. Contains no external image assets."""
from pathlib import Path

from PIL import Image, ImageDraw

from .data import write_jsonl

COLORS = {"red": "#d94444", "blue": "#3976ce", "green": "#358454", "yellow": "#e5b830"}


def generate(destination: Path) -> None:
    (destination / "images").mkdir(parents=True, exist_ok=True)
    records, queries = [], []
    for ci, (color, fill) in enumerate(COLORS.items()):
        for si, shape in enumerate(("circle", "square", "triangle")):
            group = f"{color}-{shape}"
            split = "test" if (ci + si) % 2 else "dev"
            for position in ("left", "right"):
                for count in (1, 2):
                    key = f"{group}-{position}-{count}"
                    im = Image.new("RGB", (336, 336), "#f5f3ee")
                    draw = ImageDraw.Draw(im)
                    x = 85 if position == "left" else 251
                    for y in ([168] if count == 1 else [109, 227]):
                        box = (x - 42, y - 42, x + 42, y + 42)
                        if shape == "circle":
                            draw.ellipse(box, fill=fill)
                        elif shape == "square":
                            draw.rectangle(box, fill=fill)
                        else:
                            draw.polygon([(x, y - 46), (x - 46, y + 38), (x + 46, y + 38)], fill=fill)
                    im.save(destination / "images" / f"{key}.png")
                    objects = f"{'one' if count == 1 else 'two'} {color} {shape}{'s' if count == 2 else ''}"
                    caption = f"{objects} on the {position} side of a plain background"
                    records.append({"id": key, "path": f"images/{key}.png", "caption": caption,
                                    "tags": [color, shape, position, str(count)], "license": "CC0-1.0"})
                    for suffix, text, slice_name in [
                        ("exact", caption, "composition"),
                        ("paraphrase", f"a plain canvas with {objects} placed to the {position}", "paraphrase"),
                    ]:
                        queries.append({"id": f"{key}-{suffix}", "text": text, "relevant": [key],
                                        "source_id": key, "group": group, "split": split, "slice": slice_name})
            # Broad query intentionally has multiple relevant images.
            queries.append({"id": f"{group}-attribute", "text": f"{color} {shape}",
                            "relevant": [r["id"] for r in records if r["id"].startswith(group + "-")],
                            "source_id": f"{group}-left-1", "group": group, "split": split, "slice": "attribute"})
    write_jsonl(destination / "manifest.jsonl", records)
    write_jsonl(destination / "queries.jsonl", queries)
    (destination / "DATASET_CARD.md").write_text(
        "# Shape diagnostic v1\n\n48 original programmatic images; 108 English queries. Images: CC0-1.0.\n"
        "4 colors × 3 shapes × 2 positions × 2 counts.\n"
        "Dev/test queries are disjoint at the color-shape group level, 6 groups per split.\n"
        "Both splits use the same fixed 48-image gallery. Never tune on test outcomes.\n"
        "This is a small, correlated synthetic diagnostic, not a standard benchmark.\n"
        "No training is performed. Captions are not used to generate image embeddings.\n", encoding="utf-8")
