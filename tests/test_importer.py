import json
import subprocess
import sys
from pathlib import Path

from clipscope.data import load_manifest, read_jsonl


def test_coco_import_preserves_caption_groups_and_refuses_overwrite(dataset, tmp_path):
    root, _, index, _ = dataset
    images = [{"id": i, "file_name": row["path"]} for i, row in enumerate(index.records[:10])]
    annotations = [{"image_id": i, "caption": f"caption {i} variant {j}"} for i in range(10) for j in range(2)]
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps({"images": images, "annotations": annotations}))
    destination = tmp_path / "imported"
    script = Path(__file__).resolve().parents[1] / "scripts" / "import_coco.py"
    command = [sys.executable, str(script), "--annotations", str(annotation_path), "--images", str(root),
               "--out", str(destination), "--limit", "10"]
    subprocess.run(command, check=True, capture_output=True)
    assert len(load_manifest(destination / "manifest.jsonl")) == 10
    queries = read_jsonl(destination / "queries.jsonl")
    assert len(queries) == 20
    assert len([q for q in queries if q["split"] == "test"]) == 16
    assert all(len({q["split"] for q in queries if q["group"] == key}) == 1 for key in {q["group"] for q in queries})
    assert subprocess.run(command, capture_output=True).returncode != 0
