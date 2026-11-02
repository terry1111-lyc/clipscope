import pytest
from clipscope.data import read_jsonl, write_jsonl
from clipscope.evaluation import evaluate, write_report


def test_evaluation_export_and_split(dataset, tmp_path):
    root, encoder, index, _ = dataset
    report = evaluate(index, encoder, root / "queries.jsonl")
    assert report["count"] == 54
    assert report["bootstrap_groups"] == 6
    assert set(report["slices"]) == {"composition", "paraphrase", "attribute"}
    assert len(report["queries"][0]["ranking"]) == 48
    write_report(report, tmp_path / "report.json")
    assert (tmp_path / "report.md").exists()
    reverse = evaluate(index, encoder, root / "queries.jsonl", direction="image-to-text")
    assert reverse["count"] == 24


def test_split_leakage_and_unknown_relevance_rejected(dataset):
    root, encoder, index, _ = dataset
    rows = read_jsonl(root / "queries.jsonl")
    rows[0]["split"] = "test" if rows[1]["split"] == "dev" else "dev"
    write_jsonl(root / "leak.jsonl", rows)
    with pytest.raises(ValueError, match="crosses"): evaluate(index, encoder, root / "leak.jsonl")
    rows = read_jsonl(root / "queries.jsonl"); rows[0]["relevant"] = ["missing"]
    write_jsonl(root / "bad.jsonl", rows)
    with pytest.raises(ValueError, match="relevant"): evaluate(index, encoder, root / "bad.jsonl")


def test_same_image_cannot_be_misrepresented_as_independent_groups(dataset):
    root, encoder, index, _ = dataset
    rows = read_jsonl(root / "queries.jsonl")
    rows[1]["group"] = "separate-but-correlated"
    write_jsonl(root / "groups.jsonl", rows)
    with pytest.raises(ValueError, match="same source"):
        evaluate(index, encoder, root / "groups.jsonl")
