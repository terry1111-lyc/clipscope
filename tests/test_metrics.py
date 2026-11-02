import pytest
from clipscope.metrics import retrieval_metrics, cluster_interval, paired_comparison


def test_multiple_relevant_metrics_have_distinct_meanings():
    result = retrieval_metrics(["x", "a", "b", "z"], {"a", "b"})
    assert result["hit@1"] == 0
    assert result["hit@5"] == 1
    assert result["recall@10"] == 1
    assert result["mrr"] == .5
    assert result["map@10"] == pytest.approx((.5 + 2/3) / 2)
    assert result["ndcg@10"] == pytest.approx((1/1.584962500721156 + .5) / (1 + 1/1.584962500721156))


def test_perfect_and_missing_rankings():
    assert all(v == 1 for v in retrieval_metrics(["a", "b"], {"a"}).values())
    assert all(v == 0 for v in retrieval_metrics(["x", "y"], {"a"}).values())
    with pytest.raises(ValueError): retrieval_metrics(["a", "a"], {"a"})
    with pytest.raises(ValueError): retrieval_metrics(["a"], set())


def test_cluster_bootstrap_keeps_correlated_queries_together():
    assert cluster_interval([0, 1], ["same", "same"]) == [.5, .5]
    assert cluster_interval([0, 1], ["a", "b"]) == [0, 1]
    assert cluster_interval([.2, .3, .8], ["a", "b", "c"]) == cluster_interval([.2, .3, .8], ["a", "b", "c"])


def test_paired_report_guards():
    base = {"dataset_sha256": "a", "queries_sha256": "b", "split": "test", "direction": "text-to-image",
            "queries": [{"id": "1", "group": "a", "metrics": {"mrr": .2}}]}
    other = {**base, "queries": [{"id": "1", "group": "a", "metrics": {"mrr": .5}}]}
    assert paired_comparison(base, other)["mrr"]["delta_b_minus_a"] == pytest.approx(.3)
    with pytest.raises(ValueError): paired_comparison(base, {**other, "split": "dev"})
