from collections import Counter

from select_subsets import marginal


def test_weighted_sqrt_coverage_has_diminishing_returns():
    item = {"time=night", "class=car"}
    weights = {"time=night": 2.0, "class=car": 1.0}
    empty = Counter()
    already_covered = Counter({"time=night": 3, "class=car": 8})
    assert marginal(item, empty, weights) > marginal(item, already_covered, weights) > 0


def test_modular_target_relevance_preserves_diminishing_returns():
    item = {"time=night"}
    weights = {"time=night": 2.0}
    assert marginal(item, Counter(), weights, 0.1) > marginal(
        item, Counter({"time=night": 5}), weights, 0.1)
