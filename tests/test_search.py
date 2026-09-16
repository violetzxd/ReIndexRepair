from reindexrepair.core import RepairAction
from reindexrepair.search import CandidateScore, exhaustive_minimum, pareto_frontier


def _score(name: str, edit: int, risk: float, utility: float) -> CandidateScore:
    action = RepairAction(name, "source", "delete_document")
    return CandidateScore(action, edit, risk, utility, "F2")


def test_pareto_frontier_removes_only_dominated_candidates():
    small_risky = _score("small", 1, 0.4, 0.0)
    balanced = _score("balanced", 2, 0.1, 0.1)
    dominated = _score("dominated", 3, 0.2, 0.2)
    safe_costly = _score("safe", 4, 0.0, 0.2)

    frontier = pareto_frontier([dominated, safe_costly, balanced, small_risky])

    assert [item.action.action_id for item in frontier] == ["small", "balanced", "safe"]


def test_exhaustive_minimum_uses_declared_cost_order_among_feasible_actions():
    actions = [
        RepairAction("large", "source", "delete_document"),
        RepairAction("small", "source", "delete_document"),
        RepairAction("unsafe", "source", "delete_document"),
    ]
    scores = {
        "large": _score("large", 8, 0.0, 0.01),
        "small": _score("small", 3, 0.04, 0.02),
        "unsafe": _score("unsafe", 1, 0.20, 0.0),
    }

    result = exhaustive_minimum(
        actions, lambda action: scores[action.action_id], epsilon=0.05, delta=0.02
    )

    assert result is not None
    assert result.action.action_id == "small"


def test_exhaustive_minimum_returns_none_when_no_candidate_is_feasible():
    action = RepairAction("unsafe", "source", "delete_document")
    assert (
        exhaustive_minimum(
            [action], lambda _: _score("unsafe", 1, 0.2, 0.0), epsilon=0.05, delta=0.02
        )
        is None
    )
