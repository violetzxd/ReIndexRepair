from __future__ import annotations

import pytest

from scripts.finalize_human_semantic_analysis import (
    exact_mcnemar,
    holm_adjusted,
    merge_labels,
    primary_summary,
    repair_summary,
    validate_conditions,
)


def test_merge_labels_uses_consensus_and_only_adjudicates_required_fields() -> None:
    agreement = {
        "records": [
            {
                "item_id": "case|initial",
                "needs_adjudication_fields": ["target_adopted"],
                "consensus_labels": {
                    "target_adopted": None,
                    "correct_adopted": "Yes",
                    "complete": "No",
                },
            }
        ]
    }
    adjudication = {
        "records": [
            {
                "item_id": "case|initial",
                "labels": {
                    "target_adopted": "No",
                    "correct_adopted": "No",
                    "complete": "Yes",
                },
            }
        ]
    }
    assert merge_labels(agreement, adjudication) == {
        "case|initial": {"target_adopted": False, "correct_adopted": True, "complete": False}
    }


def test_merge_labels_rejects_incomplete_adjudication_coverage() -> None:
    agreement = {
        "records": [
            {
                "item_id": "case|initial",
                "needs_adjudication_fields": ["target_adopted"],
                "consensus_labels": {"target_adopted": None, "correct_adopted": "Yes", "complete": "Yes"},
            }
        ]
    }
    with pytest.raises(ValueError, match="coverage mismatch"):
        merge_labels(agreement, None)


def test_incomplete_endpoint_sensitivity_is_directionally_extreme() -> None:
    def label(target: bool, complete: bool = True) -> dict[str, bool]:
        return {"target_adopted": target, "correct_adopted": not target, "complete": complete}

    rows = {
        "one": {
            "initial": label(True),
            "f0": label(False),
            "partial": label(True, complete=False),
            "complete": label(False, complete=False),
        },
        "two": {
            "initial": label(True),
            "f0": label(False),
            "partial": label(False),
            "complete": label(False),
        },
    }
    adjudicated = primary_summary(rows, "partial", "complete", samples=200, seed=7)
    least = primary_summary(rows, "partial", "complete", samples=200, seed=7, sensitivity="least_favorable")
    most = primary_summary(rows, "partial", "complete", samples=200, seed=7, sensitivity="most_favorable")
    assert least["risk_difference_partial_minus_complete"] == -0.5
    assert adjudicated["risk_difference_partial_minus_complete"] == 0.5
    assert most["risk_difference_partial_minus_complete"] == 0.5
    assert least["eligible"] == adjudicated["eligible"] == most["eligible"] == 2


def test_exact_tests_and_holm_are_deterministic() -> None:
    assert exact_mcnemar(10, 0) == pytest.approx(0.001953125)
    assert holm_adjusted({"substitution": 0.001953125, "boundary": 0.03125}) == {
        "substitution": pytest.approx(0.00390625),
        "boundary": pytest.approx(0.03125),
    }


def test_validate_conditions_rejects_missing_condition() -> None:
    with pytest.raises(ValueError, match="expected conditions"):
        validate_conditions({"one": {"initial": {}}}, {"initial", "f0"}, "substitution", 1)


def test_repair_oracle_rejects_incomplete_cheap_action() -> None:
    def semantic(target: bool, correct: bool, complete: bool) -> dict[str, bool]:
        return {"target_adopted": target, "correct_adopted": correct, "complete": complete}

    repair_rows = {}
    artifact_rows = []
    for index in range(30):
        incident = f"case-{index:02d}"
        repair_rows[incident] = {
            "initial": semantic(True, False, True),
            "actions": {
                "cheap": {
                    "f0": semantic(False, True, False),
                    "f2": semantic(False, True, False),
                },
                "document_quarantine": {
                    "f0": semantic(False, True, True),
                    "f2": semantic(False, True, True),
                },
            },
        }
        artifact_rows.append({
            "incident_id": incident,
            "actions": {
                "cheap": {
                    "edit_tokens": 1,
                    "f1_exact_view": {"matches_physical_f2": True, "answer": "truncated", "dense_ids": ["d1"], "served_ids": ["d1"]},
                    "f2_physical_rebuild": {"answer": "truncated", "dense_ids": ["d1"], "served_ids": ["d1"]},
                },
                "document_quarantine": {
                    "edit_tokens": 10,
                    "f1_exact_view": {"matches_physical_f2": True, "answer": "complete", "dense_ids": ["d2"], "served_ids": ["d2"]},
                    "f2_physical_rebuild": {"answer": "complete", "dense_ids": ["d2"], "served_ids": ["d2"]},
                },
            },
            "selection": {"finalist_recall": True},
        })
    result = repair_summary(repair_rows, {"records": artifact_rows})
    assert result["reindexrepair_feasible"] == 30
    assert result["selected_action_counts"] == {"document_quarantine": 30}
    assert result["fixed_trace_selected"] == 30
    assert result["document_quarantine_valid_correct"] == 30
    assert result["f2_materializations_per_incident"] == 2
