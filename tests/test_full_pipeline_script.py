from __future__ import annotations

import numpy as np
import pytest

from scripts.run_full_pipeline_dev import (
    contains_answer,
    exact_top_k,
    normalize_answer,
    select_poison_passages,
)


def test_select_poison_passages_uses_labels_not_positions() -> None:
    incident = {
        "context_texts": ["clean zero", "poison one", "clean two", "poison three"],
        "context_labels": [False, True, False, True],
    }
    assert select_poison_passages(incident, 2) == [(1, "poison one"), (3, "poison three")]


def test_select_poison_passages_rejects_short_or_misaligned_labels() -> None:
    with pytest.raises(ValueError):
        select_poison_passages({"context_texts": ["x"], "context_labels": []}, 1)
    with pytest.raises(ValueError):
        select_poison_passages({"context_texts": ["x"], "context_labels": [False]}, 1)


def test_exact_top_k_excludes_sources_and_breaks_ties_by_index() -> None:
    documents = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    assert exact_top_k(query, documents, 2) == [0, 1]
    assert exact_top_k(query, documents, 2, excluded={0}) == [1, 2]


def test_exact_top_k_can_break_boundary_ties_by_stable_id() -> None:
    documents = np.asarray([[1.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float32)
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    assert exact_top_k(query, documents, 1, stable_ids=["z", "m", "a"]) == [2]


def test_answer_containment_is_token_bounded_and_normalized() -> None:
    assert normalize_answer("Elvis-Presley!") == "elvis presley"
    assert contains_answer("It was Elvis Presley.", "Elvis-Presley")
    assert not contains_answer("The answer is 203.", "20")
