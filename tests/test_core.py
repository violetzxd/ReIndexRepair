import numpy as np
import pytest

from reindexrepair.core import (
    FixedTokenChunker,
    HashingEmbedder,
    RepairAction,
    SourceDocument,
    apply_repair,
    build_counterfactual_view,
    build_index,
    retrieve,
)


def test_f1_equals_exact_full_rebuild_for_affected_source():
    documents = [
        SourceDocument("a", "alpha beta gamma delta epsilon zeta"),
        SourceDocument("b", "one two three four five six"),
    ]
    chunker = FixedTokenChunker(chunk_size=4, overlap=1)
    embedder = HashingEmbedder(dim=128)
    base = build_index(documents, chunker, embedder)
    action = RepairAction("repair-a", "a", "replace_span", 1, 3, "safe fact")
    repaired = apply_repair(documents, action)

    f1 = build_counterfactual_view(base, documents, repaired, {"a"}, chunker, embedder)
    f2 = build_index(repaired, chunker, embedder)

    assert f1.source_hash == f2.source_hash
    assert [chunk.chunk_id for chunk in f1.chunks] == [chunk.chunk_id for chunk in f2.chunks]
    for query in ["alpha safe", "three four", "epsilon"]:
        assert [item[0].chunk_id for item in retrieve(query, f1, embedder, top_k=3)] == [
            item[0].chunk_id for item in retrieve(query, f2, embedder, top_k=3)
        ]


def test_counterfactual_view_removes_all_deleted_source_descendants():
    documents = [
        SourceDocument("poison", "query query ATTACK[x]"),
        SourceDocument("clean", "clean evidence"),
    ]
    chunker = FixedTokenChunker(chunk_size=3, overlap=0)
    embedder = HashingEmbedder(dim=64)
    base = build_index(documents, chunker, embedder)
    repaired = apply_repair(documents, RepairAction("drop", "poison", "delete_document"))
    f1 = build_counterfactual_view(base, documents, repaired, {"poison"}, chunker, embedder)
    assert f1.descendant_ids("poison") == set()
    assert all(chunk.source_id == "clean" for chunk in f1.chunks)


def test_retrieval_tie_breaking_is_deterministic():
    documents = [SourceDocument("b", "same"), SourceDocument("a", "same")]
    chunker = FixedTokenChunker(chunk_size=4, overlap=0)
    embedder = HashingEmbedder(dim=64)
    index = build_index(documents, chunker, embedder)
    first = [chunk.chunk_id for chunk, _ in retrieve("same", index, embedder, top_k=2)]
    second = [chunk.chunk_id for chunk, _ in retrieve("same", index, embedder, top_k=2)]
    assert first == second
    assert first == sorted(first)


def test_f1_rejects_undeclared_source_content_change():
    documents = [SourceDocument("a", "one two three"), SourceDocument("b", "safe")]
    chunker = FixedTokenChunker(chunk_size=2, overlap=0)
    embedder = HashingEmbedder(dim=64)
    base = build_index(documents, chunker, embedder)
    repaired = [SourceDocument("a", "one changed three"), documents[1]]

    with pytest.raises(ValueError, match="changed sources omitted"):
        build_counterfactual_view(base, documents, repaired, set(), chunker, embedder)


def test_f1_rejects_mismatched_base_snapshot():
    documents = [SourceDocument("a", "one two three")]
    other_documents = [SourceDocument("a", "different source snapshot")]
    chunker = FixedTokenChunker(chunk_size=2, overlap=0)
    embedder = HashingEmbedder(dim=64)
    base = build_index(other_documents, chunker, embedder)

    with pytest.raises(ValueError, match="base snapshot does not match"):
        build_counterfactual_view(base, documents, documents, set(), chunker, embedder)


def test_f1_rejects_incompatible_pipeline_fingerprint():
    documents = [SourceDocument("a", "one two three four")]
    chunker = FixedTokenChunker(chunk_size=2, overlap=0)
    old_embedder = HashingEmbedder(dim=8)
    new_embedder = HashingEmbedder(dim=16)
    base = build_index(documents, chunker, old_embedder)
    repaired = apply_repair(
        documents, RepairAction("edit", "a", "replace_span", 1, 2, "changed")
    )

    with pytest.raises(ValueError, match="pipeline fingerprint"):
        build_counterfactual_view(
            base, documents, repaired, {"a"}, chunker, new_embedder
        )


def test_f1_rejects_changed_chunk_configuration():
    documents = [SourceDocument("a", "one two three four five")]
    old_chunker = FixedTokenChunker(chunk_size=2, overlap=0)
    new_chunker = FixedTokenChunker(chunk_size=3, overlap=1)
    embedder = HashingEmbedder(dim=16)
    base = build_index(documents, old_chunker, embedder)

    with pytest.raises(ValueError, match="pipeline fingerprint"):
        build_counterfactual_view(
            base, documents, documents, set(), new_chunker, embedder
        )


def test_f1_and_full_rebuild_use_numeric_chunk_order_for_long_sources():
    documents = [
        SourceDocument("a", " ".join(f"token{i}" for i in range(30))),
        SourceDocument("b", " ".join(f"other{i}" for i in range(30))),
    ]
    chunker = FixedTokenChunker(chunk_size=4, overlap=2)
    embedder = HashingEmbedder(dim=128)
    base = build_index(documents, chunker, embedder)
    action = RepairAction("edit", "a", "replace_span", 1, 2, "replacement")
    repaired = apply_repair(documents, action)

    f1 = build_counterfactual_view(base, documents, repaired, {"a"}, chunker, embedder)
    f2 = build_index(repaired, chunker, embedder)

    assert [chunk.chunk_id for chunk in f1.chunks] == [
        chunk.chunk_id for chunk in f2.chunks
    ]
