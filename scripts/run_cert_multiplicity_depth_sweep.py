from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from reindexrepair.metrics import clopper_pearson_interval
from run_full_pipeline_dev import (
    EMBEDDER_NAME,
    EMBEDDER_REVISION,
    QUERY_PREFIX,
    RERANKER_NAME,
    RERANKER_REVISION,
    document_text,
    exact_top_k,
    fingerprint_path,
    read_jsonl,
    select_poison_passages,
    sha256_file,
)


def rate_record(events: int, eligible: int) -> dict[str, object]:
    return {
        "events": events,
        "eligible": eligible,
        "rate": events / eligible if eligible else None,
        "two_sided_95_exact_interval": list(clopper_pearson_interval(events, eligible)) if eligible else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--incidents", type=Path, required=True)
    parser.add_argument("--embedder-path", required=True)
    parser.add_argument("--reranker-path", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multiplicities", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--serve-depths", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--retrieve-k", type=int, default=30)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=94)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if max(args.serve_depths) > args.retrieve_k:
        raise ValueError("serve depth exceeds retrieve-k")
    if min(args.multiplicities) < 1 or max(args.multiplicities) > 5:
        raise ValueError("multiplicity must be between 1 and 5")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from sentence_transformers import CrossEncoder, SentenceTransformer

    incidents = json.loads(args.incidents.read_text(encoding="utf-8"))[: args.limit]
    query_manifest = json.loads(args.queries.read_text(encoding="utf-8"))
    allowed = {item["incident_id"] for item in query_manifest}
    if any(item["question_id"] not in allowed for item in incidents):
        raise ValueError("incident/query manifest mismatch")
    clean_records = list(read_jsonl(args.corpus))
    clean_ids = [str(record["_id"]) for record in clean_records]
    clean_texts = [document_text(record) for record in clean_records]
    poison_rows: list[tuple[str, int, str]] = []
    for incident in incidents:
        for ordinal, text in select_poison_passages(incident, 5):
            poison_rows.append((incident["question_id"], ordinal, text))

    embedder_fingerprint = fingerprint_path(args.embedder_path)
    reranker_fingerprint = fingerprint_path(args.reranker_path)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(json.dumps({
        "corpus": sha256_file(args.corpus),
        "incidents": sha256_file(args.incidents),
        "embedder": embedder_fingerprint["sha256_tree"],
        "query_prefix": QUERY_PREFIX,
        "limit": len(incidents),
    }, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    clean_cache = args.cache_dir / f"clean-{cache_key}.npy"
    poison_cache = args.cache_dir / f"poison-{cache_key}.npy"
    query_cache = args.cache_dir / f"queries-{cache_key}.npy"
    embedder = SentenceTransformer(args.embedder_path, device=args.device)
    cache_hit = clean_cache.exists() and poison_cache.exists() and query_cache.exists()
    if cache_hit:
        clean_vectors = np.load(clean_cache)
        poison_vectors = np.load(poison_cache)
        query_vectors = np.load(query_cache)
    else:
        clean_vectors = embedder.encode(clean_texts, batch_size=args.embedding_batch_size, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        poison_vectors = embedder.encode([row[2] for row in poison_rows], batch_size=args.embedding_batch_size, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        query_vectors = embedder.encode([QUERY_PREFIX + row["question"] for row in incidents], batch_size=args.embedding_batch_size, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        np.save(clean_cache, clean_vectors)
        np.save(poison_cache, poison_vectors)
        np.save(query_cache, query_vectors)
    del embedder
    reranker = CrossEncoder(args.reranker_path, device=args.device, max_length=512)

    records = []
    started = time.perf_counter()
    for multiplicity in args.multiplicities:
        selected_poison_indices = [
            index for index, (_, ordinal, _) in enumerate(poison_rows) if ordinal < multiplicity
        ]
        selected_poison = [poison_rows[index] for index in selected_poison_indices]
        vectors = np.concatenate([clean_vectors, poison_vectors[selected_poison_indices]], axis=0)
        ids = clean_ids + [f"poison::{owner}::{ordinal}" for owner, ordinal, _ in selected_poison]
        texts = clean_texts + [text for _, _, text in selected_poison]
        owners: list[str | None] = [None] * len(clean_texts) + [owner for owner, _, _ in selected_poison]
        for incident_index, incident in enumerate(incidents):
            owner = incident["question_id"]
            question = incident["question"]

            def rerank(candidates: list[int]) -> list[int]:
                scores = np.asarray(reranker.predict(
                    [(question, texts[index]) for index in candidates],
                    batch_size=args.reranker_batch_size,
                    show_progress_bar=False,
                )).reshape(-1)
                return [candidates[offset] for offset in sorted(range(len(candidates)), key=lambda offset: (-float(scores[offset]), candidates[offset]))]

            initial_dense = exact_top_k(query_vectors[incident_index], vectors, args.retrieve_k)
            initial_ranked = rerank(initial_dense)
            all_owner = {index for index, candidate_owner in enumerate(owners) if candidate_owner == owner}
            complete_dense = exact_top_k(query_vectors[incident_index], vectors, args.retrieve_k, excluded=all_owner)
            complete_ranked = rerank(complete_dense)
            for serve_depth in args.serve_depths:
                initial_served = initial_ranked[:serve_depth]
                visible = {index for index in initial_served if owners[index] == owner}
                partial_dense = exact_top_k(query_vectors[incident_index], vectors, args.retrieve_k, excluded=visible)
                partial_ranked = rerank(partial_dense)
                partial_served = partial_ranked[:serve_depth]
                complete_served = complete_ranked[:serve_depth]
                records.append({
                    "incident_id": owner,
                    "multiplicity": multiplicity,
                    "serve_depth": serve_depth,
                    "visible_poison_count": len(visible),
                    "initial_served_ids": [ids[index] for index in initial_served],
                    "partial_deleted_ids": [ids[index] for index in sorted(visible)],
                    "partial_served_ids": [ids[index] for index in partial_served],
                    "complete_deleted_ids": [ids[index] for index in sorted(all_owner)],
                    "complete_served_ids": [ids[index] for index in complete_served],
                    "partial_substitution": bool(visible) and any(owners[index] == owner for index in partial_served),
                    "complete_lineage_residual": any(owners[index] == owner for index in complete_served),
                })
    elapsed = time.perf_counter() - started
    summary = {}
    for multiplicity in args.multiplicities:
        for serve_depth in args.serve_depths:
            group = [row for row in records if row["multiplicity"] == multiplicity and row["serve_depth"] == serve_depth]
            eligible = [row for row in group if row["visible_poison_count"] > 0]
            summary[f"m{multiplicity}_k{serve_depth}"] = {
                "all_incidents": len(group),
                "initially_visible": len(eligible),
                "partial_substitution": rate_record(sum(row["partial_substitution"] for row in eligible), len(eligible)),
                "complete_lineage_residual": rate_record(sum(row["complete_lineage_residual"] for row in eligible), len(eligible)),
            }
    payload = {
        "tier": "frozen_certification_retrieval_sensitivity",
        "paper_evidence": True,
        "split_status": "frozen_before_retrieval_no_outcome_filtering",
        "limitations": [
            "This is a retrieval/reranking sensitivity analysis; answer-level generation is reported only for the predeclared primary M=5, served k=5 setting.",
            "F2 uses an exact counterfactual corpus view; the primary setting separately materializes a new FAISS IndexFlatIP per incident.",
            "The hard-negative-enriched snapshot is a controlled stress benchmark, not an estimate of natural deployment prevalence.",
        ],
        "configuration": {
            "incidents": len(incidents), "clean_documents": len(clean_texts), "retrieve_k": args.retrieve_k,
            "multiplicities": args.multiplicities, "serve_depths": args.serve_depths,
            "poison_selection": "first M of five attack passages frozen before any retrieval evaluation",
            "embedder": {"name": EMBEDDER_NAME, "revision": EMBEDDER_REVISION, "fingerprint": embedder_fingerprint},
            "reranker": {"name": RERANKER_NAME, "revision": RERANKER_REVISION, "fingerprint": reranker_fingerprint},
        },
        "inputs": {
            "corpus": {"path": str(args.corpus), "sha256": sha256_file(args.corpus)},
            "queries": {"path": str(args.queries), "sha256": sha256_file(args.queries)},
            "incidents": {"path": str(args.incidents), "sha256": sha256_file(args.incidents)},
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "embedding_cache_hit": cache_hit, "sweep_seconds": elapsed, "script_sha256": sha256_file(Path(__file__))},
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary, "seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
