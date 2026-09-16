from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np

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


def metrics(ranked_ids: list[str], relevant: set[str], retrieve_k: int, serve_k: int) -> dict[str, float]:
    dense = ranked_ids[:retrieve_k]
    served = ranked_ids[:serve_k]
    hit30 = float(bool(relevant & set(dense)))
    hit5 = float(bool(relevant & set(served)))
    reciprocal = 0.0
    dcg = 0.0
    for rank, document_id in enumerate(served, start=1):
        if document_id in relevant:
            if reciprocal == 0:
                reciprocal = 1.0 / rank
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), serve_k) + 1))
    return {"hit_at_retrieve_k": hit30, "hit_at_serve_k": hit5, "mrr_at_serve_k": reciprocal, "ndcg_at_serve_k": dcg / ideal if ideal else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--utility-queries", type=Path, required=True)
    parser.add_argument("--incidents", type=Path, required=True)
    parser.add_argument("--embedder-path", required=True)
    parser.add_argument("--reranker-path", required=True)
    parser.add_argument("--clean-vector-cache", type=Path, required=True)
    parser.add_argument("--poison-vector-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrieve-k", type=int, default=30)
    parser.add_argument("--serve-k", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import CrossEncoder, SentenceTransformer

    corpus = list(read_jsonl(args.corpus))
    clean_ids = [str(item["_id"]) for item in corpus]
    clean_texts = [document_text(item) for item in corpus]
    clean_vectors = np.load(args.clean_vector_cache).astype(np.float32)
    if len(clean_vectors) != len(clean_ids):
        raise ValueError("clean vector cache does not match corpus")
    incidents = json.loads(args.incidents.read_text(encoding="utf-8"))
    poison_rows = []
    for incident in incidents:
        for ordinal, text in select_poison_passages(incident, 5):
            poison_rows.append((incident["question_id"], ordinal, text))
    poison_vectors = np.load(args.poison_vector_cache).astype(np.float32)
    if len(poison_vectors) != len(poison_rows):
        raise ValueError("poison vector cache does not match incidents")
    poison_ids = [f"poison::{owner}::{ordinal}" for owner, ordinal, _ in poison_rows]
    poison_texts = [text for _, _, text in poison_rows]
    poisoned_vectors = np.concatenate([clean_vectors, poison_vectors], axis=0)
    poisoned_ids = clean_ids + poison_ids
    poisoned_texts = clean_texts + poison_texts
    utility_payload = json.loads(args.utility_queries.read_text(encoding="utf-8"))
    utility_queries = utility_payload["queries"]
    embedder_fingerprint = fingerprint_path(args.embedder_path)
    reranker_fingerprint = fingerprint_path(args.reranker_path)
    embedder = SentenceTransformer(args.embedder_path, device=args.device)
    query_vectors = embedder.encode(
        [QUERY_PREFIX + item["question"] for item in utility_queries],
        batch_size=args.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    del embedder
    reranker = CrossEncoder(args.reranker_path, device=args.device, max_length=512)

    def retrieve(question: str, query: np.ndarray, vectors: np.ndarray, ids: list[str], texts: list[str]) -> list[str]:
        dense = exact_top_k(query, vectors, args.retrieve_k)
        scores = np.asarray(reranker.predict(
            [(question, texts[index]) for index in dense],
            batch_size=args.reranker_batch_size,
            show_progress_bar=False,
        )).reshape(-1)
        ranked = [dense[offset] for offset in sorted(range(len(dense)), key=lambda offset: (-float(scores[offset]), dense[offset]))]
        return [ids[index] for index in ranked]

    started = time.perf_counter()
    records = []
    metric_names = ("hit_at_retrieve_k", "hit_at_serve_k", "mrr_at_serve_k", "ndcg_at_serve_k")
    for index, item in enumerate(utility_queries):
        clean_ranked = retrieve(item["question"], query_vectors[index], clean_vectors, clean_ids, clean_texts)
        poisoned_ranked = retrieve(item["question"], query_vectors[index], poisoned_vectors, poisoned_ids, poisoned_texts)
        relevant = set(item["relevant_document_ids"])
        clean_metric = metrics(clean_ranked, relevant, args.retrieve_k, args.serve_k)
        poisoned_metric = metrics(poisoned_ranked, relevant, args.retrieve_k, args.serve_k)
        records.append({
            "query_id": item["query_id"], "question": item["question"], "relevant_document_ids": sorted(relevant),
            "clean": {"ranked_ids": clean_ranked, **clean_metric},
            "poisoned": {"ranked_ids": poisoned_ranked, **poisoned_metric},
            "complete_lineage_repair": {"ranked_ids": clean_ranked, **clean_metric},
        })
    elapsed = time.perf_counter() - started
    summary = {}
    for condition in ("clean", "poisoned", "complete_lineage_repair"):
        summary[condition] = {name: float(np.mean([row[condition][name] for row in records])) for name in metric_names}
    summary["complete_lineage_utility_loss_vs_clean"] = {name: summary["clean"][name] - summary["complete_lineage_repair"][name] for name in metric_names}
    summary["poisoned_change_vs_clean"] = {name: summary["poisoned"][name] - summary["clean"][name] for name in metric_names}
    payload = {
        "tier": "frozen_global_retrieval_utility",
        "paper_evidence": True,
        "limitations": [
            "This suite measures retrieval and reranking utility, not generation accuracy.",
            "Only 170 non-incident NQ queries had a relevant document in the incident-enriched 100k snapshot; 150 were frozen by hash before utility outcomes were computed.",
            "Complete-lineage repair removes only injected documents, so exact equality to the clean baseline is an expected lineage-closure property.",
        ],
        "configuration": {
            "queries": len(records), "clean_documents": len(clean_ids), "injected_documents": len(poison_ids),
            "retrieve_k": args.retrieve_k, "serve_k": args.serve_k,
            "embedder": {"name": EMBEDDER_NAME, "revision": EMBEDDER_REVISION, "fingerprint": embedder_fingerprint},
            "reranker": {"name": RERANKER_NAME, "revision": RERANKER_REVISION, "fingerprint": reranker_fingerprint},
        },
        "inputs": {
            "corpus": {"path": str(args.corpus), "sha256": sha256_file(args.corpus)},
            "utility_queries": {"path": str(args.utility_queries), "sha256": sha256_file(args.utility_queries)},
            "incidents": {"path": str(args.incidents), "sha256": sha256_file(args.incidents)},
            "clean_vector_cache": {"path": str(args.clean_vector_cache), "sha256": sha256_file(args.clean_vector_cache)},
            "poison_vector_cache": {"path": str(args.poison_vector_cache), "sha256": sha256_file(args.poison_vector_cache)},
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "evaluation_seconds": elapsed, "script_sha256": sha256_file(Path(__file__))},
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
