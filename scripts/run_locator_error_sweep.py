from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def utility_metrics(ranked_ids: list[str], relevant: set[str], retrieve_k: int, serve_k: int) -> dict[str, float]:
    dense = ranked_ids[:retrieve_k]
    served = ranked_ids[:serve_k]
    reciprocal = 0.0
    dcg = 0.0
    for rank, document_id in enumerate(served, start=1):
        if document_id in relevant:
            if reciprocal == 0.0:
                reciprocal = 1.0 / rank
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), serve_k) + 1))
    return {
        "hit_at_retrieve_k": float(bool(relevant & set(dense))),
        "hit_at_serve_k": float(bool(relevant & set(served))),
        "mrr_at_serve_k": reciprocal,
        "ndcg_at_serve_k": dcg / ideal if ideal else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--utility-queries", type=Path, required=True)
    parser.add_argument("--incidents", type=Path, required=True)
    parser.add_argument("--embedder-path", required=True)
    parser.add_argument("--reranker-path", required=True)
    parser.add_argument("--clean-vector-cache", type=Path, required=True)
    parser.add_argument("--poison-vector-cache", type=Path, required=True)
    parser.add_argument("--attack-query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recall-levels", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--false-positive-counts", type=int, nargs="+", default=[0, 10, 100, 1000])
    parser.add_argument("--retrieve-k", type=int, default=30)
    parser.add_argument("--serve-k", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--seed", default="reindexrepair-locator-error-v1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if any(level < 0.0 or level > 1.0 for level in args.recall_levels):
        raise ValueError("recall levels must lie in [0, 1]")
    if any(count < 0 for count in args.false_positive_counts):
        raise ValueError("false-positive counts must be nonnegative")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    corpus = list(read_jsonl(args.corpus))
    clean_ids = [str(item["_id"]) for item in corpus]
    clean_texts = [document_text(item) for item in corpus]
    incidents = json.loads(args.incidents.read_text(encoding="utf-8"))
    utility_payload = json.loads(args.utility_queries.read_text(encoding="utf-8"))
    utility_queries = utility_payload["queries"]
    poison_rows: list[tuple[str, int, str]] = []
    for incident in incidents:
        for ordinal, text in select_poison_passages(incident, 5):
            poison_rows.append((incident["question_id"], ordinal, text))

    clean_vectors = np.load(args.clean_vector_cache).astype(np.float32)
    poison_vectors = np.load(args.poison_vector_cache).astype(np.float32)
    attack_queries = np.load(args.attack_query_cache).astype(np.float32)
    if len(clean_vectors) != len(clean_ids):
        raise ValueError("clean vector cache does not match corpus")
    if len(poison_vectors) != len(poison_rows):
        raise ValueError("poison vector cache does not match incidents")
    if len(attack_queries) != len(incidents):
        raise ValueError("attack query cache does not match incidents")

    poison_ids = [f"poison::{owner}::{ordinal}" for owner, ordinal, _ in poison_rows]
    poison_texts = [text for _, _, text in poison_rows]
    ids = clean_ids + poison_ids
    texts = clean_texts + poison_texts
    owners: list[str | None] = [None] * len(clean_ids) + [owner for owner, _, _ in poison_rows]
    ordinals: list[int | None] = [None] * len(clean_ids) + [ordinal for _, ordinal, _ in poison_rows]
    vectors = np.concatenate([clean_vectors, poison_vectors], axis=0)

    embedder_fingerprint = fingerprint_path(args.embedder_path)
    reranker_fingerprint = fingerprint_path(args.reranker_path)
    embedder = SentenceTransformer(args.embedder_path, device=args.device)
    utility_vectors = embedder.encode(
        [QUERY_PREFIX + item["question"] for item in utility_queries],
        batch_size=args.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    del embedder
    all_queries = np.concatenate([attack_queries, utility_vectors], axis=0)

    maximum_exclusions = len(poison_rows) + max(args.false_positive_counts)
    candidate_cap = min(len(ids), args.retrieve_k + maximum_exclusions)
    with torch.inference_mode():
        query_tensor = torch.from_numpy(all_queries).to(args.device)
        vector_tensor = torch.from_numpy(vectors).to(args.device)
        score_matrix = (query_tensor @ vector_tensor.T).cpu().numpy()
    del query_tensor, vector_tensor
    torch.cuda.empty_cache()
    candidate_lists = []
    for scores in score_matrix:
        selected = np.argpartition(scores, -candidate_cap)[-candidate_cap:]
        selected = sorted(selected.tolist(), key=lambda index: (-float(scores[index]), index))
        candidate_lists.append(selected)
    del score_matrix

    false_positive_order = sorted(
        range(len(clean_ids)),
        key=lambda index: hashlib.sha256(f"{args.seed}\0{clean_ids[index]}".encode("utf-8")).hexdigest(),
    )
    reranker = CrossEncoder(args.reranker_path, device=args.device, max_length=512)
    rerank_cache: dict[tuple[int, ...], list[int]] = {}

    def retrieve(query_index: int, question: str, excluded: set[int]) -> list[int]:
        dense = [index for index in candidate_lists[query_index] if index not in excluded][: args.retrieve_k]
        if len(dense) != args.retrieve_k:
            raise RuntimeError("candidate cap did not survive the maximum exclusion set")
        key = tuple(dense)
        if key not in rerank_cache:
            scores = np.asarray(reranker.predict(
                [(question, texts[index]) for index in dense],
                batch_size=args.reranker_batch_size,
                show_progress_bar=False,
            )).reshape(-1)
            rerank_cache[key] = [dense[offset] for offset in sorted(
                range(len(dense)), key=lambda offset: (-float(scores[offset]), dense[offset])
            )]
        return rerank_cache[key]

    started = time.perf_counter()
    records = []
    initial_risky: set[str] | None = None
    metric_names = ("hit_at_retrieve_k", "hit_at_serve_k", "mrr_at_serve_k", "ndcg_at_serve_k")
    for recall_level in sorted(args.recall_levels):
        detected_per_owner = int(round(5 * recall_level))
        detected_poison = {
            index for index, (owner, ordinal) in enumerate(zip(owners, ordinals))
            if owner is not None and ordinal is not None and ordinal < detected_per_owner
        }
        for false_positive_count in sorted(args.false_positive_counts):
            false_positives = set(false_positive_order[:false_positive_count])
            excluded = detected_poison | false_positives
            attack_rows = []
            for incident_index, incident in enumerate(incidents):
                owner = incident["question_id"]
                ranked = retrieve(incident_index, incident["question"], excluded)
                served = ranked[: args.serve_k]
                residual = any(owners[index] == owner for index in served)
                attack_rows.append({
                    "incident_id": owner,
                    "residual_same_owner_poison_at_serve_k": residual,
                    "served_ids": [ids[index] for index in served],
                })
            if recall_level == 0.0 and false_positive_count == 0:
                initial_risky = {row["incident_id"] for row in attack_rows if row["residual_same_owner_poison_at_serve_k"]}
            if initial_risky is None:
                raise RuntimeError("the sweep must include recall=0 and false-positive-count=0")

            utility_rows = []
            for offset, item in enumerate(utility_queries):
                query_index = len(incidents) + offset
                ranked = retrieve(query_index, item["question"], excluded)
                ranked_ids = [ids[index] for index in ranked]
                utility_rows.append({
                    "query_id": item["query_id"],
                    **utility_metrics(ranked_ids, set(item["relevant_document_ids"]), args.retrieve_k, args.serve_k),
                })
            condition = {
                "locator_recall": detected_per_owner / 5,
                "detected_poison_per_owner": detected_per_owner,
                "false_positive_documents": false_positive_count,
                "false_positive_rate": false_positive_count / len(clean_ids),
                "attack": {
                    "all_incidents": rate_record(sum(row["residual_same_owner_poison_at_serve_k"] for row in attack_rows), len(attack_rows)),
                    "initially_risky_incidents": rate_record(
                        sum(row["residual_same_owner_poison_at_serve_k"] for row in attack_rows if row["incident_id"] in initial_risky),
                        len(initial_risky),
                    ),
                },
                "utility": {name: float(np.mean([row[name] for row in utility_rows])) for name in metric_names},
                "attack_records": attack_rows,
                "utility_records": utility_rows,
            }
            records.append(condition)
            print(json.dumps({k: condition[k] for k in ("locator_recall", "false_positive_documents", "attack", "utility")}, sort_keys=True))

    clean_reference = next(row for row in records if row["locator_recall"] == 1.0 and row["false_positive_documents"] == 0)["utility"]
    for row in records:
        row["utility_loss_vs_clean_reference"] = {name: clean_reference[name] - row["utility"][name] for name in metric_names}
    elapsed = time.perf_counter() - started
    payload = {
        "tier": "frozen_posthoc_locator_error_sensitivity",
        "paper_evidence": True,
        "limitations": [
            "Locator errors are injected deterministically rather than produced by a specific forensic locator.",
            "Attack risk is the presence of a same-owner poison source in served top-k; generator-level outcomes are reported only in the primary paired experiment.",
            "The 100k corpus is hard-negative enriched and is not an estimate of natural deployment prevalence.",
        ],
        "configuration": {
            "incidents": len(incidents), "utility_queries": len(utility_queries), "clean_documents": len(clean_ids),
            "poison_documents": len(poison_rows), "poison_documents_per_incident": 5,
            "retrieve_k": args.retrieve_k, "serve_k": args.serve_k,
            "recall_levels": sorted({row["locator_recall"] for row in records}),
            "false_positive_counts": sorted(args.false_positive_counts), "false_positive_seed": args.seed,
            "candidate_cap": candidate_cap,
            "embedder": {"name": EMBEDDER_NAME, "revision": EMBEDDER_REVISION, "fingerprint": embedder_fingerprint},
            "reranker": {"name": RERANKER_NAME, "revision": RERANKER_REVISION, "fingerprint": reranker_fingerprint},
        },
        "inputs": {
            "corpus": {"path": str(args.corpus), "sha256": sha256_file(args.corpus)},
            "utility_queries": {"path": str(args.utility_queries), "sha256": sha256_file(args.utility_queries)},
            "incidents": {"path": str(args.incidents), "sha256": sha256_file(args.incidents)},
            "clean_vector_cache": {"path": str(args.clean_vector_cache), "sha256": sha256_file(args.clean_vector_cache)},
            "poison_vector_cache": {"path": str(args.poison_vector_cache), "sha256": sha256_file(args.poison_vector_cache)},
            "attack_query_cache": {"path": str(args.attack_query_cache), "sha256": sha256_file(args.attack_query_cache)},
        },
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "seconds": elapsed, "unique_rerank_candidate_sets": len(rerank_cache), "script_sha256": sha256_file(Path(__file__)),
        },
        "summary": {
            "initially_risky_incidents": len(initial_risky),
            "clean_reference_utility": clean_reference,
            "conditions": [{k: row[k] for k in (
                "locator_recall", "detected_poison_per_owner", "false_positive_documents", "false_positive_rate",
                "attack", "utility", "utility_loss_vs_clean_reference"
            )} for row in records],
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runtime_seconds": elapsed, "initially_risky_incidents": len(initial_risky)}, indent=2))


if __name__ == "__main__":
    main()
