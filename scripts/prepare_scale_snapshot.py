from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path

from prepare_frozen_nq_snapshot import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/nq"))
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=1_000_000)
    parser.add_argument("--hard-negatives-per-query", type=int, default=100)
    parser.add_argument("--seed", default="reindexrepair-scale-sensitivity-v1")
    args = parser.parse_args()
    corpus_path = args.dataset_dir / "corpus.jsonl"
    qrels_path = args.dataset_dir / "qrels" / "test.tsv"
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    target_queries = {item["beir_query_id"] for item in queries}
    required_ids = set()
    with qrels_path.open(encoding="utf-8") as stream:
        header = next(stream)
        if "query-id" not in header:
            raise ValueError("unexpected qrels header")
        for line in stream:
            query_id, corpus_id, score = line.rstrip("\n").split("\t")
            if query_id in target_queries and int(score) > 0:
                required_ids.add(corpus_id)
    retrieval = json.loads(args.retrieval_results.read_text(encoding="utf-8"))
    for query_id in target_queries:
        required_ids.update(list(retrieval[query_id])[: args.hard_negatives_per_query])
    background_needed = args.documents - len(required_ids)
    if background_needed < 0:
        raise ValueError("required documents exceed requested snapshot size")
    heap: list[tuple[int, str]] = []
    found_required = set()
    with corpus_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            corpus_id = str(record["_id"])
            if corpus_id in required_ids:
                found_required.add(corpus_id)
                continue
            rank = int.from_bytes(hashlib.blake2b(f"{args.seed}\0{corpus_id}".encode("utf-8"), digest_size=8).digest(), "big")
            entry = (-rank, corpus_id)
            if len(heap) < background_needed:
                heapq.heappush(heap, entry)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, entry)
    missing = required_ids - found_required
    if missing:
        raise KeyError(f"required corpus IDs missing: {sorted(missing)[:5]}")
    selected_ids = required_ids | {corpus_id for _, corpus_id in heap}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_corpus = args.output_dir / "corpus.jsonl"
    written = 0
    with corpus_path.open(encoding="utf-8") as source, output_corpus.open("w", encoding="utf-8", newline="\n") as target:
        for line in source:
            record = json.loads(line)
            if str(record["_id"]) in selected_ids:
                target.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
    if written != args.documents:
        raise ValueError(f"wrote {written} documents, expected {args.documents}")
    output_queries = args.output_dir / "certification_queries.json"
    output_queries.write_text(json.dumps(queries, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "paper_evidence_candidate": False,
        "status": "posthoc_production_scale_sensitivity_snapshot",
        "seed": args.seed,
        "documents": written,
        "required_relevant_and_hard_negative_documents": len(required_ids),
        "random_background_documents": background_needed,
        "certification_incidents": len(queries),
        "hard_negatives_per_query": args.hard_negatives_per_query,
        "ordering": "original official corpus order",
        "source_sha256": {
            "corpus": sha256_file(corpus_path), "qrels": sha256_file(qrels_path),
            "queries": sha256_file(args.queries), "retrieval_results": sha256_file(args.retrieval_results),
        },
        "output_sha256": {"corpus": sha256_file(output_corpus), "queries": sha256_file(output_queries)},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
