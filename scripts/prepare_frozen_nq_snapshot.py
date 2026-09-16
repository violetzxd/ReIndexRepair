from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/nq"))
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument(
        "--retrieval-results",
        type=Path,
        default=Path("external/PoisonedRAG/results/beir_results/nq-contriever.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=100_000)
    parser.add_argument("--hard-negatives-per-query", type=int, default=100)
    parser.add_argument("--seed", default="reindexrepair-certification-corpus-v1")
    args = parser.parse_args()

    corpus_path = args.dataset_dir / "corpus.jsonl"
    qrels_path = args.dataset_dir / "qrels" / "test.tsv"
    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    target_queries = {item["beir_query_id"] for item in queries}

    required_ids: set[str] = set()
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
    if len(required_ids) > args.documents:
        raise ValueError("required documents exceed snapshot size")

    keep_background = args.documents - len(required_ids)
    heap: list[tuple[int, str, dict]] = []
    required_records: dict[str, dict] = {}
    with corpus_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            corpus_id = str(record["_id"])
            if corpus_id in required_ids:
                required_records[corpus_id] = record
                continue
            rank = int.from_bytes(
                hashlib.blake2b(
                    f"{args.seed}\0{corpus_id}".encode("utf-8"), digest_size=8
                ).digest(),
                "big",
            )
            entry = (-rank, corpus_id, record)
            if len(heap) < keep_background:
                heapq.heappush(heap, entry)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, entry)
    missing = required_ids - set(required_records)
    if missing:
        raise KeyError(f"required corpus IDs missing: {sorted(missing)[:5]}")

    selected = list(required_records.values()) + [entry[2] for entry in heap]
    selected.sort(key=lambda item: item["_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_corpus = args.output_dir / "corpus.jsonl"
    with output_corpus.open("w", encoding="utf-8", newline="\n") as stream:
        for item in selected:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    output_queries = args.output_dir / "certification_queries.json"
    output_queries.write_text(
        json.dumps(queries, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "paper_evidence_candidate": True,
        "status": "corpus_snapshot_frozen_before_retrieval_evaluation",
        "seed": args.seed,
        "documents": len(selected),
        "required_relevant_and_hard_negative_documents": len(required_ids),
        "random_background_documents": keep_background,
        "certification_incidents": len(queries),
        "hard_negatives_per_query": args.hard_negatives_per_query,
        "source_sha256": {
            "corpus": sha256_file(corpus_path),
            "qrels": sha256_file(qrels_path),
            "queries": sha256_file(args.queries),
            "retrieval_results": sha256_file(args.retrieval_results),
        },
        "output_sha256": {
            "corpus": sha256_file(output_corpus),
            "queries": sha256_file(output_queries),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
