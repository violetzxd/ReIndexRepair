from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_frozen_nq_snapshot import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/nq"))
    parser.add_argument("--snapshot-corpus", type=Path, required=True)
    parser.add_argument("--certification-queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--seed", default="reindexrepair-global-utility-v1")
    args = parser.parse_args()
    snapshot_ids = {
        str(json.loads(line)["_id"])
        for line in args.snapshot_corpus.open(encoding="utf-8")
    }
    excluded = {
        item["beir_query_id"]
        for item in json.loads(args.certification_queries.read_text(encoding="utf-8"))
    }
    query_text = {}
    with (args.dataset_dir / "queries.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            query_text[str(item["_id"])] = item["text"]
    relevant: dict[str, list[str]] = {}
    with (args.dataset_dir / "qrels" / "test.tsv").open(encoding="utf-8") as stream:
        header = next(stream)
        if "query-id" not in header:
            raise ValueError("unexpected qrels header")
        for line in stream:
            query_id, corpus_id, score = line.rstrip("\n").split("\t")
            if int(score) > 0 and corpus_id in snapshot_ids and query_id not in excluded:
                relevant.setdefault(query_id, []).append(corpus_id)
    candidates = [query_id for query_id in relevant if query_id in query_text]
    candidates.sort(key=lambda query_id: hashlib.sha256(f"{args.seed}\0{query_id}".encode("utf-8")).hexdigest())
    selected = candidates[: args.queries]
    if len(selected) < args.queries:
        raise ValueError(f"only {len(selected)} eligible utility queries")
    payload = {
        "paper_evidence_candidate": True,
        "status": "utility_suite_selected_without_running_utility_outcomes",
        "seed": args.seed,
        "queries": [
            {"query_id": query_id, "question": query_text[query_id], "relevant_document_ids": sorted(relevant[query_id])}
            for query_id in selected
        ],
        "inputs": {
            "snapshot_corpus_sha256": sha256_file(args.snapshot_corpus),
            "certification_queries_sha256": sha256_file(args.certification_queries),
            "full_queries_sha256": sha256_file(args.dataset_dir / "queries.jsonl"),
            "qrels_sha256": sha256_file(args.dataset_dir / "qrels" / "test.tsv"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "queries": len(selected), "sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
