from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical(text: str) -> str:
    return " ".join(text.casefold().split())


def selection_score(seed: str, query_id: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{seed}\0{query_id}".encode("utf-8"), digest_size=16).digest(),
        "big",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/nq"))
    parser.add_argument(
        "--development-incidents",
        type=Path,
        default=Path(
            "external/RAG-Responsibility-Attribution/attack_feedback/PRAGB/"
            "k5_m5_e5_gpt-4o-mini.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", default="reindexrepair-certification-v1")
    args = parser.parse_args()

    queries_path = args.dataset_dir / "queries.jsonl"
    corpus_path = args.dataset_dir / "corpus.jsonl"
    qrels_path = args.dataset_dir / "qrels" / "test.tsv"
    development = json.loads(args.development_incidents.read_text(encoding="utf-8"))
    excluded_questions = {canonical(item["question"]) for item in development}

    queries: dict[str, str] = {}
    with queries_path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if canonical(item["text"]) not in excluded_questions:
                queries[str(item["_id"])] = item["text"]

    relevant: dict[str, list[str]] = {}
    with qrels_path.open(encoding="utf-8") as stream:
        header = next(stream)
        if "query-id" not in header:
            raise ValueError("unexpected qrels header")
        for line in stream:
            query_id, corpus_id, score = line.rstrip("\n").split("\t")
            if query_id in queries and int(score) > 0:
                relevant.setdefault(query_id, []).append(corpus_id)

    ranked = sorted(
        relevant,
        key=lambda query_id: (selection_score(args.seed, query_id), query_id),
    )
    selected_ids = ranked[: args.count]
    if len(selected_ids) != args.count:
        raise ValueError("not enough eligible queries")
    wanted_documents = {
        corpus_id for query_id in selected_ids for corpus_id in relevant[query_id]
    }
    documents: dict[str, dict] = {}
    with corpus_path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            corpus_id = str(item["_id"])
            if corpus_id in wanted_documents:
                documents[corpus_id] = item
    missing = wanted_documents - set(documents)
    if missing:
        raise KeyError(f"missing qrel documents: {sorted(missing)[:5]}")

    records = []
    for index, query_id in enumerate(selected_ids):
        records.append(
            {
                "certification_index": index,
                "query_id": query_id,
                "question": queries[query_id],
                "selection_score_hex": f"{selection_score(args.seed, query_id):032x}",
                "relevant_documents": [documents[item] for item in relevant[query_id]],
                "status": "selected_before_attack_generation_or_evaluation",
            }
        )

    payload = {
        "paper_evidence": False,
        "status": "candidate_selection_frozen_before_attack_generation_or_evaluation",
        "seed": args.seed,
        "count": len(records),
        "exclusion": "canonical question overlap with all 100 development incidents",
        "inputs": {
            "queries_sha256": sha256_file(queries_path),
            "corpus_sha256": sha256_file(corpus_path),
            "qrels_sha256": sha256_file(qrels_path),
            "development_incidents_sha256": sha256_file(args.development_incidents),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
