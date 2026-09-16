from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--attacks", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--incidents-output", type=Path, required=True)
    parser.add_argument("--queries-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--minimum", type=int, default=80)
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))["records"]
    attacks = json.loads(args.attacks.read_text(encoding="utf-8"))["records"]
    validation = json.loads(args.validation.read_text(encoding="utf-8"))["records"]
    candidate_by_id = {item["query_id"]: item for item in candidates}
    attack_by_id = {item["query_id"]: item for item in attacks}
    passed = {item["query_id"] for item in validation if item["passed"]}
    selected = sorted(
        (candidate_by_id[query_id] for query_id in passed),
        key=lambda item: item["certification_index"],
    )
    if len(selected) < args.minimum:
        raise ValueError(f"only {len(selected)} records passed; minimum is {args.minimum}")

    incidents = []
    queries = []
    for frozen_index, candidate in enumerate(selected):
        attack = attack_by_id[candidate["query_id"]]
        incident_id = f"nq_cert_v1_{frozen_index:03d}"
        incidents.append(
            {
                "question_id": incident_id,
                "beir_query_id": candidate["query_id"],
                "question": candidate["question"],
                "correct_answer": attack["correct_answer"],
                "target_answer": attack["target_answer"],
                "context_texts": attack["attack_passages"],
                "context_labels": [True] * 5,
                "provenance": "frozen DeepSeek-generated attack; mechanically validated before retrieval evaluation",
            }
        )
        queries.append(
            {
                "incident_id": incident_id,
                "beir_query_id": candidate["query_id"],
                "question": candidate["question"],
                "status": "frozen_certification_uninspected_retrieval_outcome",
            }
        )

    args.incidents_output.parent.mkdir(parents=True, exist_ok=True)
    args.queries_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.incidents_output.write_text(
        json.dumps(incidents, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    args.queries_output.write_text(
        json.dumps(queries, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "paper_evidence_candidate": True,
        "status": "frozen_before_any_retrieval_reranking_or_generation_evaluation",
        "mechanical_selection_only": True,
        "selection_rule": "all records passing every predeclared lexical evidence/attack validation check",
        "records": len(incidents),
        "inputs": {
            "candidates_sha256": sha256_file(args.candidates),
            "attacks_sha256": sha256_file(args.attacks),
            "validation_sha256": sha256_file(args.validation),
        },
        "outputs": {
            "incidents_sha256": sha256_file(args.incidents_output),
            "queries_sha256": sha256_file(args.queries_output),
        },
    }
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
