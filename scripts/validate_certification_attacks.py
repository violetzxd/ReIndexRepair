from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def contains(haystack: str, needle: str) -> bool:
    return f" {normalize(needle)} " in f" {normalize(haystack)} "


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--attacks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))["records"]
    attacks = json.loads(args.attacks.read_text(encoding="utf-8"))["records"]
    candidate_by_id = {item["query_id"]: item for item in candidates}
    attack_by_id = {item["query_id"]: item for item in attacks}
    if set(candidate_by_id) != set(attack_by_id):
        raise ValueError("candidate and attack query IDs differ")

    rows = []
    for query_id in sorted(candidate_by_id):
        candidate = candidate_by_id[query_id]
        attack = attack_by_id[query_id]
        evidence = "\n".join(
            f"{item.get('title', '')}\n{item.get('text', '')}"
            for item in candidate["relevant_documents"]
        )
        correct_in_evidence = contains(evidence, attack["correct_answer"])
        quote_in_evidence = contains(evidence, attack["correct_evidence_quote"])
        target_in_evidence = contains(evidence, attack["target_answer"])
        target_in_all_attacks = all(
            contains(text, attack["target_answer"]) for text in attack["attack_passages"]
        )
        correct_in_any_attack = any(
            contains(text, attack["correct_answer"]) for text in attack["attack_passages"]
        )
        checks = {
            "correct_answer_in_evidence": correct_in_evidence,
            "correct_quote_in_evidence": quote_in_evidence,
            "target_absent_from_evidence": not target_in_evidence,
            "target_in_all_attack_passages": target_in_all_attacks,
            "correct_absent_from_all_attack_passages": not correct_in_any_attack,
        }
        rows.append(
            {
                "query_id": query_id,
                "passed": all(checks.values()),
                "checks": checks,
            }
        )

    payload = {
        "paper_evidence": False,
        "status": "mechanical_validation_before_any_retrieval_evaluation",
        "inputs": {
            "candidates_sha256": sha256_file(args.candidates),
            "attacks_sha256": sha256_file(args.attacks),
        },
        "summary": {
            "records": len(rows),
            "passed": sum(item["passed"] for item in rows),
            "failed": sum(not item["passed"] for item in rows),
        },
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
