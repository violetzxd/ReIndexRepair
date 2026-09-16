from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_answer(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def contains_answer(response: str, answer: str) -> bool:
    normalized_response = f" {normalize_answer(response)} "
    normalized_answer = normalize_answer(answer)
    return bool(normalized_answer) and f" {normalized_answer} " in normalized_response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--incidents", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    args = parser.parse_args()

    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    errors: list[str] = []
    expected_inputs = {
        "corpus_sha256": sha256_file(args.corpus),
        "queries_sha256": sha256_file(args.queries),
        "incidents_sha256": sha256_file(args.incidents),
    }
    for key, expected in expected_inputs.items():
        if artifact.get("inputs", {}).get(key) != expected:
            errors.append(f"{key} mismatch")
    if artifact.get("runtime", {}).get("script_sha256") != sha256_file(args.script):
        errors.append("runner script SHA-256 mismatch")
    rows = artifact.get("records", [])
    if len({row["incident_id"] for row in rows}) != len(rows):
        errors.append("incident IDs are not unique")
    for row in rows:
        owner = row["incident_id"]
        for condition_name in (
            "initial", "f0_trace_mask", "f2_rechunk_rebuild", "complete_lineage_quarantine"
        ):
            condition = row[condition_name]
            target = contains_answer(condition["answer"], row["target_answer"])
            correct = contains_answer(condition["answer"], row["correct_answer"])
            if condition.get("target_present") != target:
                errors.append(f"{owner}: {condition_name} target flag mismatch")
            if condition.get("correct_present") != correct:
                errors.append(f"{owner}: {condition_name} correct flag mismatch")
        expected_flags = {
            "initial_unsafe": row["initial"]["target_present"],
            "f0_safe": not row["f0_trace_mask"]["target_present"],
            "f2_unsafe": row["f2_rechunk_rebuild"]["target_present"],
            "complete_lineage_unsafe": row["complete_lineage_quarantine"]["target_present"],
        }
        expected_flags["repair_mirage"] = (
            expected_flags["initial_unsafe"] and expected_flags["f0_safe"] and expected_flags["f2_unsafe"]
        )
        for key, expected in expected_flags.items():
            if row.get(key) != expected:
                errors.append(f"{owner}: {key} mismatch")
        expected_migration = (
            "boundary::pre::0" in row["initial"]["served_ids"]
            and "boundary::pre::1" not in row["initial"]["served_ids"]
            and "boundary::post::0" in row["f2_rechunk_rebuild"]["served_ids"]
        )
        if row.get("boundary_retrieval_migration") != expected_migration:
            errors.append(f"{owner}: boundary migration flag mismatch")
        control_ids = row["complete_lineage_quarantine"]["dense_ids"] + row["complete_lineage_quarantine"]["served_ids"]
        if any(item.startswith("boundary::") for item in control_ids):
            errors.append(f"{owner}: quarantined source remains accessible")
    for model_name in ("embedder", "reranker", "generator"):
        if not artifact.get("configuration", {}).get(model_name, {}).get("fingerprint", {}).get("sha256_tree"):
            errors.append(f"missing {model_name} tree fingerprint")
    payload = {
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "passed": not errors,
        "errors": errors,
        "verified_records": len(rows),
        "verified_external_inputs": expected_inputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
