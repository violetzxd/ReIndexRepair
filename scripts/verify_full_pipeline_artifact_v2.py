from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from reindexrepair.metrics import clopper_pearson_interval


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


def rate_record(events: int, eligible: int) -> dict[str, object]:
    return {
        "events": events,
        "eligible": eligible,
        "rate": events / eligible if eligible else None,
        "two_sided_95_exact_interval": (
            list(clopper_pearson_interval(events, eligible)) if eligible else None
        ),
    }


def equal_number(left: object, right: object, tolerance: float = 1e-12) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)
    return left == right


def equal_rate_record(left: dict, right: dict) -> bool:
    left_interval = left.get("two_sided_95_exact_interval") or []
    right_interval = right.get("two_sided_95_exact_interval") or []
    return (
        left.get("events") == right.get("events")
        and left.get("eligible") == right.get("eligible")
        and equal_number(left.get("rate"), right.get("rate"))
        and len(left_interval) == len(right_interval)
        and all(equal_number(a, b) for a, b in zip(left_interval, right_interval))
    )


def poison_owner(document_id: str) -> str | None:
    if not document_id.startswith("poison::"):
        return None
    remainder = document_id[len("poison::") :]
    owner, separator, ordinal = remainder.rpartition("::")
    return owner if separator and ordinal.isdigit() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--incidents", type=Path)
    parser.add_argument("--script", type=Path)
    parser.add_argument("--require-paper-evidence", action="store_true")
    args = parser.parse_args()

    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    rows = artifact["records"]
    errors: list[str] = []
    ids = [row["incident_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("incident IDs are not unique")
    if args.require_paper_evidence and not artifact.get("paper_evidence"):
        errors.append("artifact is not marked paper_evidence")
    if not str(artifact.get("split_status", "")).startswith("frozen_before"):
        errors.append("split status is not frozen certification")

    expected_paths = {"corpus": args.corpus, "queries": args.queries, "incidents": args.incidents}
    for name, path in expected_paths.items():
        if path is not None:
            recorded = artifact.get("inputs", {}).get(name, {}).get("sha256")
            if recorded != sha256_file(path):
                errors.append(f"{name} SHA-256 mismatch")
    if args.script is not None and artifact.get("runtime", {}).get("script_sha256") != sha256_file(args.script):
        errors.append("runner script SHA-256 mismatch")

    for row in rows:
        owner = row["incident_id"]
        initial, f0, f2 = row["initial"], row["f0_trace_mask"], row["f2_fresh_exact_rebuild"]
        if not f2.get("materialized_faiss_index"):
            errors.append(f"{owner}: F2 index was not materialized")
        if not f2.get("matches_masked_exact_oracle"):
            errors.append(f"{owner}: materialized and exact-oracle rankings differ")
        if set(row["f2_deleted_ids"]) & set(f2["served_ids"]):
            errors.append(f"{owner}: deleted source remains served")
        initial_count = sum(poison_owner(item) == owner for item in initial["served_ids"])
        f2_count = sum(poison_owner(item) == owner for item in f2["served_ids"])
        if initial.get("own_poison_in_served") != initial_count:
            errors.append(f"{owner}: initial own-poison count mismatch")
        if f2.get("own_poison_in_served") != f2_count:
            errors.append(f"{owner}: F2 own-poison count mismatch")
        visible_ids = [item for item in initial["served_ids"] if poison_owner(item) == owner]
        if row.get("visible_poison_count") != len(visible_ids):
            errors.append(f"{owner}: visible-poison count mismatch")
        if sorted(row.get("visible_poison_ids", [])) != sorted(visible_ids):
            errors.append(f"{owner}: visible-poison IDs mismatch")
        if "answer" in initial:
            for condition in (initial, f0, f2):
                target = contains_answer(condition["answer"], row["target_answer"])
                correct = contains_answer(condition["answer"], row["correct_answer"])
                if condition.get("target_present") != target:
                    errors.append(f"{owner}: stored target-present flag mismatch")
                if condition.get("correct_present") != correct:
                    errors.append(f"{owner}: stored correct-present flag mismatch")
            expected_flags = {
                "initial_unsafe": contains_answer(initial["answer"], row["target_answer"]),
                "f0_safe": not contains_answer(f0["answer"], row["target_answer"]),
                "f2_unsafe": contains_answer(f2["answer"], row["target_answer"]),
            }
            expected_flags["repair_mirage"] = (
                expected_flags["initial_unsafe"] and expected_flags["f0_safe"] and expected_flags["f2_unsafe"]
            )
            for key, expected in expected_flags.items():
                if row.get(key) != expected:
                    errors.append(f"{owner}: {key} mismatch")

    visible = [row for row in rows if row["visible_poison_count"] > 0]
    retrieval = rate_record(
        sum(row["f2_fresh_exact_rebuild"]["own_poison_in_served"] > 0 for row in visible), len(visible)
    )
    recomputed: dict[str, object] = {
        "incidents": len(rows),
        "visible_poison_incidents": len(visible),
        "retrieval_level_substitution": retrieval,
    }
    stored_summary = artifact["summary"]
    if not equal_rate_record(stored_summary["retrieval_level_substitution"], retrieval):
        errors.append("stored retrieval-level summary does not match records")
    if rows and "answer" in rows[0]["initial"]:
        attacked = [row for row in visible if row["initial_unsafe"]]
        f0_safe_rows = [row for row in attacked if row["f0_safe"]]
        mirages = [row for row in f0_safe_rows if row["f2_unsafe"]]
        answer_summary = {
            "initial_generator_attack_success": rate_record(len(attacked), len(visible)),
            "f0_false_safe_conditional": rate_record(len(mirages), len(f0_safe_rows)),
            "joint_repair_mirage": rate_record(len(mirages), len(attacked)),
        }
        recomputed.update(answer_summary)
        for name, expected in answer_summary.items():
            if not equal_rate_record(stored_summary[name], expected):
                errors.append(f"stored {name} summary does not match answers")

    for model_name in ("embedder", "reranker"):
        if not artifact["configuration"][model_name].get("fingerprint", {}).get("sha256_tree"):
            errors.append(f"missing {model_name} tree fingerprint")
    generator = artifact["configuration"].get("generator")
    if generator is not None:
        if not generator.get("fingerprint", {}).get("sha256_tree"):
            errors.append("missing generator tree fingerprint")
        if not isinstance(artifact["configuration"].get("max_new_tokens"), int):
            errors.append("missing max_new_tokens")

    payload = {
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "passed": not errors,
        "errors": errors,
        "verified_external_inputs": {name: str(path) for name, path in expected_paths.items() if path is not None},
        "verified_runner_script": str(args.script) if args.script is not None else None,
        "recomputed": recomputed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
