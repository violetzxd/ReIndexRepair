from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from reindexrepair.metrics import clopper_pearson_interval


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    if artifact.get("split_status") != "frozen_before_any_retrieval_reranking_or_generation_evaluation":
        errors.append("split status is not frozen certification")
    for row in rows:
        f2 = row["f2_fresh_exact_rebuild"]
        if not f2.get("materialized_faiss_index"):
            errors.append(f"{row['incident_id']}: F2 index was not materialized")
        if not f2.get("matches_masked_exact_oracle"):
            errors.append(f"{row['incident_id']}: materialized and exact-oracle rankings differ")
        if set(row["f2_deleted_ids"]) & set(f2["served_ids"]):
            errors.append(f"{row['incident_id']}: deleted source remains served")

    visible = [row for row in rows if row["visible_poison_count"] > 0]
    events = sum(
        row["f2_fresh_exact_rebuild"]["own_poison_in_served"] > 0 for row in visible
    )
    expected = {
        "events": events,
        "eligible": len(visible),
        "rate": events / len(visible) if visible else None,
        "two_sided_95_exact_interval": (
            list(clopper_pearson_interval(events, len(visible))) if visible else None
        ),
    }
    if artifact["summary"]["retrieval_level_substitution"] != expected:
        errors.append("stored retrieval-level summary does not match records")
    for model_name in ("embedder", "reranker"):
        fingerprint = artifact["configuration"][model_name].get("fingerprint", {})
        if not fingerprint.get("sha256_tree"):
            errors.append(f"missing {model_name} tree fingerprint")
    if artifact["configuration"].get("generator") is not None:
        fingerprint = artifact["configuration"]["generator"].get("fingerprint", {})
        if not fingerprint.get("sha256_tree"):
            errors.append("missing generator tree fingerprint")

    payload = {
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "passed": not errors,
        "errors": errors,
        "recomputed": {
            "incidents": len(rows),
            "visible_poison_incidents": len(visible),
            "retrieval_level_substitution": expected,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
