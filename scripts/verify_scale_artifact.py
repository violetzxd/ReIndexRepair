from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from reindexrepair.metrics import clopper_pearson_interval


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def owner(document_id: str) -> str | None:
    if not document_id.startswith("poison::"):
        return None
    value, separator, ordinal = document_id[len("poison::") :].rpartition("::")
    return value if separator and ordinal.isdigit() else None


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
    errors = []
    paths = {"corpus": args.corpus, "queries": args.queries, "incidents": args.incidents}
    for name, path in paths.items():
        if artifact["inputs"][name]["sha256"] != sha256_file(path):
            errors.append(f"{name} SHA-256 mismatch")
    if artifact["runtime"]["script_sha256"] != sha256_file(args.script):
        errors.append("runner script SHA-256 mismatch")
    rows = artifact["records"]
    if len({row["incident_id"] for row in rows}) != len(rows):
        errors.append("incident IDs are not unique")
    visible = []
    events = 0
    for row in rows:
        incident_id = row["incident_id"]
        initial_poison = [item for item in row["initial"]["served_ids"] if owner(item) == incident_id]
        if len(initial_poison) != row["visible_poison_count"] or sorted(initial_poison) != sorted(row["visible_poison_ids"]):
            errors.append(f"{incident_id}: visible poison metadata mismatch")
        if sorted(row["f2_deleted_ids"]) != sorted(initial_poison):
            errors.append(f"{incident_id}: deletion set mismatch")
        if set(row["f2_deleted_ids"]) & set(row["f2_fresh_exact_rebuild"]["served_ids"]):
            errors.append(f"{incident_id}: deleted source remains served")
        residual = sum(owner(item) == incident_id for item in row["f2_fresh_exact_rebuild"]["served_ids"])
        if residual != row["f2_fresh_exact_rebuild"]["own_poison_in_served"]:
            errors.append(f"{incident_id}: residual poison count mismatch")
        if not row["f2_fresh_exact_rebuild"].get("matches_masked_exact_oracle"):
            errors.append(f"{incident_id}: exact masked-view check failed")
        if initial_poison:
            visible.append(row)
            events += residual > 0
    summary = artifact["summary"]["retrieval_level_substitution"]
    expected_interval = list(clopper_pearson_interval(events, len(visible)))
    if summary["events"] != events or summary["eligible"] != len(visible):
        errors.append("summary event counts mismatch")
    if any(abs(left - right) > 1e-12 for left, right in zip(summary["two_sided_95_exact_interval"], expected_interval)):
        errors.append("summary exact interval mismatch")
    payload = {
        "input": str(args.input), "input_sha256": sha256_file(args.input),
        "passed": not errors, "errors": errors, "verified_records": len(rows),
        "recomputed": {"visible": len(visible), "events": events, "two_sided_95_exact_interval": expected_interval},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
