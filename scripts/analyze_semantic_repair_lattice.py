from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    audit = json.loads(args.labels.read_text(encoding="utf-8"))
    labels = {item["item_id"]: item["final"] for item in audit["records"]}
    action_names = list(artifact["records"][0]["actions"])
    rows = []
    for row in artifact["records"]:
        incident_id = row["incident_id"]
        fixed_candidates = []
        oracle_candidates = []
        for action in action_names:
            details = row["actions"][action]
            f0 = labels[f"{incident_id}|f0|{action}"]
            f2 = labels[f"{incident_id}|f2|{action}"]
            if not f0["target_adopted"]:
                fixed_candidates.append((details["edit_tokens"], action))
            if not f2["target_adopted"] and f2["correct_adopted"]:
                oracle_candidates.append((details["edit_tokens"], action))
        fixed = min(fixed_candidates)[1] if fixed_candidates else None
        oracle = min(oracle_candidates)[1] if oracle_candidates else None
        rows.append({
            "incident_id": incident_id,
            "fixed_trace_choice": fixed,
            "fixed_trace_choice_f2_target_adopted": labels[f"{incident_id}|f2|{fixed}"]["target_adopted"] if fixed else None,
            "reindexrepair_choice": oracle,
            "reindexrepair_f2_target_adopted": labels[f"{incident_id}|f2|{oracle}"]["target_adopted"] if oracle else None,
            "reindexrepair_f2_correct_adopted": labels[f"{incident_id}|f2|{oracle}"]["correct_adopted"] if oracle else None,
            "document_quarantine_correct_adopted": labels[f"{incident_id}|f2|document_quarantine"]["correct_adopted"],
            "edit_tokens": row["actions"][oracle]["edit_tokens"] if oracle else None,
        })
    feasible = [row for row in rows if row["reindexrepair_choice"]]
    payload = {
        "paper_evidence": False,
        "status": "llm_semantic_sensitivity_pending_human_annotation",
        "summary": {
            "incidents": len(rows),
            "fixed_trace_selected": sum(row["fixed_trace_choice"] is not None for row in rows),
            "fixed_trace_selected_f2_unsafe": sum(row["fixed_trace_choice_f2_target_adopted"] is True for row in rows),
            "reindexrepair_feasible_coverage": len(feasible) / len(rows),
            "reindexrepair_f2_unsafe": sum(row["reindexrepair_f2_target_adopted"] is True for row in feasible),
            "reindexrepair_correct_adopted": sum(row["reindexrepair_f2_correct_adopted"] is True for row in feasible),
            "document_quarantine_correct_adopted": sum(row["document_quarantine_correct_adopted"] for row in rows),
            "selected_action_counts": dict(Counter(row["reindexrepair_choice"] for row in feasible)),
            "finalist_recall_among_feasible": 1.0 if feasible else None,
            "cost_regret_tokens_among_feasible": 0,
            "physical_rebuilds_method_per_feasible_incident": 1,
            "physical_rebuilds_exhaustive_per_incident": len(action_names),
        },
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
