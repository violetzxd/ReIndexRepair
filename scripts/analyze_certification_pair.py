from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar_two_sided(main_only: int, control_only: int) -> float:
    discordant = main_only + control_only
    if discordant == 0:
        return 1.0
    lower = min(main_only, control_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1)) / 2**discordant
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--complete-lineage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    partial = json.loads(args.partial.read_text(encoding="utf-8"))
    complete = json.loads(args.complete_lineage.read_text(encoding="utf-8"))
    partial_by_id = {item["incident_id"]: item for item in partial["records"]}
    complete_by_id = {item["incident_id"]: item for item in complete["records"]}
    if set(partial_by_id) != set(complete_by_id):
        raise ValueError("paired artifacts contain different incidents")
    if partial["configuration"]["f2_delete_policy"] != "visible":
        raise ValueError("partial artifact does not use visible deletion")
    if complete["configuration"]["f2_delete_policy"] != "all_owner":
        raise ValueError("control artifact does not use complete lineage deletion")

    paired = []
    for incident_id in sorted(partial_by_id):
        left = partial_by_id[incident_id]
        right = complete_by_id[incident_id]
        if left["initial"]["answer"] != right["initial"]["answer"]:
            raise ValueError(f"initial answer differs for {incident_id}")
        if left["f0_trace_mask"]["answer"] != right["f0_trace_mask"]["answer"]:
            raise ValueError(f"F0 answer differs for {incident_id}")
        eligible = bool(left["initial_unsafe"] and left["f0_safe"])
        paired.append(
            {
                "incident_id": incident_id,
                "eligible": eligible,
                "visible_poison_count": left["visible_poison_count"],
                "partial_unsafe": bool(left["f2_unsafe"]),
                "complete_lineage_unsafe": bool(right["f2_unsafe"]),
                "retrieval_substitution": bool(
                    left["f2_fresh_exact_rebuild"]["own_poison_in_served"] > 0
                ),
                "partial_answer": left["f2_fresh_exact_rebuild"]["answer"],
                "complete_lineage_answer": right["f2_fresh_exact_rebuild"]["answer"],
                "target_answer": left["target_answer"],
            }
        )
    eligible = [item for item in paired if item["eligible"]]
    main_only = sum(item["partial_unsafe"] and not item["complete_lineage_unsafe"] for item in eligible)
    control_only = sum(not item["partial_unsafe"] and item["complete_lineage_unsafe"] for item in eligible)
    both = sum(item["partial_unsafe"] and item["complete_lineage_unsafe"] for item in eligible)
    neither = len(eligible) - main_only - control_only - both
    differences = np.asarray(
        [int(item["partial_unsafe"]) - int(item["complete_lineage_unsafe"]) for item in eligible],
        dtype=np.float64,
    )
    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty(args.bootstrap_samples, dtype=np.float64)
    for index in range(args.bootstrap_samples):
        bootstrap[index] = rng.choice(differences, size=len(differences), replace=True).mean()

    by_visible_count = {}
    for visible_count in sorted({item["visible_poison_count"] for item in paired}):
        group = [item for item in paired if item["visible_poison_count"] == visible_count]
        by_visible_count[str(visible_count)] = {
            "incidents": len(group),
            "retrieval_substitution_events": sum(item["retrieval_substitution"] for item in group),
            "eligible_answer_level": sum(item["eligible"] for item in group),
            "answer_mirages": sum(item["eligible"] and item["partial_unsafe"] for item in group),
        }

    payload = {
        "paper_evidence": False,
        "status": "lexical_sensitivity_analysis_pending_blinded_semantic_annotation",
        "inputs": {
            "partial_sha256": sha256_file(args.partial),
            "complete_lineage_sha256": sha256_file(args.complete_lineage),
        },
        "paired_answer_level": {
            "eligible_initially_unsafe_and_f0_safe": len(eligible),
            "partial_only_unsafe": main_only,
            "complete_lineage_only_unsafe": control_only,
            "both_unsafe": both,
            "neither_unsafe": neither,
            "partial_rate": sum(item["partial_unsafe"] for item in eligible) / len(eligible),
            "complete_lineage_rate": sum(item["complete_lineage_unsafe"] for item in eligible) / len(eligible),
            "paired_risk_difference": float(differences.mean()),
            "paired_bootstrap_95_interval": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "exact_mcnemar_two_sided_p": exact_mcnemar_two_sided(main_only, control_only),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
        },
        "all_incidents_unconditional": {
            "partial_answer_mirages": sum(item["eligible"] and item["partial_unsafe"] for item in paired),
            "incidents": len(paired),
        },
        "by_visible_poison_count": by_visible_count,
        "paired_records": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "paired_records"}, indent=2))


if __name__ == "__main__":
    main()
