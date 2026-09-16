from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reindexrepair.metrics import clopper_pearson_interval


def rate(events: int, eligible: int) -> dict[str, object]:
    return {
        "events": events,
        "eligible": eligible,
        "rate": events / eligible,
        "two_sided_95_exact_interval": list(clopper_pearson_interval(events, eligible)),
    }


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    from math import comb
    lower = min(left_only, right_only)
    return min(1.0, 2.0 * sum(comb(discordant, index) for index in range(lower + 1)) / (2**discordant))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-analysis", type=Path, required=True)
    parser.add_argument("--boundary-audit", type=Path, required=True)
    parser.add_argument("--utility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    main = json.loads(args.main_analysis.read_text(encoding="utf-8"))
    boundary = json.loads(args.boundary_audit.read_text(encoding="utf-8"))
    utility = json.loads(args.utility.read_text(encoding="utf-8"))
    main_p = float(main["paired_primary"]["exact_mcnemar_two_sided_p"])
    paired = boundary["summary"]["paired_rebuilt_vs_complete_lineage"]
    boundary_p = exact_mcnemar(paired["rebuilt_only"], paired["complete_lineage_only"])
    ordered = sorted([("substitution", main_p), ("boundary", boundary_p)], key=lambda item: item[1])
    holm = {}
    running = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - rank) * p_value)
        running = max(running, adjusted)
        holm[name] = running

    boundary_eligible = int(boundary["summary"]["eligible_initial_target_and_f0_safe"])
    boundary_mirages = int(boundary["summary"]["repair_mirage"]["events"])
    boundary_control = int(boundary["summary"]["complete_lineage_recurrence"]["events"])
    frozen = int(boundary["summary"]["incidents"])
    differences = np.asarray([
        row["complete_lineage_repair"]["hit_at_serve_k"] - row["poisoned"]["hit_at_serve_k"]
        for row in utility["records"]
    ], dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    bootstrap = np.asarray([
        rng.choice(differences, size=len(differences), replace=True).mean()
        for _ in range(args.bootstrap_samples)
    ])
    utility_counts = {}
    for condition in ("clean", "poisoned", "complete_lineage_repair"):
        utility_counts[condition] = {
            "hit_at_30": int(sum(row[condition]["hit_at_retrieve_k"] for row in utility["records"])),
            "hit_at_5": int(sum(row[condition]["hit_at_serve_k"] for row in utility["records"])),
            "queries": len(utility["records"]),
        }
    payload = {
        "paper_evidence": False,
        "status": "statistics_over_llm_semantic_audit_pending_human_labels",
        "primary_hypotheses": {
            "substitution": {"raw_p": main_p, "holm_adjusted_p": holm["substitution"]},
            "boundary": {
                "raw_p": boundary_p,
                "holm_adjusted_p": holm["boundary"],
                "partial_recurrence": rate(boundary_mirages, boundary_eligible),
                "complete_lineage_recurrence": rate(boundary_control, boundary_eligible),
                "unconditional_partial_recurrence": rate(boundary_mirages, frozen),
            },
        },
        "utility": {
            "raw_counts": utility_counts,
            "poisoned_to_complete_lineage_hit_at_5_gains": int(sum(differences == 1)),
            "poisoned_to_complete_lineage_hit_at_5_losses": int(sum(differences == -1)),
            "paired_mean_change": float(differences.mean()),
            "paired_bootstrap_95_interval": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
