from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from reindexrepair.metrics import clopper_pearson_interval


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate_record(events: int, eligible: int) -> dict[str, object]:
    return {
        "events": events,
        "eligible": eligible,
        "rate": events / eligible if eligible else None,
        "two_sided_95_exact_interval": list(clopper_pearson_interval(events, eligible)) if eligible else None,
    }


def cohen_kappa(left: list[bool], right: list[bool]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("kappa inputs must be equally sized and nonempty")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    p_left = sum(left) / len(left)
    p_right = sum(right) / len(right)
    expected = p_left * p_right + (1 - p_left) * (1 - p_right)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def exact_mcnemar_two_sided(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(left_only, right_only) + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    records = artifact["records"]
    by_incident: dict[str, dict[str, dict]] = {}
    for record in records:
        incident_id, condition = record["item_id"].split("|", 1)
        by_incident.setdefault(incident_id, {})[condition] = record["final"]
    required = {"initial", "f0", "partial_f2", "complete_f2"}
    if any(set(labels) != required for labels in by_incident.values()):
        raise ValueError("each incident must contain exactly four conditions")
    eligible = {
        incident_id: labels for incident_id, labels in by_incident.items()
        if labels["initial"]["target_adopted"] and not labels["f0"]["target_adopted"]
    }
    left_only = sum(labels["partial_f2"]["target_adopted"] and not labels["complete_f2"]["target_adopted"] for labels in eligible.values())
    right_only = sum(not labels["partial_f2"]["target_adopted"] and labels["complete_f2"]["target_adopted"] for labels in eligible.values())
    both = sum(labels["partial_f2"]["target_adopted"] and labels["complete_f2"]["target_adopted"] for labels in eligible.values())
    differences = np.asarray([
        int(labels["partial_f2"]["target_adopted"]) - int(labels["complete_f2"]["target_adopted"])
        for labels in eligible.values()
    ], dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    bootstrap = np.asarray([
        rng.choice(differences, size=len(differences), replace=True).mean()
        for _ in range(args.bootstrap_samples)
    ])
    agreement = {}
    for key in ("target_adopted", "correct_adopted", "complete"):
        left = [record["blind_pass_1"][key] for record in records]
        right = [record["blind_pass_2"][key] for record in records]
        agreement[key] = {
            "raw_agreement": sum(a == b for a, b in zip(left, right)) / len(left),
            "cohen_kappa": cohen_kappa(left, right),
        }
    partial_events = sum(labels["partial_f2"]["target_adopted"] for labels in eligible.values())
    control_events = sum(labels["complete_f2"]["target_adopted"] for labels in eligible.values())
    payload = {
        "paper_evidence": False,
        "status": "llm_semantic_sensitivity_pending_independent_human_annotation",
        "input_sha256": sha256_file(args.input),
        "sample_flow": {
            "frozen_incidents": len(by_incident),
            "outputs_judged": len(records),
            "initial_target_adopted": sum(labels["initial"]["target_adopted"] for labels in by_incident.values()),
            "eligible_initial_target_and_f0_safe": len(eligible),
            "incomplete_final_labels": sum(not record["final"]["complete"] for record in records),
        },
        "inter_pass_agreement": agreement,
        "paired_primary": {
            "partial_visible_deletion": rate_record(partial_events, len(eligible)),
            "complete_lineage_deletion": rate_record(control_events, len(eligible)),
            "partial_only": left_only,
            "complete_only": right_only,
            "both": both,
            "neither": len(eligible) - left_only - right_only - both,
            "paired_risk_difference": float(differences.mean()),
            "paired_bootstrap_95_interval": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
            "exact_mcnemar_two_sided_p": exact_mcnemar_two_sided(left_only, right_only),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
        },
        "unconditional": {
            "partial_mirages": partial_events,
            "complete_lineage_target_adoptions": control_events,
            "frozen_incidents": len(by_incident),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
