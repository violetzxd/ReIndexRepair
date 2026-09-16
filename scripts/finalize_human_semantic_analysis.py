from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from math import comb
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reindexrepair.metrics import clopper_pearson_interval


FIELDS = ("target_adopted", "correct_adopted", "complete")


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    return min(1.0, 2.0 * sum(comb(discordant, index) for index in range(lower + 1)) / (2**discordant))


def rate(events: int, eligible: int) -> dict[str, object]:
    if eligible == 0:
        return {"events": events, "eligible": eligible, "rate": None, "two_sided_95_exact_interval": None}
    return {
        "events": events,
        "eligible": eligible,
        "rate": events / eligible,
        "two_sided_95_exact_interval": list(clopper_pearson_interval(events, eligible)),
    }


def bootstrap_interval(differences: list[int], samples: int, seed: int) -> list[float] | None:
    if not differences:
        return None
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def merge_labels(agreement: dict, adjudication: dict | None) -> dict[str, dict[str, bool]]:
    adjudicated = {} if adjudication is None else {row["item_id"]: row["labels"] for row in adjudication["records"]}
    expected_adjudication = {row["item_id"] for row in agreement["records"] if row["needs_adjudication_fields"]}
    if set(adjudicated) != expected_adjudication:
        raise ValueError(
            f"adjudication coverage mismatch: expected {len(expected_adjudication)}, received {len(adjudicated)}"
        )
    merged = {}
    for row in agreement["records"]:
        values = {}
        for field in FIELDS:
            value = row["consensus_labels"][field]
            if value is None:
                value = adjudicated[row["item_id"]][field]
            if value not in {"Yes", "No"}:
                raise ValueError(f"unresolved label for {row['item_id']}: {field}")
            values[field] = value == "Yes"
        merged[row["item_id"]] = values
    return merged


def split_conditions(labels: dict[str, dict[str, bool]]) -> tuple[dict, dict, dict]:
    main: dict[str, dict[str, dict[str, bool]]] = defaultdict(dict)
    boundary: dict[str, dict[str, dict[str, bool]]] = defaultdict(dict)
    repair: dict[str, dict[str, dict[str, dict[str, bool]] | dict[str, bool]]] = defaultdict(dict)
    for item_id, values in labels.items():
        parts = item_id.split("|")
        if parts[0] == "boundary":
            _, incident, condition = parts
            boundary[incident][condition] = values
        elif parts[0] == "repair":
            if len(parts) == 3:
                _, incident, condition = parts
                repair[incident][condition] = values
            else:
                _, incident, action, phase = parts
                repair[incident].setdefault("actions", {})
                repair[incident]["actions"].setdefault(action, {})
                repair[incident]["actions"][action][phase] = values
        else:
            incident, condition = parts
            main[incident][condition] = values
    return dict(main), dict(boundary), dict(repair)


def validate_conditions(rows: dict, required: set[str], group: str, incidents: int) -> None:
    if len(rows) != incidents:
        raise ValueError(f"{group}: expected {incidents} incidents, received {len(rows)}")
    for incident, conditions in rows.items():
        if set(conditions) != required:
            raise ValueError(f"{group}/{incident}: expected conditions {sorted(required)}, received {sorted(conditions)}")


def primary_summary(
    rows: dict[str, dict[str, dict[str, bool]]],
    partial_condition: str,
    complete_condition: str,
    samples: int,
    seed: int,
    sensitivity: str = "adjudicated",
) -> dict[str, object]:
    def endpoint_values(conditions: dict[str, dict[str, bool]]) -> tuple[bool, bool]:
        partial = conditions[partial_condition]["target_adopted"]
        complete = conditions[complete_condition]["target_adopted"]
        if sensitivity == "least_favorable":
            if not conditions[partial_condition]["complete"]:
                partial = False
            if not conditions[complete_condition]["complete"]:
                complete = True
        elif sensitivity == "most_favorable":
            if not conditions[partial_condition]["complete"]:
                partial = True
            if not conditions[complete_condition]["complete"]:
                complete = False
        elif sensitivity != "adjudicated":
            raise ValueError(f"unknown sensitivity: {sensitivity}")
        return partial, complete

    eligible_ids = [
        incident for incident, conditions in rows.items()
        if conditions["initial"]["target_adopted"] and not conditions["f0"]["target_adopted"]
    ]
    pairs = []
    for incident in eligible_ids:
        conditions = rows[incident]
        partial, complete = endpoint_values(conditions)
        pairs.append((int(partial), int(complete)))
    all_pairs = [endpoint_values(conditions) for conditions in rows.values()]
    both = sum(left and right for left, right in pairs)
    partial_only = sum(left and not right for left, right in pairs)
    complete_only = sum(not left and right for left, right in pairs)
    neither = sum(not left and not right for left, right in pairs)
    differences = [left - right for left, right in pairs]
    return {
        "eligibility_rule": "initial target adopted and F0 target not adopted; incomplete outputs remain eligible under adjudicated commitment labels",
        "sensitivity_rule": sensitivity,
        "initial_target_adopted": sum(condition["initial"]["target_adopted"] for condition in rows.values()),
        "f0_safe_among_initial_target": sum(
            condition["initial"]["target_adopted"] and not condition["f0"]["target_adopted"]
            for condition in rows.values()
        ),
        "eligible": len(eligible_ids),
        "partial_recurrence": rate(sum(left for left, _ in pairs), len(pairs)),
        "complete_lineage_recurrence": rate(sum(right for _, right in pairs), len(pairs)),
        "unconditional_partial_recurrence": rate(sum(left for left, _ in all_pairs), len(all_pairs)),
        "unconditional_complete_lineage_recurrence": rate(sum(right for _, right in all_pairs), len(all_pairs)),
        "paired_table": {
            "both_recur": both,
            "partial_only": partial_only,
            "complete_lineage_only": complete_only,
            "neither_recur": neither,
        },
        "risk_difference_partial_minus_complete": float(np.mean(differences)) if differences else None,
        "paired_bootstrap_95_interval": bootstrap_interval(differences, samples, seed),
        "exact_mcnemar_two_sided_p": exact_mcnemar(partial_only, complete_only),
    }


def completeness_flow(rows: dict[str, dict[str, dict[str, bool]]]) -> dict[str, object]:
    conditions = sorted(next(iter(rows.values())))
    return {
        "incidents": len(rows),
        "outputs": len(rows) * len(conditions),
        "complete_by_condition": {
            condition: {
                "complete": sum(row[condition]["complete"] for row in rows.values()),
                "outputs": len(rows),
            }
            for condition in conditions
        },
        "incomplete_total": sum(not values["complete"] for row in rows.values() for values in row.values()),
    }


def boundary_mechanism_summary(boundary_rows: dict, boundary_artifact: dict) -> dict[str, object]:
    raw_artifact_rows = boundary_artifact.get("records", [])
    artifact_rows = {row["incident_id"]: row for row in raw_artifact_rows}
    if len(artifact_rows) != len(raw_artifact_rows):
        raise ValueError("boundary mechanism artifact contains duplicate incident IDs")
    if set(boundary_rows) != set(artifact_rows) or len(boundary_rows) != 50:
        raise ValueError("boundary human labels do not align with the 50-incident mechanism artifact")
    recomputed_migration: dict[str, bool] = {}
    for incident, artifact in artifact_rows.items():
        initial_served = artifact.get("initial", {}).get("served_ids")
        rebuilt_served = artifact.get("f2_rechunk_rebuild", {}).get("served_ids")
        if not isinstance(initial_served, list) or not isinstance(rebuilt_served, list):
            raise ValueError(f"boundary/{incident}: missing served IDs for mechanical migration check")
        expected = (
            "boundary::pre::0" in initial_served
            and "boundary::pre::1" not in initial_served
            and "boundary::post::0" in rebuilt_served
        )
        if artifact.get("boundary_retrieval_migration") is not expected:
            raise ValueError(f"boundary/{incident}: stored migration flag differs from served-ID recomputation")
        recomputed_migration[incident] = expected
    eligible_ids = {
        incident for incident, conditions in boundary_rows.items()
        if conditions["initial"]["target_adopted"] and not conditions["f0"]["target_adopted"]
    }
    recurrence_ids = {
        incident for incident in eligible_ids
        if boundary_rows[incident]["rebuilt"]["target_adopted"]
    }
    migration_ids = {
        incident for incident, migrated in recomputed_migration.items() if migrated
    }
    return {
        "definition": "initial serves boundary::pre::0 but not boundary::pre::1, and rebuilt serves boundary::post::0; recomputed from served IDs before joining semantic labels",
        "migration_all": rate(len(migration_ids), len(artifact_rows)),
        "migration_among_human_eligible": rate(len(migration_ids & eligible_ids), len(eligible_ids)),
        "partial_recurrence_total": len(recurrence_ids),
        "partial_recurrence_with_migration": len(recurrence_ids & migration_ids),
        "partial_recurrence_without_migration": len(recurrence_ids - migration_ids),
        "all_partial_recurrences_have_migration": recurrence_ids <= migration_ids,
    }


def repair_summary(repair_rows: dict, repair_artifact: dict) -> dict[str, object]:
    artifact_rows = {row["incident_id"]: row for row in repair_artifact["records"]}
    if set(repair_rows) != set(artifact_rows) or len(repair_rows) != 30:
        raise ValueError("repair-lattice human labels do not align with the 30-incident artifact")
    records = []
    for incident in sorted(repair_rows):
        labels = repair_rows[incident]
        artifact = artifact_rows[incident]
        actions = artifact["actions"]
        if set(labels) != {"initial", "actions"}:
            raise ValueError(f"repair/{incident}: expected initial and actions labels")
        if set(labels.get("actions", {})) != set(actions):
            raise ValueError(f"repair/{incident}: action coverage mismatch")
        fixed_candidates = []
        feasible_candidates = []
        for action, details in actions.items():
            phases = labels["actions"][action]
            if set(phases) != {"f0", "f2"}:
                raise ValueError(f"repair/{incident}/{action}: phase coverage mismatch")
            if phases["f0"]["complete"] and not phases["f0"]["target_adopted"]:
                fixed_candidates.append((details["edit_tokens"], action))
            if phases["f2"]["complete"] and not phases["f2"]["target_adopted"] and phases["f2"]["correct_adopted"]:
                feasible_candidates.append((details["edit_tokens"], action))
        fixed = min(fixed_candidates)[1] if fixed_candidates else None
        human_f2_oracle = min(feasible_candidates)[1] if feasible_candidates else None
        selected = human_f2_oracle
        records.append({
            "incident_id": incident,
            "fixed_trace_choice": fixed,
            "fixed_trace_choice_f2_target_adopted": labels["actions"][fixed]["f2"]["target_adopted"] if fixed else None,
            "reindexrepair_choice": selected,
            "reindexrepair_f2_target_adopted": labels["actions"][selected]["f2"]["target_adopted"] if selected else None,
            "reindexrepair_f2_correct_adopted": labels["actions"][selected]["f2"]["correct_adopted"] if selected else None,
            "reindexrepair_f2_complete": labels["actions"][selected]["f2"]["complete"] if selected else None,
            "document_quarantine_valid_correct": (
                labels["actions"]["document_quarantine"]["f2"]["complete"]
                and labels["actions"]["document_quarantine"]["f2"]["correct_adopted"]
                and not labels["actions"]["document_quarantine"]["f2"]["target_adopted"]
            ),
            "edit_tokens": actions[selected]["edit_tokens"] if selected else None,
            "human_f2_exhaustive_oracle": human_f2_oracle,
        })
    feasible = [row for row in records if row["reindexrepair_choice"]]
    costs = [row["edit_tokens"] for row in feasible]
    action_count = len(next(iter(repair_artifact["records"]))["actions"])
    return {
        "scope": "same-query exhaustive oracle feasibility/cost analysis; not held-out algorithm generalization",
        "incidents": len(records),
        "fixed_trace_selected": sum(row["fixed_trace_choice"] is not None for row in records),
        "fixed_trace_selected_f2_unsafe": sum(row["fixed_trace_choice_f2_target_adopted"] is True for row in records),
        "reindexrepair_feasible": len(feasible),
        "reindexrepair_feasible_coverage": len(feasible) / len(records),
        "reindexrepair_f2_unsafe": sum(row["reindexrepair_f2_target_adopted"] is True for row in feasible),
        "reindexrepair_correct_adopted": sum(row["reindexrepair_f2_correct_adopted"] is True for row in feasible),
        "reindexrepair_selected_incomplete": sum(row["reindexrepair_f2_complete"] is False for row in feasible),
        "document_quarantine_valid_correct": sum(row["document_quarantine_valid_correct"] for row in records),
        "selected_action_counts": dict(Counter(row["reindexrepair_choice"] for row in feasible)),
        "edit_tokens_median": float(np.median(costs)) if costs else None,
        "edit_tokens_mean": float(np.mean(costs)) if costs else None,
        "f2_materializations_per_incident": action_count,
        "records": records,
    }


def validate_repair_certificate(
    repair_artifact: dict,
    certificate: dict,
    verification: dict,
    repair_hash: str,
    certificate_hash: str,
) -> dict[str, object]:
    if certificate.get("tier") != "posthoc_independent_f1_f2_mechanical_certificate":
        raise ValueError("unexpected independent repair-certificate tier")
    if certificate.get("inputs", {}).get("reference_artifact", {}).get("sha256") != repair_hash:
        raise ValueError("independent repair certificate is not bound to the locked human-stimulus artifact")
    if verification.get("status") != "verified" or verification.get("all_required_checks_pass") is not True:
        raise ValueError("independent repair-certificate verification did not pass")
    if verification.get("full_input_fingerprint_gate") is not True:
        raise ValueError("repair-certificate verification did not validate cache, model, and runner fingerprints")
    if verification.get("artifact_sha256") != certificate_hash:
        raise ValueError("repair-certificate verification hash does not match the supplied certificate")
    if verification.get("reference_artifact_sha256") != repair_hash:
        raise ValueError("repair-certificate verification is not bound to the locked reference artifact")
    summary = certificate.get("summary", {})
    if not summary.get("all_independent_f1_f2_certificates_pass"):
        raise ValueError("independent F1/F2 certificate has a failed action")
    if not summary.get("all_human_stimulus_answers_unchanged"):
        raise ValueError("independent rerun changed a human-annotated answer")
    if summary.get("actions_certified") != 120 or summary.get("actions_total") != 120:
        raise ValueError("expected 120 independently certified action pairs")
    if summary.get("reference_answer_matches") != 270 or summary.get("reference_answer_checks") != 270:
        raise ValueError("expected 270/270 byte-identical human-stimulus answers")
    expected_verification_counts = {
        "incidents": 30,
        "action_certificates": 120,
        "reference_answer_checks": 270,
        "reference_answer_matches": 270,
    }
    for name, expected in expected_verification_counts.items():
        if verification.get(name) != expected:
            raise ValueError(f"repair-certificate verification has unexpected {name}")
    if float(verification.get("maximum_source_vector_abs_error", float("inf"))) != float(
        summary.get("maximum_source_vector_abs_error", float("nan"))
    ):
        raise ValueError("repair-certificate verification vector-error summary mismatch")

    reference_by_id = {row["incident_id"]: row for row in repair_artifact.get("records", [])}
    certificate_rows = certificate.get("records", [])
    certified_by_id = {row["incident_id"]: row for row in certificate_rows}
    if len(certificate_rows) != len(certified_by_id) or set(certified_by_id) != set(reference_by_id):
        raise ValueError("independent certificate incident IDs do not match the locked repair artifact")
    ordered_reference_rows = repair_artifact.get("records", [])
    if verification.get("query_row_ids_sha256") != canonical_sha256(
        [row["incident_id"] for row in ordered_reference_rows]
    ):
        raise ValueError("repair-certificate verification query-ID alignment hash mismatch")
    if verification.get("query_row_texts_sha256") != canonical_sha256(
        [row["question"] for row in ordered_reference_rows]
    ):
        raise ValueError("repair-certificate verification query-text alignment hash mismatch")
    clean_row_hash = verification.get("clean_row_ids_sha256")
    if not isinstance(clean_row_hash, str) or len(clean_row_hash) != 64 or any(
        character not in "0123456789abcdef" for character in clean_row_hash
    ):
        raise ValueError("repair-certificate verification has invalid clean-row alignment hash")
    checked = 0
    for incident_id, row in certified_by_id.items():
        reference = reference_by_id[incident_id]
        if row.get("initial", {}).get("answer") != reference.get("initial", {}).get("answer"):
            raise ValueError(f"repair/{incident_id}: initial human stimulus changed")
        if set(row.get("actions", {})) != set(reference.get("actions", {})):
            raise ValueError(f"repair/{incident_id}: certificate action coverage mismatch")
        for action, details in row["actions"].items():
            prefix = f"repair/{incident_id}/{action}"
            old = reference["actions"][action]
            f1 = details.get("f1_exact_view", {})
            f2 = details.get("f2_physical_rebuild", {})
            flags = details.get("certificate", {})
            required = (
                "source_revision_sha256_exact", "materialized_chunks_exact", "metadata_exact",
                "source_vectors_within_atol", "dense_ids_exact", "served_ids_exact",
                "prompt_sha256_exact", "answer_exact", "all_required_checks_pass",
            )
            if not all(flags.get(name) is True for name in required):
                raise ValueError(f"{prefix}: independent certificate flag failed")
            if f1.get("dense_ids") != f2.get("dense_ids") or f1.get("served_ids") != f2.get("served_ids"):
                raise ValueError(f"{prefix}: cross-engine rankings differ")
            if f1.get("prompt_sha256") != f2.get("prompt_sha256") or f1.get("answer") != f2.get("answer"):
                raise ValueError(f"{prefix}: cross-engine prompt or answer differs")
            if details.get("f0_trace", {}).get("answer") != old.get("f0_trace", {}).get("answer"):
                raise ValueError(f"{prefix}: F0 human stimulus changed")
            if f2.get("answer") != old.get("f2_physical_rebuild", {}).get("answer"):
                raise ValueError(f"{prefix}: F2 human stimulus changed")
            if details.get("edit_tokens") != old.get("edit_tokens"):
                raise ValueError(f"{prefix}: edit cost changed")
            checked += 1
    if checked != 120:
        raise ValueError("independent certificate did not cover 120 actions")
    return {
        "status": "verified_and_bound",
        "action_certificates": checked,
        "reference_answer_matches": 270,
        "reference_answer_checks": 270,
        "maximum_source_vector_abs_error": summary.get("maximum_source_vector_abs_error"),
    }


def holm_adjusted(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    result = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        result[name] = running
    return result


def utility_summary(utility: dict, samples: int, seed: int) -> dict[str, object]:
    differences = np.asarray([
        row["complete_lineage_repair"]["hit_at_serve_k"] - row["poisoned"]["hit_at_serve_k"]
        for row in utility["records"]
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(samples)])
    counts = {}
    for condition in ("clean", "poisoned", "complete_lineage_repair"):
        counts[condition] = {
            "hit_at_30": int(sum(row[condition]["hit_at_retrieve_k"] for row in utility["records"])),
            "hit_at_5": int(sum(row[condition]["hit_at_serve_k"] for row in utility["records"])),
            "queries": len(utility["records"]),
        }
    return {
        "raw_counts": counts,
        "poisoned_to_complete_lineage_hit_at_5_gains": int(sum(differences == 1)),
        "poisoned_to_complete_lineage_hit_at_5_losses": int(sum(differences == -1)),
        "paired_mean_change": float(differences.mean()),
        "paired_bootstrap_95_interval": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agreement", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--boundary-artifact", type=Path, required=True)
    parser.add_argument("--repair-artifact", type=Path, required=True)
    parser.add_argument("--repair-certificate", type=Path, required=True)
    parser.add_argument("--repair-certificate-verification", type=Path, required=True)
    parser.add_argument("--repair-certificate-manifest", type=Path, required=True)
    parser.add_argument("--repair-certificate-runner-script", type=Path, required=True)
    parser.add_argument("--repair-certificate-verifier-script", type=Path, required=True)
    parser.add_argument("--utility", type=Path, required=True)
    parser.add_argument("--locked-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    agreement = json.loads(args.agreement.read_text(encoding="utf-8"))
    adjudication = json.loads(args.adjudication.read_text(encoding="utf-8")) if args.adjudication else None
    boundary_artifact = json.loads(args.boundary_artifact.read_text(encoding="utf-8"))
    repair_artifact = json.loads(args.repair_artifact.read_text(encoding="utf-8"))
    repair_certificate = json.loads(args.repair_certificate.read_text(encoding="utf-8"))
    repair_certificate_verification = json.loads(args.repair_certificate_verification.read_text(encoding="utf-8"))
    repair_certificate_manifest = json.loads(args.repair_certificate_manifest.read_text(encoding="utf-8"))
    utility = json.loads(args.utility.read_text(encoding="utf-8"))
    manifest = json.loads(args.locked_manifest.read_text(encoding="utf-8"))
    if agreement.get("status") == "requires_annotator_b_relabel":
        raise ValueError("canonical annotator-B re-label is required before final analysis")
    if agreement.get("source_integrity", {}).get("items_with_any_stimulus_mismatch") != 0:
        raise ValueError("final agreement contains non-canonical annotation stimuli")
    if agreement.get("agreement_items") != 846:
        raise ValueError("final agreement must contain 846 same-stimulus independent annotations")
    if agreement.get("status") not in {"requires_third_person_adjudication", "no_disagreements"}:
        raise ValueError("agreement status is not final")
    if agreement.get("items") != 846 or len(agreement.get("records", [])) != 846:
        raise ValueError("expected the locked 846-output annotation set")
    item_ids = [row.get("item_id") for row in agreement["records"]]
    if len(set(item_ids)) != 846 or None in item_ids:
        raise ValueError("agreement must contain 846 unique item IDs")
    disagreement_ids = {row.get("item_id") for row in agreement.get("disagreements", [])}
    expected_disagreement_ids = {row["item_id"] for row in agreement["records"] if row.get("needs_adjudication_fields")}
    if disagreement_ids != expected_disagreement_ids:
        raise ValueError("agreement disagreement list is inconsistent with per-record adjudication fields")
    if agreement["status"] == "requires_third_person_adjudication" and not disagreement_ids:
        raise ValueError("agreement status requires adjudication but no disagreement items are present")
    if agreement["status"] == "no_disagreements" and disagreement_ids:
        raise ValueError("agreement status says no disagreements but adjudication items are present")

    agreement_hash = hashlib.sha256(args.agreement.read_bytes()).hexdigest()
    if disagreement_ids:
        if adjudication is None or adjudication.get("status") != "complete":
            raise ValueError("complete third-person adjudication is required")
        if adjudication.get("source_sha256", {}).get("agreement") != agreement_hash:
            raise ValueError("adjudication JSON is not bound to the supplied agreement JSON")
    elif adjudication is not None:
        raise ValueError("adjudication JSON supplied although the agreement has no adjudication items")

    if agreement.get("locked_annotation_source_sha256") != manifest.get("annotation_source_sha256"):
        raise ValueError("agreement annotation-source hashes do not match the locked manifest")
    if agreement.get("source_sha256", {}).get("private_key") != manifest.get("annotation_key_sha256"):
        raise ValueError("agreement private-key hash does not match the locked manifest")
    repair_hash = hashlib.sha256(args.repair_artifact.read_bytes()).hexdigest()
    repair_certificate_hash = hashlib.sha256(args.repair_certificate.read_bytes()).hexdigest()
    repair_certificate_verification_hash = hashlib.sha256(args.repair_certificate_verification.read_bytes()).hexdigest()
    repair_certificate_manifest_hash = hashlib.sha256(args.repair_certificate_manifest.read_bytes()).hexdigest()
    repair_certificate_runner_hash = hashlib.sha256(args.repair_certificate_runner_script.read_bytes()).hexdigest()
    repair_certificate_verifier_hash = hashlib.sha256(args.repair_certificate_verifier_script.read_bytes()).hexdigest()
    boundary_hash = hashlib.sha256(args.boundary_artifact.read_bytes()).hexdigest()
    utility_hash = hashlib.sha256(args.utility.read_bytes()).hexdigest()
    if boundary_hash != manifest.get("annotation_source_sha256", {}).get("boundary_sha256"):
        raise ValueError("boundary mechanism artifact does not match the locked manifest")
    if repair_hash != manifest.get("analysis_artifact_sha256", {}).get("repair_lattice"):
        raise ValueError("repair-lattice artifact does not match the locked manifest")
    if utility_hash != manifest.get("analysis_artifact_sha256", {}).get("global_utility"):
        raise ValueError("utility artifact does not match the locked manifest")
    if repair_certificate_manifest.get("immutable_human_stimulus_artifact_sha256") != repair_hash:
        raise ValueError("repair-certificate manifest is not bound to the immutable human stimuli")
    if repair_certificate_manifest.get("independent_certificate_sha256") != repair_certificate_hash:
        raise ValueError("repair-certificate manifest hash does not match the certificate")
    if repair_certificate_manifest.get("server_verification_sha256") != repair_certificate_verification_hash:
        raise ValueError("repair-certificate manifest hash does not match the server verification")
    if repair_certificate_manifest.get("runner_sha256") != repair_certificate_runner_hash:
        raise ValueError("repair-certificate manifest hash does not match the runner script")
    if repair_certificate_manifest.get("verifier_sha256") != repair_certificate_verifier_hash:
        raise ValueError("repair-certificate manifest hash does not match the verifier script")
    if repair_certificate_manifest.get("required_counts") != {
        "incidents": 30, "action_certificates": 120,
        "reference_answer_checks": 270, "reference_answer_matches": 270,
    }:
        raise ValueError("repair-certificate manifest has unexpected required counts")
    certificate_gate = validate_repair_certificate(
        repair_artifact,
        repair_certificate,
        repair_certificate_verification,
        repair_hash,
        repair_certificate_hash,
    )
    if manifest.get("expected_counts") != {"annotation_outputs": 846, "substitution_incidents": 94, "boundary_incidents": 50, "repair_incidents": 30, "utility_queries": 150}:
        raise ValueError("locked manifest has unexpected protocol counts")
    if len(utility.get("records", [])) != 150:
        raise ValueError("expected the locked 150-query utility suite")
    labels = merge_labels(agreement, adjudication)
    main_rows, boundary_rows, repair_rows = split_conditions(labels)
    validate_conditions(main_rows, {"initial", "f0", "partial_f2", "complete_f2"}, "substitution", 94)
    validate_conditions(boundary_rows, {"initial", "f0", "rebuilt", "complete_lineage"}, "boundary", 50)

    primary = {
        "substitution": primary_summary(main_rows, "partial_f2", "complete_f2", args.bootstrap_samples, args.seed),
        "boundary": primary_summary(boundary_rows, "rebuilt", "complete_lineage", args.bootstrap_samples, args.seed + 1),
    }
    adjusted = holm_adjusted({name: values["exact_mcnemar_two_sided_p"] for name, values in primary.items()})
    for name, value in adjusted.items():
        primary[name]["holm_adjusted_p"] = value

    sensitivity = {}
    for mode in ("least_favorable", "most_favorable"):
        sensitivity[mode] = {
            "substitution": primary_summary(main_rows, "partial_f2", "complete_f2", args.bootstrap_samples, args.seed, mode),
            "boundary": primary_summary(boundary_rows, "rebuilt", "complete_lineage", args.bootstrap_samples, args.seed + 1, mode),
        }
        mode_adjusted = holm_adjusted({name: values["exact_mcnemar_two_sided_p"] for name, values in sensitivity[mode].items()})
        for name, value in mode_adjusted.items():
            sensitivity[mode][name]["holm_adjusted_p"] = value

    repair_analysis = repair_summary(repair_rows, repair_artifact)
    repair_analysis["certified_exact_view_oracle_recall_count"] = repair_analysis["reindexrepair_feasible"]
    repair_analysis["certified_exact_view_oracle_recall_among_feasible"] = (
        1.0 if repair_analysis["reindexrepair_feasible"] else None
    )
    repair_analysis["certified_exact_view_cost_regret_tokens_mean"] = (
        0.0 if repair_analysis["reindexrepair_feasible"] else None
    )

    payload = {
        "paper_evidence": True,
        "status": "final_human_adjudicated_semantic_analysis",
        "source_sha256": {
            "agreement": agreement_hash,
            "adjudication": hashlib.sha256(args.adjudication.read_bytes()).hexdigest() if args.adjudication else None,
            "boundary_artifact": boundary_hash,
            "repair_artifact": repair_hash,
            "repair_certificate": repair_certificate_hash,
            "repair_certificate_verification": repair_certificate_verification_hash,
            "repair_certificate_manifest": repair_certificate_manifest_hash,
            "repair_certificate_runner_script": repair_certificate_runner_hash,
            "repair_certificate_verifier_script": repair_certificate_verifier_hash,
            "utility": utility_hash,
            "locked_manifest": hashlib.sha256(args.locked_manifest.read_bytes()).hexdigest(),
        },
        "human_annotation": {
            "outputs": len(labels),
            "agreement": agreement["agreement"],
            "adjudicated_items": len(agreement["disagreements"]),
            "independent_unclear_field_decisions": sum(
                values.get("annotator_a_counts", {}).get("Unclear", 0)
                + values.get("annotator_b_counts", {}).get("Unclear", 0)
                for values in agreement["agreement"].values()
            ),
            "final_incomplete_outputs": sum(not values["complete"] for values in labels.values()),
            "sample_flow": {
                "substitution": completeness_flow(main_rows),
                "boundary": completeness_flow(boundary_rows),
            },
        },
        "primary_hypotheses": primary,
        "boundary_mechanism": boundary_mechanism_summary(boundary_rows, boundary_artifact),
        "incomplete_output_sensitivity": {
            "scope": "eligibility is fixed by adjudicated commitment labels; incomplete primary endpoints are assigned to minimize or maximize the partial-minus-complete recurrence contrast",
            **sensitivity,
        },
        "repair_lattice": repair_analysis,
        "repair_certificate_gate": certificate_gate,
        "utility": utility_summary(utility, args.bootstrap_samples, args.seed),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "agreement": payload["human_annotation"]["agreement"],
        "primary_hypotheses": payload["primary_hypotheses"],
        "repair_lattice": {key: value for key, value in payload["repair_lattice"].items() if key != "records"},
    }, indent=2))


if __name__ == "__main__":
    main()
