from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_full_pipeline_dev import document_text, fingerprint_path, read_jsonl, sha256_file
from run_repair_lattice_independent_cert import ACTION_NAMES, canonical_sha256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--incidents", type=Path, required=True)
    parser.add_argument("--clean-vector-cache", type=Path)
    parser.add_argument("--query-vector-cache", type=Path)
    parser.add_argument("--embedder-path")
    parser.add_argument("--reranker-path")
    parser.add_argument("--generator-path")
    parser.add_argument("--runner-script", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_artifact.read_text(encoding="utf-8"))
    incidents = json.loads(args.incidents.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates_by_id = {row["query_id"]: row for row in candidates["records"]}
    incident_by_id = {row["question_id"]: row for row in incidents[:30]}
    reference_by_id = {row["incident_id"]: row for row in reference["records"]}
    records = artifact.get("records", [])
    record_by_id = {row["incident_id"]: row for row in records}
    require(len(records) == len(record_by_id) == 30, "expected 30 unique incident records")
    require(set(record_by_id) == set(reference_by_id), "incident IDs differ from immutable reference")
    require(list(record_by_id) == [row["question_id"] for row in incidents[:30]], "certificate rows are not aligned with frozen incident order")
    require(artifact.get("tier") == "posthoc_independent_f1_f2_mechanical_certificate", "unexpected artifact tier")

    input_paths = {
        "corpus": args.corpus,
        "candidates": args.candidates,
        "incidents": args.incidents,
        "reference_artifact": args.reference_artifact,
    }
    for name, path in input_paths.items():
        require(path.exists(), f"missing verifier input: {path}")
        require(
            artifact["inputs"][name]["sha256"] == sha256_file(path),
            f"{name} hash mismatch",
        )

    excluded_relevant = {
        str(document["_id"])
        for incident in incidents[:30]
        for document in candidates_by_id[incident["beir_query_id"]]["relevant_documents"]
    }
    corpus = list(read_jsonl(args.corpus))
    base_ids = [str(row["_id"]) for row in corpus if str(row["_id"]) not in excluded_relevant]
    base_texts = [document_text(row) for row in corpus if str(row["_id"]) not in excluded_relevant]
    stored_base = artifact["configuration"]["unchanged_base_bundle"]
    require(stored_base["count"] == len(base_ids), "unchanged-base row count mismatch")
    require(stored_base["ids_sha256"] == canonical_sha256(base_ids), "unchanged-base ID alignment mismatch")
    require(stored_base["texts_sha256"] == canonical_sha256(base_texts), "unchanged-base text alignment mismatch")

    fingerprint_arguments = (
        args.clean_vector_cache, args.query_vector_cache, args.embedder_path,
        args.reranker_path, args.generator_path, args.runner_script,
    )
    if any(value is not None for value in fingerprint_arguments) and not all(value is not None for value in fingerprint_arguments):
        raise ValueError("all cache/model/runner fingerprint arguments must be supplied together")
    full_input_fingerprint_gate = all(value is not None for value in fingerprint_arguments)
    if full_input_fingerprint_gate:
        require(sha256_file(args.clean_vector_cache) == artifact["inputs"]["clean_vector_cache"]["sha256"], "clean vector-cache hash mismatch")
        require(sha256_file(args.query_vector_cache) == artifact["inputs"]["query_vector_cache"]["sha256"], "query vector-cache hash mismatch")
        clean_vectors = np.load(args.clean_vector_cache, mmap_mode="r")
        query_vectors = np.load(args.query_vector_cache, mmap_mode="r")
        require(len(clean_vectors) == len(corpus), "clean vector-cache row count does not align with corpus")
        require(len(query_vectors) >= len(incidents[:30]), "query vector-cache has fewer rows than frozen incidents")
        transform = artifact["configuration"]["transform_fingerprint_bundle"]
        require(fingerprint_path(args.embedder_path) == transform["embedder"]["fingerprint"], "embedder tree fingerprint mismatch")
        require(fingerprint_path(args.reranker_path) == transform["reranker"]["fingerprint"], "reranker tree fingerprint mismatch")
        require(fingerprint_path(args.generator_path) == transform["generator"]["fingerprint"], "generator tree fingerprint mismatch")
        require(sha256_file(args.runner_script) == artifact["runtime"]["script_sha256"], "runner script hash mismatch")

    base_bundle = stored_base
    checks = 0
    maximum_error = 0.0
    reference_checks = 0
    reference_matches = 0
    selected_count = 0
    fixed_count = 0
    selected_unsafe = 0
    selected_correct = 0
    for incident_id in sorted(record_by_id):
        row = record_by_id[incident_id]
        old = reference_by_id[incident_id]
        incident = incident_by_id[incident_id]
        require(row["question"] == incident["question"], f"{incident_id}: question drift")
        require(row["correct_answer"] == incident["correct_answer"], f"{incident_id}: correct-answer drift")
        require(row["target_answer"] == incident["target_answer"], f"{incident_id}: target-answer drift")
        require(set(row["actions"]) == set(ACTION_NAMES), f"{incident_id}: action grammar mismatch")
        require(row["initial"]["answer"] == old["initial"]["answer"], f"{incident_id}: initial answer drift")
        require(row["initial"]["reference_answer_exact"] is True, f"{incident_id}: invalid initial reference flag")
        reference_checks += 1
        reference_matches += 1
        for action in ACTION_NAMES:
            details = row["actions"][action]
            prefix = f"{incident_id}/{action}"
            require(details["action_spec_sha256"] == canonical_sha256(details["action_spec"]), f"{prefix}: action hash mismatch")
            lineage = details["lineage"]
            require(lineage["seed_kind"] == "source_root", f"{prefix}: seed must be source_root")
            require(lineage["seed_node_ids"] == [lineage["source_root_id"]], f"{prefix}: malformed seed node")
            require(len(lineage["invalidated_descendant_ids"]) == len(set(lineage["invalidated_descendant_ids"])), f"{prefix}: duplicate invalidated IDs")
            require(len(lineage["materialized_descendant_ids"]) == len(set(lineage["materialized_descendant_ids"])), f"{prefix}: duplicate materialized IDs")
            require(set(lineage["invalidated_descendant_ids"]).isdisjoint(lineage["materialized_descendant_ids"]), f"{prefix}: old and new descendant IDs overlap")
            require(
                lineage["closure_sha256"] == canonical_sha256({
                    "seed_node_ids": lineage["seed_node_ids"],
                    "invalidated_descendant_ids": lineage["invalidated_descendant_ids"],
                    "materialized_descendant_ids": lineage["materialized_descendant_ids"],
                }),
                f"{prefix}: closure hash mismatch",
            )
            require(
                details["f1_source_revision_sha256"]
                == details["f2_source_revision_sha256"]
                == lineage["source_revision_after_sha256"],
                f"{prefix}: repaired source revision mismatch",
            )
            require(details["unchanged_base_reuse"] == base_bundle, f"{prefix}: unchanged-base bundle drift")
            f1 = details["f1_exact_view"]
            f2 = details["f2_physical_rebuild"]
            certificate = details["certificate"]
            required_flags = (
                "source_revision_sha256_exact",
                "materialized_chunks_exact",
                "metadata_exact",
                "source_vectors_within_atol",
                "dense_ids_exact",
                "served_ids_exact",
                "prompt_sha256_exact",
                "answer_exact",
            )
            require(all(certificate.get(name) is True for name in required_flags), f"{prefix}: certificate flag failed")
            require(certificate.get("all_required_checks_pass") is True, f"{prefix}: aggregate certificate flag failed")
            require(f1["dense_ids"] == f2["dense_ids"], f"{prefix}: dense ranking mismatch")
            require(f1["served_ids"] == f2["served_ids"], f"{prefix}: served ranking mismatch")
            require(f1["prompt_sha256"] == f2["prompt_sha256"], f"{prefix}: prompt mismatch")
            require(f1["answer"] == f2["answer"], f"{prefix}: answer mismatch")
            require(f1["metadata_sha256"] == f2["metadata_sha256"], f"{prefix}: metadata mismatch")
            require(float(certificate["source_vectors_max_abs_error"]) <= float(artifact["configuration"]["vector_atol"]), f"{prefix}: vector tolerance exceeded")
            maximum_error = max(maximum_error, float(certificate["source_vectors_max_abs_error"]))
            require(details["f0_trace"]["answer"] == old["actions"][action]["f0_trace"]["answer"], f"{prefix}: F0 answer drift")
            require(f2["answer"] == old["actions"][action]["f2_physical_rebuild"]["answer"], f"{prefix}: F2 answer drift")
            require(details["f0_trace"]["reference_answer_exact"] is True, f"{prefix}: invalid F0 reference flag")
            require(f2["reference_answer_exact"] is True, f"{prefix}: invalid F2 reference flag")
            checks += 1
            reference_checks += 2
            reference_matches += 2

        lexical_feasible = [
            (details["edit_tokens"], action)
            for action, details in row["actions"].items()
            if not details["f1_exact_view"]["target_present"]
            and details["f1_exact_view"]["correct_present"]
            and details["certificate"]["all_required_checks_pass"]
        ]
        fixed_feasible = [
            (details["edit_tokens"], action)
            for action, details in row["actions"].items()
            if not details["f0_trace"]["target_present"]
        ]
        expected_selected = min(lexical_feasible)[1] if lexical_feasible else None
        expected_fixed = min(fixed_feasible)[1] if fixed_feasible else None
        require(row["selection"]["reindexrepair_best_validated"] == expected_selected, f"{incident_id}: lexical selection mismatch")
        require(row["selection"]["exhaustive_f2_minimum"] == expected_selected, f"{incident_id}: stored lexical oracle mismatch")
        require(row["selection"]["fixed_trace_cheapest_safe"] == expected_fixed, f"{incident_id}: fixed-trace selection mismatch")
        require(row["selection"]["tie_break"] == ["edit_tokens_ascending", "action_name_lexicographic"], f"{incident_id}: selection tie-break drift")
        if expected_selected:
            selected_count += 1
            selected_unsafe += int(row["actions"][expected_selected]["f2_physical_rebuild"]["target_present"])
            selected_correct += int(row["actions"][expected_selected]["f2_physical_rebuild"]["correct_present"])
        fixed_count += int(expected_fixed is not None)

    summary = artifact["summary"]
    require(checks == 120, "expected 120 action certificates")
    require(summary["actions_certified"] == summary["actions_total"] == checks, "certificate summary mismatch")
    require(summary["all_independent_f1_f2_certificates_pass"] is True, "global certificate failed")
    require(summary["all_human_stimulus_answers_unchanged"] is True, "reference answer preservation failed")
    require(summary["reference_answer_checks"] == reference_checks == 270, "reference check count mismatch")
    require(summary["reference_answer_matches"] == reference_matches, "reference match count mismatch")
    require(abs(float(summary["maximum_source_vector_abs_error"]) - maximum_error) <= 1e-12, "maximum vector error mismatch")
    require(summary["reindexrepair_selected"] == selected_count, "selected-count summary mismatch")
    require(summary["fixed_trace_selected"] == fixed_count, "fixed-count summary mismatch")
    require(summary["reindexrepair_f2_unsafe"] == selected_unsafe, "selected-safety summary mismatch")
    require(summary["reindexrepair_correct_present"] == selected_correct, "selected-correctness summary mismatch")

    verification = {
        "status": "verified",
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "reference_artifact_sha256": hashlib.sha256(args.reference_artifact.read_bytes()).hexdigest(),
        "incidents": len(records),
        "action_certificates": checks,
        "reference_answer_checks": reference_checks,
        "reference_answer_matches": reference_matches,
        "maximum_source_vector_abs_error": maximum_error,
        "query_row_ids_sha256": canonical_sha256([row["question_id"] for row in incidents[:30]]),
        "query_row_texts_sha256": canonical_sha256([row["question"] for row in incidents[:30]]),
        "clean_row_ids_sha256": canonical_sha256([str(row["_id"]) for row in corpus]),
        "full_input_fingerprint_gate": full_input_fingerprint_gate,
        "all_required_checks_pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verification, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
