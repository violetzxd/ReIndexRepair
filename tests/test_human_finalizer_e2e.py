from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def consensus_record(item_id: str, target: bool, correct: bool, complete: bool = True) -> dict:
    return {
        "item_id": item_id,
        "needs_adjudication_fields": [],
        "consensus_labels": {
            "target_adopted": "Yes" if target else "No",
            "correct_adopted": "Yes" if correct else "No",
            "complete": "Yes" if complete else "No",
        },
    }


def test_locked_846_output_finalizer_and_utility_coverage(tmp_path: Path) -> None:
    records = []
    for index in range(94):
        incident = f"main-{index:03d}"
        records.extend([
            consensus_record(f"{incident}|initial", True, False),
            consensus_record(f"{incident}|f0", False, True),
            consensus_record(f"{incident}|partial_f2", index < 12, index >= 12),
            consensus_record(f"{incident}|complete_f2", index < 2, index >= 2),
        ])
    for index in range(50):
        incident = f"boundary-{index:03d}"
        records.extend([
            consensus_record(f"boundary|{incident}|initial", True, False),
            consensus_record(f"boundary|{incident}|f0", False, True),
            consensus_record(f"boundary|{incident}|rebuilt", index < 6, index >= 6),
            consensus_record(f"boundary|{incident}|complete_lineage", False, True),
        ])

    actions = ["target_span", "visible_sentence", "all_attack_sentences", "document_quarantine"]
    repair_artifact_rows = []
    for index in range(30):
        incident = f"repair-{index:03d}"
        records.append(consensus_record(f"repair|{incident}|initial", True, False))
        artifact_actions = {}
        for action_index, action in enumerate(actions):
            records.append(consensus_record(f"repair|{incident}|{action}|f0", False, True))
            records.append(consensus_record(f"repair|{incident}|{action}|f2", False, True))
            answer = f"answer-{action}"
            artifact_actions[action] = {
                "edit_tokens": action_index + 1,
                "f0_trace": {"answer": f"f0-answer-{action}"},
                "f1_exact_view": {"matches_physical_f2": True, "answer": answer, "dense_ids": [action], "served_ids": [action]},
                "f2_physical_rebuild": {"answer": answer, "dense_ids": [action], "served_ids": [action]},
            }
        repair_artifact_rows.append({
            "incident_id": incident,
            "question": f"question-{incident}",
            "actions": artifact_actions,
            "selection": {"finalist_recall": True},
        })
    assert len(records) == 846

    boundary_path = tmp_path / "boundary.json"
    boundary_path.write_text(json.dumps({"records": [
        {
            "incident_id": f"boundary-{index:03d}",
            "boundary_retrieval_migration": index < 23,
            "initial": {"served_ids": ["boundary::pre::0"] if index < 23 else []},
            "f2_rechunk_rebuild": {"served_ids": ["boundary::post::0"] if index < 23 else []},
        }
        for index in range(50)
    ]}), encoding="utf-8")

    locked_sources = {
        "partial_sha256": "1" * 64,
        "complete_lineage_sha256": "2" * 64,
        "boundary_sha256": digest(boundary_path),
        "repair_sha256": "4" * 64,
    }
    key_hash = "5" * 64
    agreement = {
        "status": "no_disagreements",
        "items": 846,
        "agreement_items": 846,
        "source_integrity": {"items_with_any_stimulus_mismatch": 0},
        "source_sha256": {"private_key": key_hash},
        "locked_annotation_source_sha256": locked_sources,
        "agreement": {
            field: {"raw_agreement": 1.0, "cohen_kappa_three_class": 1.0}
            for field in ("target_adopted", "correct_adopted", "complete")
        },
        "records": records,
        "disagreements": [],
    }
    agreement_path = tmp_path / "agreement.json"
    agreement_path.write_text(json.dumps(agreement), encoding="utf-8")
    repair_path = tmp_path / "repair.json"
    repair_path.write_text(json.dumps({"records": repair_artifact_rows}), encoding="utf-8")

    certificate_rows = []
    for reference_row in repair_artifact_rows:
        incident = reference_row["incident_id"]
        certificate_actions = {}
        for action, old in reference_row["actions"].items():
            ids = [action]
            prompt_hash = canonical_digest([incident, action, "prompt"])
            answer = old["f2_physical_rebuild"]["answer"]
            action_spec = {"action": action, "edit_tokens": old["edit_tokens"]}
            certificate_actions[action] = {
                "action_spec": action_spec,
                "action_spec_sha256": canonical_digest(action_spec),
                "edit_tokens": old["edit_tokens"],
                "f0_trace": {"answer": old["f0_trace"]["answer"], "reference_answer_exact": True},
                "f1_exact_view": {"answer": answer, "dense_ids": ids, "served_ids": ids, "prompt_sha256": prompt_hash},
                "f2_physical_rebuild": {"answer": answer, "dense_ids": ids, "served_ids": ids, "prompt_sha256": prompt_hash, "reference_answer_exact": True},
                "certificate": {
                    "source_revision_sha256_exact": True,
                    "materialized_chunks_exact": True,
                    "metadata_exact": True,
                    "source_vectors_within_atol": True,
                    "dense_ids_exact": True,
                    "served_ids_exact": True,
                    "prompt_sha256_exact": True,
                    "answer_exact": True,
                    "all_required_checks_pass": True,
                    "source_vectors_max_abs_error": 0.0,
                },
            }
        certificate_rows.append({
            "incident_id": incident,
            "initial": {"answer": f"answer-initial-{incident}", "reference_answer_exact": True},
            "actions": certificate_actions,
        })
        reference_row["initial"] = {"answer": f"answer-initial-{incident}"}
    repair_path.write_text(json.dumps({"records": repair_artifact_rows}), encoding="utf-8")
    certificate = {
        "tier": "posthoc_independent_f1_f2_mechanical_certificate",
        "inputs": {"reference_artifact": {"sha256": digest(repair_path)}},
        "summary": {
            "all_independent_f1_f2_certificates_pass": True,
            "all_human_stimulus_answers_unchanged": True,
            "actions_certified": 120,
            "actions_total": 120,
            "reference_answer_matches": 270,
            "reference_answer_checks": 270,
            "maximum_source_vector_abs_error": 0.0,
        },
        "records": certificate_rows,
    }
    certificate_path = tmp_path / "repair_certificate.json"
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")
    verification_path = tmp_path / "repair_certificate_verification.json"
    verification_path.write_text(json.dumps({
        "status": "verified",
        "all_required_checks_pass": True,
        "full_input_fingerprint_gate": True,
        "artifact_sha256": digest(certificate_path),
        "reference_artifact_sha256": digest(repair_path),
        "incidents": 30,
        "action_certificates": 120,
        "reference_answer_checks": 270,
        "reference_answer_matches": 270,
        "maximum_source_vector_abs_error": 0.0,
        "query_row_ids_sha256": canonical_digest([row["incident_id"] for row in repair_artifact_rows]),
        "query_row_texts_sha256": canonical_digest([row["question"] for row in repair_artifact_rows]),
        "clean_row_ids_sha256": "a" * 64,
    }), encoding="utf-8")
    certificate_manifest_path = tmp_path / "repair_certificate_manifest.json"
    certificate_runner_path = tmp_path / "runner.py"
    certificate_verifier_path = tmp_path / "verifier.py"
    certificate_runner_path.write_text("# frozen runner\n", encoding="utf-8")
    certificate_verifier_path.write_text("# frozen verifier\n", encoding="utf-8")
    certificate_manifest_path.write_text(json.dumps({
        "immutable_human_stimulus_artifact_sha256": digest(repair_path),
        "independent_certificate_sha256": digest(certificate_path),
        "server_verification_sha256": digest(verification_path),
        "runner_sha256": digest(certificate_runner_path),
        "verifier_sha256": digest(certificate_verifier_path),
        "required_counts": {
            "incidents": 30,
            "action_certificates": 120,
            "reference_answer_checks": 270,
            "reference_answer_matches": 270,
        },
    }), encoding="utf-8")

    utility_records = []
    for index in range(150):
        utility_records.append({
            "clean": {"hit_at_retrieve_k": index < 148, "hit_at_serve_k": index < 121},
            "poisoned": {"hit_at_retrieve_k": index < 148, "hit_at_serve_k": index < 120},
            "complete_lineage_repair": {"hit_at_retrieve_k": index < 148, "hit_at_serve_k": index < 121},
        })
    utility_path = tmp_path / "utility.json"
    utility_path.write_text(json.dumps({"records": utility_records}), encoding="utf-8")
    manifest = {
        "annotation_source_sha256": locked_sources,
        "annotation_key_sha256": key_hash,
        "analysis_artifact_sha256": {"repair_lattice": digest(repair_path), "global_utility": digest(utility_path)},
        "expected_counts": {
            "annotation_outputs": 846,
            "substitution_incidents": 94,
            "boundary_incidents": 50,
            "repair_incidents": 30,
            "utility_queries": 150,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_path = tmp_path / "final.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "finalize_human_semantic_analysis.py"
    command = [
        sys.executable, str(script), "--agreement", str(agreement_path),
        "--boundary-artifact", str(boundary_path),
        "--repair-artifact", str(repair_path), "--utility", str(utility_path),
        "--repair-certificate", str(certificate_path),
        "--repair-certificate-verification", str(verification_path),
        "--repair-certificate-manifest", str(certificate_manifest_path),
        "--repair-certificate-runner-script", str(certificate_runner_path),
        "--repair-certificate-verifier-script", str(certificate_verifier_path),
        "--locked-manifest", str(manifest_path), "--output", str(output_path),
        "--bootstrap-samples", "100",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["paper_evidence"] is True
    assert result["primary_hypotheses"]["substitution"]["eligible"] == 94
    assert result["primary_hypotheses"]["substitution"]["paired_table"]["partial_only"] == 10
    assert result["primary_hypotheses"]["boundary"]["paired_table"]["partial_only"] == 6
    assert result["boundary_mechanism"]["migration_all"]["events"] == 23
    assert result["boundary_mechanism"]["partial_recurrence_with_migration"] == 6
    assert result["boundary_mechanism"]["all_partial_recurrences_have_migration"] is True
    assert result["repair_lattice"]["reindexrepair_feasible"] == 30
    assert result["repair_lattice"]["certified_exact_view_oracle_recall_count"] == 30
    assert result["repair_certificate_gate"]["action_certificates"] == 120
    assert result["repair_certificate_gate"]["reference_answer_matches"] == 270

    macros_path = tmp_path / "results.tex"
    macro_script = Path(__file__).resolve().parents[1] / "scripts" / "render_human_paper_inputs.py"
    macro_run = subprocess.run([
        sys.executable, str(macro_script), "--statistics", str(output_path), "--output", str(macros_path),
    ], capture_output=True, text=True)
    assert macro_run.returncode == 0, macro_run.stderr
    macros = macros_path.read_text(encoding="utf-8")
    assert r"\newcommand{\EvidenceBanner}{}" in macros
    assert r"\newcommand{\SubPartialCount}{12/94}" in macros
    assert r"\newcommand{\BoundaryPartialCount}{6/50}" in macros
    assert r"\newcommand{\BoundaryMechanismSentence}" in macros

    figure_path = tmp_path / "human_results.pdf"
    figure_script = Path(__file__).resolve().parents[1] / "scripts" / "make_paper_figures.py"
    locator_path = Path(__file__).resolve().parents[1] / "artifacts" / "remote" / "artifacts" / "locator_error_sweep_100k_v1.json"
    if not locator_path.exists():
        locator_path = Path(__file__).resolve().parents[1] / "artifacts" / "locator_error_sweep_100k_v1.json"
    figure_run = subprocess.run([
        sys.executable, str(figure_script), "--human-statistics", str(output_path),
        "--locator-sweep", str(locator_path), "--output", str(figure_path),
    ], capture_output=True, text=True)
    assert figure_run.returncode == 0, figure_run.stderr
    assert figure_path.exists()
    assert figure_path.with_suffix(".png").exists()

    certificate["summary"]["all_independent_f1_f2_certificates_pass"] = False
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["artifact_sha256"] = digest(certificate_path)
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    certificate_manifest = json.loads(certificate_manifest_path.read_text(encoding="utf-8"))
    certificate_manifest["independent_certificate_sha256"] = digest(certificate_path)
    certificate_manifest["server_verification_sha256"] = digest(verification_path)
    certificate_manifest_path.write_text(json.dumps(certificate_manifest), encoding="utf-8")
    certificate_rejected = subprocess.run(command, capture_output=True, text=True)
    assert certificate_rejected.returncode != 0
    assert "independent F1/F2 certificate has a failed action" in certificate_rejected.stderr
    certificate["summary"]["all_independent_f1_f2_certificates_pass"] = True
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")
    verification["artifact_sha256"] = digest(certificate_path)
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    certificate_manifest["independent_certificate_sha256"] = digest(certificate_path)
    certificate_manifest["server_verification_sha256"] = digest(verification_path)
    certificate_manifest_path.write_text(json.dumps(certificate_manifest), encoding="utf-8")

    verification["action_certificates"] = 119
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    certificate_manifest["server_verification_sha256"] = digest(verification_path)
    certificate_manifest_path.write_text(json.dumps(certificate_manifest), encoding="utf-8")
    invalid_verification_count = subprocess.run(command, capture_output=True, text=True)
    assert invalid_verification_count.returncode != 0
    assert "unexpected action_certificates" in invalid_verification_count.stderr
    verification["action_certificates"] = 120

    verification["query_row_ids_sha256"] = "0" * 64
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    certificate_manifest["server_verification_sha256"] = digest(verification_path)
    certificate_manifest_path.write_text(json.dumps(certificate_manifest), encoding="utf-8")
    invalid_query_alignment = subprocess.run(command, capture_output=True, text=True)
    assert invalid_query_alignment.returncode != 0
    assert "query-ID alignment hash mismatch" in invalid_query_alignment.stderr
    verification["query_row_ids_sha256"] = canonical_digest(
        [row["incident_id"] for row in repair_artifact_rows]
    )
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    certificate_manifest["server_verification_sha256"] = digest(verification_path)
    certificate_manifest_path.write_text(json.dumps(certificate_manifest), encoding="utf-8")

    certificate_runner_path.write_text("# tampered runner\n", encoding="utf-8")
    invalid_runner_hash = subprocess.run(command, capture_output=True, text=True)
    assert invalid_runner_hash.returncode != 0
    assert "runner script" in invalid_runner_hash.stderr
    certificate_runner_path.write_text("# frozen runner\n", encoding="utf-8")

    manifest["annotation_source_sha256"]["boundary_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    boundary_hash_mismatch = subprocess.run(command, capture_output=True, text=True)
    assert boundary_hash_mismatch.returncode != 0
    assert "agreement annotation-source hashes do not match" in boundary_hash_mismatch.stderr
    manifest["annotation_source_sha256"]["boundary_sha256"] = digest(boundary_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    manifest["analysis_artifact_sha256"]["global_utility"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    hash_mismatch = subprocess.run(command, capture_output=True, text=True)
    assert hash_mismatch.returncode != 0
    assert "utility artifact does not match" in hash_mismatch.stderr
    manifest["analysis_artifact_sha256"]["global_utility"] = digest(utility_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    utility_records.pop()
    utility_path.write_text(json.dumps({"records": utility_records}), encoding="utf-8")
    manifest["analysis_artifact_sha256"]["global_utility"] = digest(utility_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    invalid = subprocess.run(command, capture_output=True, text=True)
    assert invalid.returncode != 0
    assert "150-query utility suite" in invalid.stderr
