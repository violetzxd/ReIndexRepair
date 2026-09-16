from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from openpyxl import Workbook


HEADERS = [
    "Item code", "Question", "Correct reference", "Target answer", "System response",
    "Target adopted?", "Correct adopted?", "Complete?", "Confidence", "Brief note",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def save_labels(path: Path, rows: list[list[object]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Labels"
    sheet.append(HEADERS)
    for row in rows:
        sheet.append(row)
    workbook.save(path)


def test_canonical_b_relabel_closes_source_mismatch_and_restores_846_item_agreement(tmp_path: Path) -> None:
    partial_rows = []
    complete_rows = []
    for index in range(94):
        incident = f"main-{index:03d}"
        shared = {
            "incident_id": incident,
            "question": f"Question {index}?",
            "correct_answer": f"Correct {index}",
            "target_answer": f"Target {index}",
            "initial": {"answer": f"Initial {index}"},
            "f0_trace_mask": {"answer": f"F0 {index}"},
        }
        partial_rows.append({**shared, "f2_fresh_exact_rebuild": {"answer": f"Partial F2 {index}"}})
        complete_rows.append({**shared, "f2_fresh_exact_rebuild": {"answer": f"Complete F2 {index}"}})

    boundary_rows = []
    for index in range(50):
        boundary_rows.append({
            "incident_id": f"boundary-{index:03d}",
            "question": f"Boundary question {index}?",
            "correct_answer": f"Boundary correct {index}",
            "target_answer": f"Boundary target {index}",
            "initial": {"answer": f"Boundary initial {index}"},
            "f0_trace_mask": {"answer": f"Boundary F0 {index}"},
            "f2_rechunk_rebuild": {"answer": f"Boundary rebuilt {index}"},
            "complete_lineage_quarantine": {"answer": f"Boundary complete {index}"},
        })

    actions = ["target_span", "visible_sentence", "all_attack_sentences", "document_quarantine"]
    repair_rows = []
    for index in range(30):
        repair_rows.append({
            "incident_id": f"repair-{index:03d}",
            "question": f"Repair question {index}?",
            "correct_answer": f"Repair correct {index}",
            "target_answer": f"Repair target {index}",
            "initial": {"answer": f"Repair initial {index}"},
            "actions": {
                action: {
                    "f0_trace": {"answer": f"Repair {index} {action} F0"},
                    "f2_physical_rebuild": {"answer": f"Repair {index} {action} F2"},
                }
                for action in actions
            },
        })

    partial_path = tmp_path / "partial.json"
    complete_path = tmp_path / "complete.json"
    boundary_path = tmp_path / "boundary.json"
    repair_path = tmp_path / "repair.json"
    write_json(partial_path, {"records": partial_rows})
    write_json(complete_path, {"records": complete_rows})
    write_json(boundary_path, {"records": boundary_rows})
    write_json(repair_path, {"records": repair_rows})

    canonical: list[tuple[str, str, str, str, str]] = []
    for index in range(94):
        incident = f"main-{index:03d}"
        shared = (f"Question {index}?", f"Correct {index}", f"Target {index}")
        canonical.extend([
            (f"{incident}|initial", *shared, f"Initial {index}"),
            (f"{incident}|f0", *shared, f"F0 {index}"),
            (f"{incident}|partial_f2", *shared, f"Partial F2 {index}"),
            (f"{incident}|complete_f2", *shared, f"Complete F2 {index}"),
        ])
    for index in range(50):
        incident = f"boundary-{index:03d}"
        shared = (f"Boundary question {index}?", f"Boundary correct {index}", f"Boundary target {index}")
        canonical.extend([
            (f"boundary|{incident}|initial", *shared, f"Boundary initial {index}"),
            (f"boundary|{incident}|f0", *shared, f"Boundary F0 {index}"),
            (f"boundary|{incident}|rebuilt", *shared, f"Boundary rebuilt {index}"),
            (f"boundary|{incident}|complete_lineage", *shared, f"Boundary complete {index}"),
        ])
    for index in range(30):
        incident = f"repair-{index:03d}"
        shared = (f"Repair question {index}?", f"Repair correct {index}", f"Repair target {index}")
        canonical.append((f"repair|{incident}|initial", *shared, f"Repair initial {index}"))
        for action in actions:
            canonical.extend([
                (f"repair|{incident}|{action}|f0", *shared, f"Repair {index} {action} F0"),
                (f"repair|{incident}|{action}|f2", *shared, f"Repair {index} {action} F2"),
            ])
    assert len(canonical) == 846

    mapping_a = {f"A-{index:04d}": item[0] for index, item in enumerate(canonical)}
    mapping_b = {f"B-{index:04d}": item[0] for index, item in enumerate(canonical)}
    rows_a = []
    rows_b = []
    mismatch_item = canonical[0][0]
    mismatch_code = "B-0000"
    for index, (_, question, correct, target, response) in enumerate(canonical):
        labels = ["No", "Yes", "Yes", "High", ""]
        rows_a.append([f"A-{index:04d}", question, correct, target, response, *labels])
        b_response = "Wrong transcribed response" if index == 0 else response
        rows_b.append([f"B-{index:04d}", question, correct, target, b_response, *labels])
    a_path = tmp_path / "a.xlsx"
    b_path = tmp_path / "b.xlsx"
    save_labels(a_path, rows_a)
    save_labels(b_path, rows_b)

    key_path = tmp_path / "key.json"
    write_json(key_path, {
        "source": {
            "partial_sha256": digest(partial_path),
            "complete_lineage_sha256": digest(complete_path),
            "boundary_sha256": digest(boundary_path),
            "repair_sha256": digest(repair_path),
        },
        "A": mapping_a,
        "B": mapping_b,
    })
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_human_annotation_workbooks.py"
    common = [
        sys.executable, str(script), "--annotator-a", str(a_path), "--annotator-b", str(b_path),
        "--key", str(key_path), "--partial-artifact", str(partial_path),
        "--complete-lineage-artifact", str(complete_path), "--boundary-artifact", str(boundary_path),
        "--repair-artifact", str(repair_path),
    ]
    v1_path = tmp_path / "agreement_v1.json"
    first = subprocess.run([*common, "--output", str(v1_path)], capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    assert v1["status"] == "requires_annotator_b_relabel"
    assert v1["agreement_items"] == 845
    assert [row["item_id"] for row in v1["relabel_required"]] == [mismatch_item]

    relabel_path = tmp_path / "b_relabel.xlsx"
    _, question, correct, target, response = canonical[0]
    save_labels(relabel_path, [["BR-ONE", question, correct, target, response, "No", "Yes", "Yes", "High", ""]])
    relabel_key_path = tmp_path / "relabel_key.json"
    write_json(relabel_key_path, {
        "agreement_sha256": digest(v1_path),
        "mapping": {"BR-ONE": mismatch_item},
        "original_b_codes": {mismatch_item: mismatch_code},
    })
    v2_path = tmp_path / "agreement_v2.json"
    second = subprocess.run([
        *common, "--annotator-b-relabel", str(relabel_path), "--relabel-key", str(relabel_key_path),
        "--relabel-agreement", str(v1_path), "--output", str(v2_path),
    ], capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    assert v2["status"] == "no_disagreements"
    assert v2["agreement_items"] == 846
    assert v2["relabel_required"] == []
    assert v2["source_integrity"]["items_with_any_stimulus_mismatch"] == 0
    assert v2["source_integrity"]["annotator_b_items_relabelled_from_canonical"] == 1

    save_labels(relabel_path, [["BR-ONE", question, correct, target, response, "Yes", "Yes", "Yes", "High", ""]])
    v2_disagreement_path = tmp_path / "agreement_v2_disagreement.json"
    third = subprocess.run([
        *common, "--annotator-b-relabel", str(relabel_path), "--relabel-key", str(relabel_key_path),
        "--relabel-agreement", str(v1_path), "--output", str(v2_disagreement_path),
    ], capture_output=True, text=True)
    assert third.returncode == 0, third.stderr
    v2_disagreement = json.loads(v2_disagreement_path.read_text(encoding="utf-8"))
    assert v2_disagreement["status"] == "requires_third_person_adjudication"
    assert v2_disagreement["agreement_items"] == 846
    assert [row["item_id"] for row in v2_disagreement["disagreements"]] == [mismatch_item]
    assert v2_disagreement["disagreements"][0]["disputed_fields"] == ["target_adopted"]
