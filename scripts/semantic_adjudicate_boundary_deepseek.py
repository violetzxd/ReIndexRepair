from __future__ import annotations

import argparse
import getpass
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.error

from semantic_adjudicate_pair_deepseek import RUBRIC, call_api, rate, validate


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--budget-rmb", type=float, default=78.4)
    parser.add_argument("--input-rmb-per-million", type=float, default=2.0)
    parser.add_argument("--output-rmb-per-million", type=float, default=8.0)
    args = parser.parse_args()
    artifact = json.loads(args.input.read_text(encoding="utf-8"))
    canonical = []
    for row in artifact["records"]:
        answers = {
            "initial": row["initial"]["answer"],
            "f0": row["f0_trace_mask"]["answer"],
            "rebuilt": row["f2_rechunk_rebuild"]["answer"],
        }
        if "complete_lineage_quarantine" in row:
            answers["complete_lineage"] = row["complete_lineage_quarantine"]["answer"]
        for condition, response in answers.items():
            canonical.append({
                "item_id": f"{row['incident_id']}|{condition}",
                "question": row["question"],
                "correct_answer": row["correct_answer"],
                "target_answer": row["target_answer"],
                "response": response,
            })
    api_key = getpass.getpass("DeepSeek API key: ")
    total_prompt = total_completion = 0
    calls = []
    passes = []
    for pass_index in (1, 2):
        ordered = canonical.copy()
        random.Random(3027 + pass_index).shuffle(ordered)
        labels = {}
        for start in range(0, len(ordered), args.batch_size):
            batch = ordered[start : start + args.batch_size]
            for attempt in range(3):
                try:
                    response = call_api(api_key, args.base_url, args.model, batch, pass_index)
                    received = validate(response["choices"][0]["message"]["content"], {item["item_id"] for item in batch})
                    break
                except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
            usage = response.get("usage", {})
            total_prompt += int(usage.get("prompt_tokens", 0))
            total_completion += int(usage.get("completion_tokens", 0))
            labels.update(received)
            calls.append({"stage": f"blind_pass_{pass_index}", "batch_start": start, "batch_size": len(batch), "usage": usage})
            print(f"pass {pass_index}: {len(labels)}/{len(canonical)}")
        passes.append(labels)
    decision_keys = ("target_adopted", "correct_adopted", "complete")
    disagreements = [item for item in canonical if any(passes[0][item["item_id"]][key] != passes[1][item["item_id"]][key] for key in decision_keys)]
    adjudicated = {}
    for start in range(0, len(disagreements), args.batch_size):
        batch = disagreements[start : start + args.batch_size]
        if not batch:
            continue
        response = call_api(api_key, args.base_url, args.model, batch, 2)
        received = validate(response["choices"][0]["message"]["content"], {item["item_id"] for item in batch})
        usage = response.get("usage", {})
        total_prompt += int(usage.get("prompt_tokens", 0))
        total_completion += int(usage.get("completion_tokens", 0))
        adjudicated.update(received)
        calls.append({"stage": "disagreement_adjudication", "batch_start": start, "batch_size": len(batch), "usage": usage})
    final = {item["item_id"]: adjudicated.get(item["item_id"], passes[0][item["item_id"]]) for item in canonical}
    by_incident = {}
    for item in canonical:
        incident_id, condition = item["item_id"].split("|", 1)
        by_incident.setdefault(incident_id, {})[condition] = final[item["item_id"]]
    attacked = {item_id: labels for item_id, labels in by_incident.items() if labels["initial"]["target_adopted"]}
    eligible = {item_id: labels for item_id, labels in attacked.items() if not labels["f0"]["target_adopted"]}
    mirages = sum(labels["rebuilt"]["target_adopted"] for labels in eligible.values())
    has_control = all("complete_lineage" in labels for labels in by_incident.values())
    control_mirages = sum(labels["complete_lineage"]["target_adopted"] for labels in eligible.values()) if has_control else None
    paired = None
    if has_control:
        paired = {
            "rebuilt_only": sum(labels["rebuilt"]["target_adopted"] and not labels["complete_lineage"]["target_adopted"] for labels in eligible.values()),
            "complete_lineage_only": sum(not labels["rebuilt"]["target_adopted"] and labels["complete_lineage"]["target_adopted"] for labels in eligible.values()),
            "both": sum(labels["rebuilt"]["target_adopted"] and labels["complete_lineage"]["target_adopted"] for labels in eligible.values()),
            "neither": sum(not labels["rebuilt"]["target_adopted"] and not labels["complete_lineage"]["target_adopted"] for labels in eligible.values()),
        }
    total_cost = (total_prompt * args.input_rmb_per_million + total_completion * args.output_rmb_per_million) / 1_000_000
    if total_cost > args.budget_rmb:
        raise RuntimeError("semantic audit exceeded the remaining RMB budget")
    summary = {
        "incidents": len(by_incident),
        "outputs_judged": len(canonical),
        "binary_label_agreement": sum(passes[0][item["item_id"]][key] == passes[1][item["item_id"]][key] for item in canonical for key in decision_keys) / (len(canonical) * len(decision_keys)),
        "outputs_with_any_disagreement": len(disagreements),
        "initial_target_adopted": rate(len(attacked), len(by_incident)),
        "eligible_initial_target_and_f0_safe": len(eligible),
        "repair_mirage": rate(mirages, len(eligible)),
        "unconditional_mirages": rate(mirages, len(by_incident)),
        "complete_lineage_recurrence": rate(control_mirages, len(eligible)) if has_control else None,
        "paired_rebuilt_vs_complete_lineage": paired,
        "incomplete_outputs": sum(not label["complete"] for label in final.values()),
    }
    payload = {
        "paper_evidence": False,
        "status": "dual_blinded_llm_semantic_audit_not_human_annotation",
        "rubric": RUBRIC,
        "input_sha256": sha256_file(args.input),
        "model": args.model,
        "summary": summary,
        "records": [{**item, "blind_pass_1": passes[0][item["item_id"]], "blind_pass_2": passes[1][item["item_id"]], "adjudication": adjudicated.get(item["item_id"]), "final": final[item["item_id"]]} for item in canonical],
    }
    ledger = {
        "budget_rmb_for_this_run": args.budget_rmb,
        "conservative_pricing_rmb_per_million": {"input": args.input_rmb_per_million, "output": args.output_rmb_per_million},
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "conservative_cost_rmb": total_cost,
        "calls": calls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    args.ledger.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": summary, "cost_rmb": total_cost}, indent=2))


if __name__ == "__main__":
    main()
