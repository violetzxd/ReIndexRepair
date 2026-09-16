from __future__ import annotations

import argparse
import getpass
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.error

from semantic_adjudicate_pair_deepseek import RUBRIC, call_api, validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--budget-rmb", type=float, default=78.2)
    parser.add_argument("--input-rmb-per-million", type=float, default=2.0)
    parser.add_argument("--output-rmb-per-million", type=float, default=8.0)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    canonical = []
    for row in source["records"]:
        shared = {"question": row["question"], "correct_answer": row["correct_answer"], "target_answer": row["target_answer"]}
        canonical.append({"item_id": f"{row['incident_id']}|initial", **shared, "response": row["initial"]["answer"]})
        for action, details in row["actions"].items():
            canonical.append({"item_id": f"{row['incident_id']}|f0|{action}", **shared, "response": details["f0_trace"]["answer"]})
            canonical.append({"item_id": f"{row['incident_id']}|f2|{action}", **shared, "response": details["f2_physical_rebuild"]["answer"]})
    api_key = getpass.getpass("DeepSeek API key: ")
    total_prompt = total_completion = 0
    calls = []
    passes = []
    for pass_index in (1, 2):
        ordered = canonical.copy()
        random.Random(4027 + pass_index).shuffle(ordered)
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
    keys = ("target_adopted", "correct_adopted", "complete")
    disagreements = [item for item in canonical if any(passes[0][item["item_id"]][key] != passes[1][item["item_id"]][key] for key in keys)]
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
    cost = (total_prompt * args.input_rmb_per_million + total_completion * args.output_rmb_per_million) / 1_000_000
    if cost > args.budget_rmb:
        raise RuntimeError("semantic audit exceeded remaining project budget")
    payload = {
        "paper_evidence": False,
        "status": "dual_blinded_llm_semantic_audit_not_human_annotation",
        "rubric": RUBRIC,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "summary": {
            "incidents": len(source["records"]), "outputs_judged": len(canonical),
            "binary_label_agreement": sum(passes[0][item["item_id"]][key] == passes[1][item["item_id"]][key] for item in canonical for key in keys) / (len(canonical) * len(keys)),
            "outputs_with_any_disagreement": len(disagreements),
            "incomplete_outputs": sum(not final[item["item_id"]]["complete"] for item in canonical),
        },
        "records": [{**item, "blind_pass_1": passes[0][item["item_id"]], "blind_pass_2": passes[1][item["item_id"]], "adjudication": adjudicated.get(item["item_id"]), "final": final[item["item_id"]]} for item in canonical],
    }
    ledger = {
        "budget_rmb_for_this_run": args.budget_rmb,
        "conservative_pricing_rmb_per_million": {"input": args.input_rmb_per_million, "output": args.output_rmb_per_million},
        "prompt_tokens": total_prompt, "completion_tokens": total_completion, "conservative_cost_rmb": cost, "calls": calls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    args.ledger.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "cost_rmb": cost}, indent=2))


if __name__ == "__main__":
    main()
