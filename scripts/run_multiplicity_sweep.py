from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from run_candidate_set_pilot import (
    MODEL_NAME,
    MODEL_REVISION,
    QUERY_PREFIX,
    rank_indices,
    sha256_file,
)


def select_poison_subset(
    incident_id: str, poison_indices: list[int], multiplicity: int, seed: int
) -> set[int]:
    ordered = sorted(
        poison_indices,
        key=lambda index: hashlib.sha256(
            f"{seed}\0{incident_id}\0{index}".encode("utf-8")
        ).digest(),
    )
    return set(ordered[:multiplicity])


def incident_bootstrap_interval(
    per_incident_values: dict[str, list[float]], *, samples: int, seed: int
) -> tuple[float, float]:
    incident_ids = sorted(per_incident_values)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        selected = rng.choice(incident_ids, size=len(incident_ids), replace=True)
        means[draw] = np.mean(
            [np.mean(per_incident_values[str(incident_id)]) for incident_id in selected]
        )
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "external/RAG-Responsibility-Attribution/attack_feedback/PRAGB/"
            "k5_m5_e5_gpt-4o-mini.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/multiplicity_sweep.json")
    )
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--multiplicity", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--subset-seed", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", default=MODEL_NAME)
    args = parser.parse_args()

    multiplicities = sorted(set(args.multiplicity))
    top_ks = sorted(set(args.top_k))
    subset_seeds = sorted(set(args.subset_seed))
    if multiplicities[0] < 1 or multiplicities[-1] > 5:
        raise ValueError("multiplicity must lie in [1, 5] for this source artifact")
    if top_ks[0] < 1 or top_ks[-1] > args.candidate_limit:
        raise ValueError("top-k must lie within the candidate pool")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer

    records = json.loads(args.input.read_text(encoding="utf-8"))
    questions = [QUERY_PREFIX + item["question"] for item in records]
    passages: list[str] = []
    passage_slices: list[tuple[int, int]] = []
    labels: list[list[bool]] = []
    for item in records:
        texts = item["context_texts"][: args.candidate_limit]
        item_labels = [bool(value) for value in item["context_labels"][: args.candidate_limit]]
        if len(texts) != args.candidate_limit or sum(item_labels) != 5:
            raise ValueError("each candidate pool must contain 100 candidates and five poisons")
        start = len(passages)
        passages.extend(texts)
        passage_slices.append((start, len(passages)))
        labels.append(item_labels)

    started = time.perf_counter()
    model = SentenceTransformer(
        args.model_path,
        revision=MODEL_REVISION if args.model_path == MODEL_NAME else None,
        device=args.device,
    )
    query_vectors = model.encode(
        questions,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    passage_vectors = model.encode(
        passages,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    cell_records: dict[str, list[dict]] = defaultdict(list)
    for incident_index, item in enumerate(records):
        start, end = passage_slices[incident_index]
        full_ranking = rank_indices(
            query_vectors[incident_index], passage_vectors[start:end]
        )
        item_labels = labels[incident_index]
        poison_indices = [index for index, label in enumerate(item_labels) if label]
        for subset_seed in subset_seeds:
            for multiplicity in multiplicities:
                selected_poison = select_poison_subset(
                    item["question_id"], poison_indices, multiplicity, subset_seed
                )
                active_ranking = [
                    index
                    for index in full_ranking
                    if not item_labels[index] or index in selected_poison
                ]
                for top_k in top_ks:
                    visible = active_ranking[:top_k]
                    visible_poison = {
                        index for index in visible if index in selected_poison
                    }
                    retrieval_positive = bool(visible_poison)
                    remaining = [
                        index for index in active_ranking if index not in visible_poison
                    ]
                    promoted = retrieval_positive and any(
                        index in selected_poison for index in remaining[:top_k]
                    )
                    cell_records[f"m{multiplicity}_k{top_k}"].append(
                        {
                            "incident_id": item["question_id"],
                            "subset_seed": subset_seed,
                            "retrieval_poison_present": retrieval_positive,
                            "poison_promoted": promoted,
                            "selected_poison_indices": sorted(selected_poison),
                            "visible_indices": visible,
                            "post_removal_indices": remaining[:top_k],
                        }
                    )

    summaries: dict[str, dict] = {}
    for cell, values in sorted(cell_records.items()):
        eligible = [value for value in values if value["retrieval_poison_present"]]
        nested: dict[str, list[float]] = defaultdict(list)
        by_seed: dict[str, list[bool]] = defaultdict(list)
        for value in eligible:
            nested[value["incident_id"]].append(float(value["poison_promoted"]))
            by_seed[str(value["subset_seed"])].append(value["poison_promoted"])
        interval = (
            incident_bootstrap_interval(
                nested, samples=args.bootstrap_samples, seed=20270912
            )
            if nested
            else None
        )
        summaries[cell] = {
            "eligible_incident_seed_pairs": len(eligible),
            "unique_incidents": len(nested),
            "incident_averaged_poison_promotion_rate": (
                float(np.mean([np.mean(items) for items in nested.values()]))
                if nested
                else None
            ),
            "incident_cluster_bootstrap_95_interval": (
                list(interval) if interval is not None else None
            ),
            "per_subset_seed_rate": {
                seed: float(np.mean(items)) for seed, items in sorted(by_seed.items())
            },
        }

    payload = {
        "tier": "diagnostic_multiplicity_depth_sweep",
        "paper_evidence": False,
        "limitations": [
            "Uses preselected successful misgeneration incidents, not all attack attempts.",
            "Uses a restricted candidate set, not full-corpus retrieval or reranking.",
            "Measures oracle-labelled poison promotion, not F0/F2 generator outcomes.",
            "Artificially subsamples the five supplied poison passages to vary multiplicity.",
        ],
        "input": {
            "path": str(args.input),
            "sha256": sha256_file(args.input),
            "candidate_limit": args.candidate_limit,
            "incidents": len(records),
        },
        "design": {
            "multiplicity": multiplicities,
            "top_k": top_ks,
            "subset_seed": subset_seeds,
            "poison_subset_rule": "lowest SHA-256(seed, incident_id, candidate_index)",
            "bootstrap_unit": "incident",
            "bootstrap_samples": args.bootstrap_samples,
        },
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "load_path": args.model_path,
            "query_prefix": QUERY_PREFIX,
            "normalize_embeddings": True,
            "device": args.device,
        },
        "runtime": {
            "seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "summary": summaries,
        "records": dict(cell_records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
