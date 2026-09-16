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

from reindexrepair.metrics import clopper_pearson_interval


MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_indices(query_vector: np.ndarray, document_vectors: np.ndarray) -> list[int]:
    scores = document_vectors @ query_vector
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))


def rate_record(recurrences: int, trials: int) -> dict:
    if trials == 0:
        return {
            "recurrences": 0,
            "trials": 0,
            "rate": None,
            "two_sided_95_exact_interval": None,
        }
    interval = clopper_pearson_interval(recurrences, trials)
    return {
        "recurrences": recurrences,
        "trials": trials,
        "rate": recurrences / trials,
        "two_sided_95_exact_interval": list(interval),
    }


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
    parser.add_argument("--output", type=Path, default=Path("artifacts/candidate_set_pilot.json"))
    parser.add_argument("--candidate-limit", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--model-path", default=MODEL_NAME)
    args = parser.parse_args()

    if args.candidate_limit < 5:
        raise ValueError("candidate-limit must include the five poison passages")
    top_ks = tuple(sorted(set(args.top_k)))
    if not top_ks or top_ks[0] < 1 or top_ks[-1] > args.candidate_limit:
        raise ValueError("each top-k must be between 1 and candidate-limit")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("expected a list of attack feedback records")

    questions = [QUERY_PREFIX + item["question"] for item in records]
    passages: list[str] = []
    passage_slices: list[tuple[int, int]] = []
    labels: list[list[bool]] = []
    for item in records:
        texts = item["context_texts"][: args.candidate_limit]
        item_labels = [bool(value) for value in item["context_labels"][: args.candidate_limit]]
        if len(texts) != args.candidate_limit or len(texts) != len(item_labels):
            raise ValueError("candidate set is shorter than requested")
        start = len(passages)
        passages.extend(texts)
        passage_slices.append((start, len(passages)))
        labels.append(item_labels)

    started = time.perf_counter()
    model = SentenceTransformer(
        args.model_path,
        revision=MODEL_REVISION if args.model_path == MODEL_NAME else None,
        device="cpu",
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
    encode_seconds = time.perf_counter() - started

    results_by_k: dict[str, list[dict]] = defaultdict(list)
    for incident_index, item in enumerate(records):
        start, end = passage_slices[incident_index]
        ranking = rank_indices(query_vectors[incident_index], passage_vectors[start:end])
        item_labels = labels[incident_index]
        for top_k in top_ks:
            visible = ranking[:top_k]
            visible_poison = {index for index in visible if item_labels[index]}
            poison_retrieved = bool(visible_poison)
            remaining = [index for index in ranking if index not in visible_poison]
            post_rebuild = remaining[:top_k]
            recurrence = poison_retrieved and any(
                item_labels[index] for index in post_rebuild
            )

            # Logic-determined sanity check: remove every oracle-labelled hidden
            # poison before deleting the visible poisons. A zero here is not an
            # empirical negative control and must not be assigned a confidence
            # interval or used as paper evidence.
            control_corpus = [
                index
                for index in ranking
                if (not item_labels[index]) or index in visible_poison
            ]
            control_after_delete = [
                index for index in control_corpus if index not in visible_poison
            ][:top_k]
            control_recurrence = poison_retrieved and any(
                item_labels[index] for index in control_after_delete
            )
            results_by_k[str(top_k)].append(
                {
                    "incident_id": item["question_id"],
                    "poison_retrieved": poison_retrieved,
                    "visible_poison_count": len(visible_poison),
                    "recurrence": recurrence,
                    "control_recurrence": control_recurrence,
                    "visible_indices": visible,
                    "post_rebuild_indices": post_rebuild,
                }
            )

    summaries: dict[str, dict] = {}
    for top_k, values in results_by_k.items():
        retrieval_positive = [value for value in values if value["poison_retrieved"]]
        sanity_failures = sum(
            value["control_recurrence"] for value in retrieval_positive
        )
        summaries[top_k] = {
            "preselected_successful_misgeneration_incidents": len(values),
            "retrieval_poison_present": len(retrieval_positive),
            "retrieval_poison_rate": len(retrieval_positive) / len(values),
            "retrieval_substitution_proxy": rate_record(
                sum(value["recurrence"] for value in retrieval_positive),
                len(retrieval_positive),
            ),
            "oracle_no_backup_sanity_check": {
                "failures": sanity_failures,
                "trials": len(retrieval_positive),
                "by_construction": True,
                "empirical_control": False,
            },
        }

    payload = {
        "tier": "pilot_retrieval_candidate_set",
        "paper_evidence": False,
        "limitations": [
            f"Ranks only the {args.candidate_limit} supplied candidates per incident, not the full corpus.",
            "Measures poison retrieval recurrence, not generator attack success.",
            "Uses oracle poison labels and document removal; it is a go/no-go probe.",
            "The oracle no-backup check is zero by construction and is not an empirical control.",
            "The source artifact contains successful top-5 attacks with exactly five poison passages per incident.",
        ],
        "input": {
            "path": str(args.input),
            "sha256": sha256_file(args.input),
            "record_count": len(records),
            "candidate_limit": args.candidate_limit,
            "top_k": list(top_ks),
        },
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "load_path": args.model_path,
            "query_prefix": QUERY_PREFIX,
            "normalize_embeddings": True,
            "device": "cpu",
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "encode_seconds": encode_seconds,
        },
        "summary": summaries,
        "records": dict(results_by_k),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
