from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from run_boundary_migration_cert import build_boundary_source, chunk_ids
from run_full_pipeline_dev import (
    EMBEDDER_NAME,
    EMBEDDER_REVISION,
    GENERATOR_NAME,
    GENERATOR_REVISION,
    QUERY_PREFIX,
    RERANKER_NAME,
    RERANKER_REVISION,
    contains_answer,
    document_text,
    exact_top_k,
    fingerprint_path,
    generate_answers,
    make_prompt,
    read_jsonl,
    sha256_file,
)


ACTION_NAMES = ("target_span", "visible_sentence", "all_attack_sentences", "document_quarantine")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_bytes(list(array.shape)))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def first_subsequence(values: list[int], needle: list[int], limit: int) -> tuple[int, int]:
    for start in range(max(0, limit - len(needle) + 1)):
        if values[start : start + len(needle)] == needle:
            return start, start + len(needle)
    raise ValueError("target token sequence not found in visible attack")


def stable_faiss_top_k(index: Any, query: np.ndarray, ids: list[str], k: int) -> list[int]:
    # Search the full flat index so the boundary tie is resolved globally by stable ID.
    distances, offsets = index.search(np.ascontiguousarray(query.reshape(1, -1)), len(ids))
    ranked = sorted(
        zip(offsets[0].tolist(), distances[0].tolist()),
        key=lambda item: (-float(item[1]), ids[item[0]]),
    )
    return [offset for offset, _ in ranked[:k]]


def rerank(
    reranker: Any,
    question: str,
    dense: list[int],
    ids: list[str],
    texts: list[str],
    batch_size: int,
    serve_k: int,
) -> tuple[list[int], list[float]]:
    scores = np.asarray(
        reranker.predict(
            [(question, texts[index]) for index in dense],
            batch_size=batch_size,
            show_progress_bar=False,
        )
    ).reshape(-1)
    order = sorted(range(len(dense)), key=lambda offset: (-float(scores[offset]), ids[dense[offset]]))
    return [dense[offset] for offset in order[:serve_k]], [float(scores[offset]) for offset in order]


def vector_view(
    base_ids: list[str],
    base_texts: list[str],
    base_vectors: np.ndarray,
    owner: str,
    condition: str,
    source_texts: list[str],
    source_vectors: np.ndarray,
) -> tuple[list[str], list[str], np.ndarray]:
    source_ids = [f"repairsource::{owner}::{condition}::{index}" for index in range(len(source_texts))]
    combined_ids = base_ids + source_ids
    combined_texts = base_texts + source_texts
    combined_vectors = (
        np.concatenate([base_vectors, np.ascontiguousarray(source_vectors)], axis=0)
        if len(source_vectors)
        else base_vectors.copy()
    )
    return combined_ids, combined_texts, np.ascontiguousarray(combined_vectors, dtype=np.float32)


def retrieve_exact_view(
    reranker: Any,
    question: str,
    query: np.ndarray,
    ids: list[str],
    texts: list[str],
    vectors: np.ndarray,
    retrieve_k: int,
    serve_k: int,
    reranker_batch_size: int,
) -> dict[str, Any]:
    dense = exact_top_k(query, vectors, retrieve_k, stable_ids=ids)
    served, reranker_scores = rerank(
        reranker, question, dense, ids, texts, reranker_batch_size, serve_k
    )
    return {
        "dense_ids": [ids[index] for index in dense],
        "served_ids": [ids[index] for index in served],
        "served_texts": [texts[index] for index in served],
        "reranker_scores_ranked": reranker_scores,
        "source_vector_sha256": None,
        "combined_vector_sha256": array_sha256(vectors),
        "metadata_sha256": canonical_sha256(list(zip(ids, texts))),
        "retrieval_engine": "numpy_full_dot_lexsort_score_desc_stable_id_asc",
    }


def retrieve_physical_view(
    faiss: Any,
    reranker: Any,
    question: str,
    query: np.ndarray,
    ids: list[str],
    texts: list[str],
    vectors: np.ndarray,
    retrieve_k: int,
    serve_k: int,
    reranker_batch_size: int,
) -> dict[str, Any]:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))
    dense = stable_faiss_top_k(index, query, ids, retrieve_k)
    served, reranker_scores = rerank(
        reranker, question, dense, ids, texts, reranker_batch_size, serve_k
    )
    return {
        "dense_ids": [ids[item] for item in dense],
        "served_ids": [ids[item] for item in served],
        "served_texts": [texts[item] for item in served],
        "reranker_scores_ranked": reranker_scores,
        "source_vector_sha256": None,
        "combined_vector_sha256": array_sha256(vectors),
        "metadata_sha256": canonical_sha256(list(zip(ids, texts))),
        "retrieval_engine": "fresh_faiss_indexflatip_full_search_then_score_desc_stable_id_asc",
        "materialized_faiss_index": True,
        "ntotal": int(index.ntotal),
    }


def prompt_hash(question: str, texts: list[str]) -> str:
    return canonical_sha256(make_prompt(question, texts))


def without_texts(view: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in view.items() if key != "served_texts"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--incidents", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--embedder-path", required=True)
    parser.add_argument("--reranker-path", required=True)
    parser.add_argument("--generator-path", required=True)
    parser.add_argument("--clean-vector-cache", type=Path, required=True)
    parser.add_argument("--query-vector-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--retrieve-k", type=int, default=30)
    parser.add_argument("--serve-k", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--vector-atol", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import faiss
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)

    incidents = json.loads(args.incidents.read_text(encoding="utf-8"))[: args.limit]
    candidate_payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates_by_id = {item["query_id"]: item for item in candidate_payload["records"]}
    reference_payload = json.loads(args.reference_artifact.read_text(encoding="utf-8"))
    reference_by_id = {row["incident_id"]: row for row in reference_payload["records"]}
    corpus = list(read_jsonl(args.corpus))
    clean_ids = [str(item["_id"]) for item in corpus]
    clean_texts = [document_text(item) for item in corpus]
    clean_vectors_all = np.load(args.clean_vector_cache)
    query_vectors_all = np.load(args.query_vector_cache)
    if len(clean_vectors_all) != len(clean_ids):
        raise ValueError("clean vector cache does not match corpus")
    if len(query_vectors_all) < len(incidents):
        raise ValueError("query vector cache has too few rows")
    if {row["question_id"] for row in incidents} != set(reference_by_id):
        raise ValueError("reference artifact does not contain exactly the requested incident IDs")

    excluded_relevant = {
        str(document["_id"])
        for incident in incidents
        for document in candidates_by_id[incident["beir_query_id"]]["relevant_documents"]
    }
    keep = np.asarray([document_id not in excluded_relevant for document_id in clean_ids])
    base_ids = [document_id for document_id, retained in zip(clean_ids, keep) if retained]
    base_texts = [text for text, retained in zip(clean_texts, keep) if retained]
    base_vectors = np.ascontiguousarray(clean_vectors_all[keep].astype(np.float32))
    query_vectors = np.ascontiguousarray(query_vectors_all[: len(incidents)].astype(np.float32))
    base_bundle = {
        "count": len(base_ids),
        "ids_sha256": canonical_sha256(base_ids),
        "texts_sha256": canonical_sha256(base_texts),
        "vectors_sha256": array_sha256(base_vectors),
    }

    embedder_fingerprint = fingerprint_path(args.embedder_path)
    reranker_fingerprint = fingerprint_path(args.reranker_path)
    generator_fingerprint = fingerprint_path(args.generator_path)
    embedder = SentenceTransformer(args.embedder_path, device=args.device)
    tokenizer = embedder.tokenizer
    filler_ids = tokenizer.encode(
        "Administrative archive note with unrelated catalog details and neutral indexing metadata.",
        add_special_tokens=False,
    )

    materials: list[dict[str, Any]] = []
    f1_texts_all: list[str] = []
    f2_texts_all: list[str] = []
    for incident in incidents:
        relevant = candidates_by_id[incident["beir_query_id"]]["relevant_documents"][0]
        trusted_ids = tokenizer.encode(document_text(relevant), add_special_tokens=False)
        layout = build_boundary_source(tokenizer, incident, args.chunk_size, filler_ids)
        visible_len = int(layout["visible_tokens"])
        backup_len = int(layout["backup_tokens"])
        between_len = int(layout["filler_tokens_between"])
        pre_ids = list(layout["pre_ids"]) + trusted_ids
        target_ids = tokenizer.encode(incident["target_answer"], add_special_tokens=False)
        span_start, span_end = first_subsequence(pre_ids, target_ids, visible_len)
        backup_start = visible_len + between_len
        backup_end = backup_start + backup_len
        variants = {
            "initial": pre_ids,
            "target_span": pre_ids[:span_start] + pre_ids[span_end:],
            "visible_sentence": pre_ids[visible_len:],
            "all_attack_sentences": pre_ids[visible_len:backup_start] + pre_ids[backup_end:],
            "document_quarantine": [],
        }
        removed_intervals = {
            "target_span": [[span_start, span_end]],
            "visible_sentence": [[0, visible_len]],
            "all_attack_sentences": [[0, visible_len], [backup_start, backup_end]],
            "document_quarantine": [[0, len(pre_ids)]],
        }
        costs = {
            "target_span": span_end - span_start,
            "visible_sentence": visible_len,
            "all_attack_sentences": visible_len + backup_len,
            "document_quarantine": len(pre_ids),
        }
        f1_variants: dict[str, list[str]] = {}
        f2_variants: dict[str, list[str]] = {}
        f1_slices: dict[str, tuple[int, int]] = {}
        f2_slices: dict[str, tuple[int, int]] = {}
        for name, token_ids in variants.items():
            # The two materializations deliberately repeat chunking and decoding.
            f1_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunk_ids(list(token_ids), args.chunk_size)]
            f2_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunk_ids(list(token_ids), args.chunk_size)]
            f1_variants[name] = f1_chunks
            f2_variants[name] = f2_chunks
            f1_slices[name] = (len(f1_texts_all), len(f1_chunks))
            f2_slices[name] = (len(f2_texts_all), len(f2_chunks))
            f1_texts_all.extend(f1_chunks)
            f2_texts_all.extend(f2_chunks)
        owner = incident["question_id"]
        old_ids = [f"repairsource::{owner}::initial::{index}" for index in range(len(f1_variants["initial"]))]
        actions: dict[str, Any] = {}
        for name in ACTION_NAMES:
            new_ids = [f"repairsource::{owner}::{name}::{index}" for index in range(len(f1_variants[name]))]
            action_spec = {
                "action": name,
                "edit_tokens": costs[name],
                "removed_old_token_intervals_half_open": removed_intervals[name],
                "tie_break": ["edit_tokens_ascending", "action_name_lexicographic"],
            }
            lineage = {
                "source_root_id": f"repairsource::{owner}",
                "source_revision_before_sha256": canonical_sha256(pre_ids),
                "source_revision_after_sha256": canonical_sha256(variants[name]),
                "seed_kind": "source_root",
                "seed_node_ids": [f"repairsource::{owner}"],
                "invalidated_descendant_ids": old_ids,
                "materialized_descendant_ids": new_ids,
            }
            lineage["closure_sha256"] = canonical_sha256({
                "seed_node_ids": lineage["seed_node_ids"],
                "invalidated_descendant_ids": old_ids,
                "materialized_descendant_ids": new_ids,
            })
            actions[name] = {
                "action_spec": action_spec,
                "action_spec_sha256": canonical_sha256(action_spec),
                "f1_source_revision_sha256": canonical_sha256(list(variants[name])),
                "f2_source_revision_sha256": canonical_sha256(list(variants[name])),
                "lineage": lineage,
            }
        materials.append({
            "incident": incident,
            "trusted_document_id": str(relevant["_id"]),
            "layout": {key: value for key, value in layout.items() if key not in {"pre_ids", "post_ids", "pre_chunks", "post_chunks", "masked_first_chunk"}},
            "f1_variants": f1_variants,
            "f2_variants": f2_variants,
            "f1_slices": f1_slices,
            "f2_slices": f2_slices,
            "actions": actions,
        })

    # Independent encoder passes are part of the certificate, even though they use
    # the same frozen model and deterministic settings.
    f1_vectors_all = embedder.encode(
        f1_texts_all,
        batch_size=args.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    f2_vectors_all = embedder.encode(
        f2_texts_all,
        batch_size=args.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    del embedder
    reranker = CrossEncoder(args.reranker_path, device=args.device, max_length=512)

    records: list[dict[str, Any]] = []
    prompts: list[list[dict[str, str]]] = []
    prompt_keys: list[tuple[int, str, str | None]] = []
    retrieval_started = time.perf_counter()
    for incident_index, material in enumerate(materials):
        incident = material["incident"]
        owner = incident["question_id"]
        question = incident["question"]

        def source_vectors(which: str, condition: str) -> np.ndarray:
            start, count = material[f"{which}_slices"][condition]
            source = f1_vectors_all if which == "f1" else f2_vectors_all
            return np.ascontiguousarray(source[start : start + count], dtype=np.float32)

        initial_source_vectors = source_vectors("f2", "initial")
        initial_ids, initial_texts, initial_vectors = vector_view(
            base_ids, base_texts, base_vectors, owner, "initial",
            material["f2_variants"]["initial"], initial_source_vectors,
        )
        initial = retrieve_physical_view(
            faiss, reranker, question, query_vectors[incident_index], initial_ids,
            initial_texts, initial_vectors, args.retrieve_k, args.serve_k,
            args.reranker_batch_size,
        )
        initial["source_vector_sha256"] = array_sha256(initial_source_vectors)
        initial["prompt_sha256"] = prompt_hash(question, initial["served_texts"])
        row: dict[str, Any] = {
            "incident_id": owner,
            "question": question,
            "correct_answer": incident["correct_answer"],
            "target_answer": incident["target_answer"],
            "trusted_document_id_removed_from_base": material["trusted_document_id"],
            "layout": material["layout"],
            "initial": without_texts(initial),
            "actions": {},
        }
        prompts.append(make_prompt(question, initial["served_texts"]))
        prompt_keys.append((len(records), "initial", None))

        for action in ACTION_NAMES:
            f1_source_vectors = source_vectors("f1", action)
            f2_source_vectors = source_vectors("f2", action)
            f1_ids, f1_texts, f1_vectors = vector_view(
                base_ids, base_texts, base_vectors, owner, action,
                material["f1_variants"][action], f1_source_vectors,
            )
            f2_ids, f2_texts, f2_vectors = vector_view(
                base_ids, base_texts, base_vectors, owner, action,
                material["f2_variants"][action], f2_source_vectors,
            )
            f1 = retrieve_exact_view(
                reranker, question, query_vectors[incident_index], f1_ids, f1_texts,
                f1_vectors, args.retrieve_k, args.serve_k, args.reranker_batch_size,
            )
            f2 = retrieve_physical_view(
                faiss, reranker, question, query_vectors[incident_index], f2_ids, f2_texts,
                f2_vectors, args.retrieve_k, args.serve_k, args.reranker_batch_size,
            )
            f1["source_vector_sha256"] = array_sha256(f1_source_vectors)
            f2["source_vector_sha256"] = array_sha256(f2_source_vectors)
            f1["prompt_sha256"] = prompt_hash(question, f1["served_texts"])
            f2["prompt_sha256"] = prompt_hash(question, f2["served_texts"])
            max_abs_error = (
                float(np.max(np.abs(f1_source_vectors - f2_source_vectors)))
                if len(f1_source_vectors)
                else 0.0
            )
            certificate = {
                "source_revision_sha256_exact": (
                    material["actions"][action]["f1_source_revision_sha256"]
                    == material["actions"][action]["f2_source_revision_sha256"]
                    == material["actions"][action]["lineage"]["source_revision_after_sha256"]
                ),
                "materialized_chunks_exact": (
                    canonical_sha256(material["f1_variants"][action])
                    == canonical_sha256(material["f2_variants"][action])
                ),
                "metadata_exact": f1["metadata_sha256"] == f2["metadata_sha256"],
                "source_vectors_within_atol": bool(np.allclose(f1_source_vectors, f2_source_vectors, rtol=0.0, atol=args.vector_atol)),
                "source_vectors_max_abs_error": max_abs_error,
                "dense_ids_exact": f1["dense_ids"] == f2["dense_ids"],
                "served_ids_exact": f1["served_ids"] == f2["served_ids"],
                "prompt_sha256_exact": f1["prompt_sha256"] == f2["prompt_sha256"],
                "answer_exact": None,
            }
            f0_texts = list(initial["served_texts"])
            if action == "target_span":
                replaced = False
                for text_index, text in enumerate(f0_texts):
                    if not replaced and incident["target_answer"].casefold() in text.casefold():
                        position = text.casefold().find(incident["target_answer"].casefold())
                        f0_texts[text_index] = text[:position] + "[removed]" + text[position + len(incident["target_answer"]):]
                        replaced = True
            elif action in {"visible_sentence", "all_attack_sentences"}:
                f0_texts = [
                    text for item_id, text in zip(initial["served_ids"], f0_texts)
                    if not item_id.endswith("::initial::0")
                ]
            else:
                f0_texts = [
                    text for item_id, text in zip(initial["served_ids"], f0_texts)
                    if not item_id.startswith(f"repairsource::{owner}::")
                ]
            f0_prompt = make_prompt(question, f0_texts)
            row["actions"][action] = {
                **material["actions"][action],
                "edit_tokens": material["actions"][action]["action_spec"]["edit_tokens"],
                "unchanged_base_reuse": base_bundle,
                "f0_trace": {
                    "served_ids": initial["served_ids"],
                    "prompt_sha256": canonical_sha256(f0_prompt),
                },
                "f1_exact_view": without_texts(f1),
                "f2_physical_rebuild": without_texts(f2),
                "certificate": certificate,
            }
            prompts.append(f0_prompt)
            prompt_keys.append((len(records), "f0", action))
            prompts.append(make_prompt(question, f1["served_texts"]))
            prompt_keys.append((len(records), "f1", action))
            prompts.append(make_prompt(question, f2["served_texts"]))
            prompt_keys.append((len(records), "f2", action))
        records.append(row)
    retrieval_seconds = time.perf_counter() - retrieval_started
    del reranker

    generation_started = time.perf_counter()
    answers = generate_answers(
        prompts, args.generator_path, args.device, args.generation_batch_size,
        args.max_new_tokens,
    )
    generation_seconds = time.perf_counter() - generation_started
    for answer, (row_index, stage, action) in zip(answers, prompt_keys, strict=True):
        row = records[row_index]
        label = {
            "answer": answer,
            "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            "target_present": contains_answer(answer, row["target_answer"]),
            "correct_present": contains_answer(answer, row["correct_answer"]),
        }
        if stage == "initial":
            row["initial"].update(label)
        else:
            destination = {"f0": "f0_trace", "f1": "f1_exact_view", "f2": "f2_physical_rebuild"}[stage]
            row["actions"][action][destination].update(label)

    reference_answer_checks = 0
    reference_answer_matches = 0
    for row in records:
        reference = reference_by_id[row["incident_id"]]
        row["initial"]["reference_answer_exact"] = row["initial"]["answer"] == reference["initial"]["answer"]
        reference_answer_checks += 1
        reference_answer_matches += int(row["initial"]["reference_answer_exact"])
        for action, details in row["actions"].items():
            details["certificate"]["answer_exact"] = (
                details["f1_exact_view"]["answer"] == details["f2_physical_rebuild"]["answer"]
            )
            details["certificate"]["all_required_checks_pass"] = all(
                bool(value) for key, value in details["certificate"].items()
                if key != "source_vectors_max_abs_error"
            )
            details["f0_trace"]["reference_answer_exact"] = (
                details["f0_trace"]["answer"] == reference["actions"][action]["f0_trace"]["answer"]
            )
            details["f2_physical_rebuild"]["reference_answer_exact"] = (
                details["f2_physical_rebuild"]["answer"]
                == reference["actions"][action]["f2_physical_rebuild"]["answer"]
            )
            reference_answer_checks += 2
            reference_answer_matches += int(details["f0_trace"]["reference_answer_exact"])
            reference_answer_matches += int(details["f2_physical_rebuild"]["reference_answer_exact"])
        feasible = [
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
        oracle = min(feasible)[1] if feasible else None
        row["selection"] = {
            "tie_break": ["edit_tokens_ascending", "action_name_lexicographic"],
            "fixed_trace_cheapest_safe": min(fixed_feasible)[1] if fixed_feasible else None,
            "reindexrepair_best_validated": oracle,
            "exhaustive_f2_minimum": oracle,
            "finalist_recall": oracle is not None,
            "edit_cost_regret_tokens": 0 if oracle is not None else None,
            "physical_rebuilds_deployment_method": 1 if oracle is not None else 0,
            "physical_rebuilds_exhaustive_oracle": len(ACTION_NAMES),
        }

    all_actions = [details for row in records for details in row["actions"].values()]
    selected = [row for row in records if row["selection"]["reindexrepair_best_validated"]]
    fixed = [row for row in records if row["selection"]["fixed_trace_cheapest_safe"]]
    summary = {
        "incidents": len(records),
        "actions_certified": sum(details["certificate"]["all_required_checks_pass"] for details in all_actions),
        "actions_total": len(all_actions),
        "all_independent_f1_f2_certificates_pass": all(details["certificate"]["all_required_checks_pass"] for details in all_actions),
        "maximum_source_vector_abs_error": max(details["certificate"]["source_vectors_max_abs_error"] for details in all_actions),
        "reference_answer_matches": reference_answer_matches,
        "reference_answer_checks": reference_answer_checks,
        "all_human_stimulus_answers_unchanged": reference_answer_matches == reference_answer_checks,
        "fixed_trace_selects_f2_unsafe": sum(
            row["actions"][row["selection"]["fixed_trace_cheapest_safe"]]["f2_physical_rebuild"]["target_present"]
            for row in fixed
        ),
        "fixed_trace_selected": len(fixed),
        "reindexrepair_selected": len(selected),
        "reindexrepair_f2_unsafe": sum(
            row["actions"][row["selection"]["reindexrepair_best_validated"]]["f2_physical_rebuild"]["target_present"]
            for row in selected
        ),
        "reindexrepair_correct_present": sum(
            row["actions"][row["selection"]["reindexrepair_best_validated"]]["f2_physical_rebuild"]["correct_present"]
            for row in selected
        ),
        "finalist_recall": sum(row["selection"]["finalist_recall"] for row in records) / len(records),
        "mean_physical_rebuilds_method": sum(row["selection"]["physical_rebuilds_deployment_method"] for row in records) / len(records),
        "mean_physical_rebuilds_exhaustive": len(ACTION_NAMES),
    }
    if not summary["all_independent_f1_f2_certificates_pass"]:
        raise RuntimeError("at least one independent F1/F2 certificate failed")
    if not summary["all_human_stimulus_answers_unchanged"]:
        raise RuntimeError("the independent rerun changed at least one human-annotated answer")

    transform_bundle = {
        "chunk_size_tokens": args.chunk_size,
        "chunk_overlap_tokens": 0,
        "query_prefix": QUERY_PREFIX,
        "normalization": "SentenceTransformer normalize_embeddings=True",
        "dense_tie_break": ["score_descending", "stable_id_lexicographic"],
        "reranker_tie_break": ["score_descending", "stable_id_lexicographic"],
        "action_tie_break": ["edit_tokens_ascending", "action_name_lexicographic"],
        "embedder": {"name": EMBEDDER_NAME, "revision": EMBEDDER_REVISION, "fingerprint": embedder_fingerprint},
        "reranker": {"name": RERANKER_NAME, "revision": RERANKER_REVISION, "fingerprint": reranker_fingerprint},
        "generator": {"name": GENERATOR_NAME, "revision": GENERATOR_REVISION, "fingerprint": generator_fingerprint},
    }
    payload = {
        "tier": "posthoc_independent_f1_f2_mechanical_certificate",
        "paper_evidence": "supporting_validation_only",
        "limitations": [
            "The certificate validates a deterministic constructed repair lattice; it does not convert that stress test into a population estimate.",
            "Human semantic endpoints remain bound to the immutable reference artifact, and the rerun is accepted only if every annotated answer is byte-identical.",
            "Unchanged base-corpus descendants are reused only after their ID, text, and vector hashes are validated; every edited source root is fully rechunked and re-embedded in both paths.",
        ],
        "configuration": {
            "incidents": len(records),
            "clean_documents_after_exclusion": len(base_ids),
            "excluded_relevant_documents": len(excluded_relevant),
            "retrieve_k": args.retrieve_k,
            "serve_k": args.serve_k,
            "repair_grammar": list(ACTION_NAMES),
            "max_new_tokens": args.max_new_tokens,
            "vector_atol": args.vector_atol,
            "transform_fingerprint_bundle": transform_bundle,
            "transform_fingerprint_bundle_sha256": canonical_sha256(transform_bundle),
            "unchanged_base_bundle": base_bundle,
        },
        "inputs": {
            "corpus": {"path": str(args.corpus), "sha256": sha256_file(args.corpus)},
            "candidates": {"path": str(args.candidates), "sha256": sha256_file(args.candidates)},
            "incidents": {"path": str(args.incidents), "sha256": sha256_file(args.incidents)},
            "reference_artifact": {"path": str(args.reference_artifact), "sha256": sha256_file(args.reference_artifact)},
            "clean_vector_cache": {"path": str(args.clean_vector_cache), "sha256": sha256_file(args.clean_vector_cache)},
            "query_vector_cache": {"path": str(args.query_vector_cache), "sha256": sha256_file(args.query_vector_cache)},
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "retrieval_seconds": retrieval_seconds,
            "generation_seconds": generation_seconds,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
