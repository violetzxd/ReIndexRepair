from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

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


def first_subsequence(values: list[int], needle: list[int], limit: int) -> tuple[int, int]:
    for start in range(max(0, limit - len(needle) + 1)):
        if values[start : start + len(needle)] == needle:
            return start, start + len(needle)
    raise ValueError("target token sequence not found in visible attack")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--incidents", type=Path, required=True)
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
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import faiss
    from sentence_transformers import CrossEncoder, SentenceTransformer

    incidents = json.loads(args.incidents.read_text(encoding="utf-8"))[: args.limit]
    candidate_payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates_by_id = {item["query_id"]: item for item in candidate_payload["records"]}
    corpus = list(read_jsonl(args.corpus))
    clean_ids = [str(item["_id"]) for item in corpus]
    clean_texts = [document_text(item) for item in corpus]
    clean_vectors_all = np.load(args.clean_vector_cache)
    query_vectors_all = np.load(args.query_vector_cache)
    if len(clean_vectors_all) != len(clean_ids):
        raise ValueError("clean vector cache does not match corpus")
    if len(query_vectors_all) < len(incidents):
        raise ValueError("query vector cache has too few rows")
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

    embedder_fingerprint = fingerprint_path(args.embedder_path)
    reranker_fingerprint = fingerprint_path(args.reranker_path)
    generator_fingerprint = fingerprint_path(args.generator_path)
    embedder = SentenceTransformer(args.embedder_path, device=args.device)
    tokenizer = embedder.tokenizer
    filler_ids = tokenizer.encode(
        "Administrative archive note with unrelated catalog details and neutral indexing metadata.",
        add_special_tokens=False,
    )
    action_names = ("target_span", "visible_sentence", "all_attack_sentences", "document_quarantine")
    incident_material = []
    all_chunk_texts = []
    for incident in incidents:
        relevant = candidates_by_id[incident["beir_query_id"]]["relevant_documents"][0]
        trusted_text = document_text(relevant)
        trusted_ids = tokenizer.encode(trusted_text, add_special_tokens=False)
        layout = build_boundary_source(tokenizer, incident, args.chunk_size, filler_ids)
        visible_len = int(layout["visible_tokens"])
        backup_len = int(layout["backup_tokens"])
        between_len = int(layout["filler_tokens_between"])
        pre_ids = list(layout["pre_ids"]) + trusted_ids
        target_ids = tokenizer.encode(incident["target_answer"], add_special_tokens=False)
        span_start, span_end = first_subsequence(pre_ids, target_ids, visible_len)
        sentence_ids = pre_ids[visible_len:]
        backup_start = visible_len + between_len
        backup_end = backup_start + backup_len
        all_attack_ids = pre_ids[visible_len:backup_start] + pre_ids[backup_end:]
        variants = {
            "initial": pre_ids,
            "target_span": pre_ids[:span_start] + pre_ids[span_end:],
            "visible_sentence": sentence_ids,
            "all_attack_sentences": all_attack_ids,
            "document_quarantine": [],
        }
        text_variants = {
            name: [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunk_ids(token_ids, args.chunk_size)]
            for name, token_ids in variants.items()
        }
        slices = {}
        for name, texts in text_variants.items():
            start = len(all_chunk_texts)
            all_chunk_texts.extend(texts)
            slices[name] = (start, len(texts))
        incident_material.append({
            "incident": incident,
            "trusted_document_id": str(relevant["_id"]),
            "trusted_text": trusted_text,
            "layout": {key: value for key, value in layout.items() if key not in {"pre_ids", "post_ids", "pre_chunks", "post_chunks", "masked_first_chunk"}},
            "variants": text_variants,
            "slices": slices,
            "costs": {
                "target_span": span_end - span_start,
                "visible_sentence": visible_len,
                "all_attack_sentences": visible_len + backup_len,
                "document_quarantine": len(pre_ids),
            },
        })
    variant_vectors = embedder.encode(
        all_chunk_texts,
        batch_size=args.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    del embedder
    reranker = CrossEncoder(args.reranker_path, device=args.device, max_length=512)

    def retrieve(question: str, query: np.ndarray, owner: str, condition: str, source_texts: list[str], source_vectors: np.ndarray, materialize: bool) -> dict:
        combined = np.concatenate([base_vectors, source_vectors], axis=0) if len(source_vectors) else base_vectors
        combined_ids = base_ids + [f"repairsource::{owner}::{condition}::{index}" for index in range(len(source_texts))]
        combined_texts = base_texts + source_texts
        expected = exact_top_k(query, combined, args.retrieve_k)
        if materialize:
            index = faiss.IndexFlatIP(combined.shape[1])
            index.add(np.ascontiguousarray(combined))
            _, offsets = index.search(np.ascontiguousarray(query.reshape(1, -1)), args.retrieve_k)
            dense = sorted(offsets[0].tolist(), key=lambda item: (-float(combined[item] @ query), item))
        else:
            dense = expected
        scores = np.asarray(reranker.predict(
            [(question, combined_texts[item]) for item in dense],
            batch_size=args.reranker_batch_size,
            show_progress_bar=False,
        )).reshape(-1)
        ranked = [dense[offset] for offset in sorted(range(len(dense)), key=lambda offset: (-float(scores[offset]), dense[offset]))]
        served = ranked[: args.serve_k]
        return {
            "dense_ids": [combined_ids[item] for item in dense],
            "served_ids": [combined_ids[item] for item in served],
            "served_texts": [combined_texts[item] for item in served],
            "matches_exact_view": dense == expected,
            "materialized_faiss_index": materialize,
        }

    records = []
    prompts = []
    prompt_keys = []
    retrieval_started = time.perf_counter()
    for incident_index, material in enumerate(incident_material):
        incident = material["incident"]
        owner = incident["question_id"]
        question = incident["question"]
        initial_start, initial_count = material["slices"]["initial"]
        initial = retrieve(
            question, query_vectors[incident_index], owner, "initial", material["variants"]["initial"],
            variant_vectors[initial_start : initial_start + initial_count], True,
        )
        row = {
            "incident_id": owner,
            "question": question,
            "correct_answer": incident["correct_answer"],
            "target_answer": incident["target_answer"],
            "trusted_document_id_removed_from_base": material["trusted_document_id"],
            "layout": material["layout"],
            "initial": {key: value for key, value in initial.items() if key != "served_texts"},
            "actions": {},
        }
        prompts.append(make_prompt(question, initial["served_texts"]))
        prompt_keys.append((len(records), "initial", None))
        for action in action_names:
            start, count = material["slices"][action]
            f2 = retrieve(
                question, query_vectors[incident_index], owner, action, material["variants"][action],
                variant_vectors[start : start + count], True,
            )
            f0_texts = list(initial["served_texts"])
            if action == "target_span":
                replaced = False
                for index, text in enumerate(f0_texts):
                    if not replaced and incident["target_answer"].casefold() in text.casefold():
                        position = text.casefold().find(incident["target_answer"].casefold())
                        f0_texts[index] = text[:position] + "[removed]" + text[position + len(incident["target_answer"]):]
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
            row["actions"][action] = {
                "edit_tokens": material["costs"][action],
                "f0_trace": {"served_ids": initial["served_ids"]},
                "f1_exact_view": {
                    "dense_ids": f2["dense_ids"], "served_ids": f2["served_ids"],
                    "matches_physical_f2": f2["matches_exact_view"],
                },
                "f2_physical_rebuild": {key: value for key, value in f2.items() if key != "served_texts"},
            }
            prompts.append(make_prompt(question, f0_texts))
            prompt_keys.append((len(records), "f0", action))
            prompts.append(make_prompt(question, f2["served_texts"]))
            prompt_keys.append((len(records), "f2", action))
        records.append(row)
    retrieval_seconds = time.perf_counter() - retrieval_started
    del reranker
    generation_started = time.perf_counter()
    answers = generate_answers(prompts, args.generator_path, args.device, args.generation_batch_size, args.max_new_tokens)
    generation_seconds = time.perf_counter() - generation_started
    for answer, (row_index, stage, action) in zip(answers, prompt_keys, strict=True):
        row = records[row_index]
        label = {
            "answer": answer,
            "target_present": contains_answer(answer, row["target_answer"]),
            "correct_present": contains_answer(answer, row["correct_answer"]),
        }
        if stage == "initial":
            row["initial"].update(label)
        elif stage == "f0":
            row["actions"][action]["f0_trace"].update(label)
        else:
            row["actions"][action]["f1_exact_view"].update(label)
            row["actions"][action]["f2_physical_rebuild"].update(label)
    for row in records:
        feasible = [
            (details["edit_tokens"], name)
            for name, details in row["actions"].items()
            if not details["f1_exact_view"]["target_present"] and details["f1_exact_view"]["correct_present"]
        ]
        oracle = min(feasible)[1] if feasible else None
        f0_feasible = [
            (details["edit_tokens"], name)
            for name, details in row["actions"].items()
            if not details["f0_trace"]["target_present"]
        ]
        row["selection"] = {
            "fixed_trace_cheapest_safe": min(f0_feasible)[1] if f0_feasible else None,
            "reindexrepair_best_validated": oracle,
            "exhaustive_f2_minimum": oracle,
            "finalist_recall": oracle is not None,
            "edit_cost_regret_tokens": 0 if oracle is not None else None,
            "physical_rebuilds_deployment_method": 1 if oracle is not None else 0,
            "physical_rebuilds_exhaustive_oracle": len(action_names),
        }
    selected = [row for row in records if row["selection"]["reindexrepair_best_validated"]]
    fixed = [row for row in records if row["selection"]["fixed_trace_cheapest_safe"]]
    summary = {
        "incidents": len(records),
        "all_f1_rankings_match_physical_f2": all(action["f1_exact_view"]["matches_physical_f2"] for row in records for action in row["actions"].values()),
        "fixed_trace_selects_f2_unsafe": sum(row["actions"][row["selection"]["fixed_trace_cheapest_safe"]]["f2_physical_rebuild"]["target_present"] for row in fixed),
        "fixed_trace_selected": len(fixed),
        "reindexrepair_selected": len(selected),
        "reindexrepair_f2_unsafe": sum(row["actions"][row["selection"]["reindexrepair_best_validated"]]["f2_physical_rebuild"]["target_present"] for row in selected),
        "reindexrepair_correct_present": sum(row["actions"][row["selection"]["reindexrepair_best_validated"]]["f2_physical_rebuild"]["correct_present"] for row in selected),
        "document_quarantine_correct_present": sum(row["actions"]["document_quarantine"]["f2_physical_rebuild"]["correct_present"] for row in records),
        "finalist_recall": sum(row["selection"]["finalist_recall"] for row in records) / len(records),
        "mean_physical_rebuilds_method": sum(row["selection"]["physical_rebuilds_deployment_method"] for row in records) / len(records),
        "mean_physical_rebuilds_exhaustive": len(action_names),
    }
    payload = {
        "tier": "posthoc_constructed_repair_lattice_stress_test",
        "paper_evidence": False,
        "limitations": [
            "The first 30 frozen incidents and deterministic repair grammar are used without outcome filtering, but the source layout is a constructed stress test.",
            "Answer labels are lexical diagnostics pending blinded human semantic annotation.",
            "The clean copies of relevant documents are removed so local utility measures whether the repaired source preserves its trusted content; this is not global utility.",
        ],
        "configuration": {
            "incidents": len(records), "clean_documents_after_exclusion": len(base_ids), "excluded_relevant_documents": len(excluded_relevant),
            "chunk_size_tokens": args.chunk_size, "chunk_overlap_tokens": 0, "retrieve_k": args.retrieve_k, "serve_k": args.serve_k,
            "repair_grammar": list(action_names), "max_new_tokens": args.max_new_tokens,
            "embedder": {"name": EMBEDDER_NAME, "revision": EMBEDDER_REVISION, "fingerprint": embedder_fingerprint},
            "reranker": {"name": RERANKER_NAME, "revision": RERANKER_REVISION, "fingerprint": reranker_fingerprint},
            "generator": {"name": GENERATOR_NAME, "revision": GENERATOR_REVISION, "fingerprint": generator_fingerprint},
        },
        "inputs": {
            "corpus": {"path": str(args.corpus), "sha256": sha256_file(args.corpus)},
            "candidates": {"path": str(args.candidates), "sha256": sha256_file(args.candidates)},
            "incidents": {"path": str(args.incidents), "sha256": sha256_file(args.incidents)},
            "clean_vector_cache": {"path": str(args.clean_vector_cache), "sha256": sha256_file(args.clean_vector_cache)},
            "query_vector_cache": {"path": str(args.query_vector_cache), "sha256": sha256_file(args.query_vector_cache)},
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "retrieval_seconds": retrieval_seconds, "generation_seconds": generation_seconds, "script_sha256": sha256_file(Path(__file__))},
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
