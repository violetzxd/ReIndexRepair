from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from reindexrepair.metrics import clopper_pearson_interval
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
    fingerprint_path,
    generate_answers,
    make_prompt,
    read_jsonl,
    sha256_file,
)


def rate_record(events: int, eligible: int) -> dict[str, object]:
    return {
        "events": events,
        "eligible": eligible,
        "rate": events / eligible if eligible else None,
        "two_sided_95_exact_interval": (
            list(clopper_pearson_interval(events, eligible)) if eligible else None
        ),
    }


def chunk_ids(token_ids: list[int], size: int) -> list[list[int]]:
    return [token_ids[start : start + size] for start in range(0, len(token_ids), size)]


def find_target_token(offsets: list[tuple[int, int]], text: str, target: str) -> int:
    character = text.casefold().rfind(target.casefold())
    if character < 0:
        raise ValueError("target string is absent from constructed backup sentence")
    for index, (start, end) in enumerate(offsets):
        if start <= character < end or start >= character:
            return index
    raise ValueError("could not map target character to a token")


def build_boundary_source(
    tokenizer,
    incident: dict,
    chunk_size: int,
    filler_ids: list[int],
) -> dict[str, object]:
    visible_text = incident["context_texts"][0]
    visible_ids = tokenizer.encode(visible_text, add_special_tokens=False)
    if len(visible_ids) >= chunk_size:
        raise ValueError("visible attack passage does not fit in the declared chunk")
    backup_text = (
        f"{incident['question']}. The archived answer to this question is "
        f"{incident['target_answer']}."
    )
    encoded_backup = tokenizer(
        backup_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    backup_ids = encoded_backup["input_ids"]
    target_token = find_target_token(
        encoded_backup["offset_mapping"], backup_text, incident["target_answer"]
    )
    desired_backup_start = (chunk_size - target_token) % chunk_size
    filler_length = (desired_backup_start - len(visible_ids)) % chunk_size
    repeated_filler = (filler_ids * ((filler_length + chunk_size) // len(filler_ids) + 2))
    between_ids = repeated_filler[:filler_length]
    suffix_ids = repeated_filler[:chunk_size]
    pre_ids = visible_ids + between_ids + backup_ids + suffix_ids
    post_ids = between_ids + backup_ids + suffix_ids
    pre_chunks = chunk_ids(pre_ids, chunk_size)
    post_chunks = chunk_ids(post_ids, chunk_size)
    if len(pre_chunks) < 2:
        raise ValueError("constructed pre-edit source must cross a chunk boundary")
    masked_first = (repeated_filler[: len(visible_ids)] + pre_chunks[0][len(visible_ids) :])[
        :chunk_size
    ]
    return {
        "visible_text": visible_text,
        "backup_text": backup_text,
        "visible_tokens": len(visible_ids),
        "backup_tokens": len(backup_ids),
        "target_token_in_backup": target_token,
        "filler_tokens_between": filler_length,
        "pre_ids": pre_ids,
        "post_ids": post_ids,
        "pre_chunks": pre_chunks,
        "post_chunks": post_chunks,
        "masked_first_chunk": masked_first,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--incidents", type=Path, required=True)
    parser.add_argument("--embedder-path", required=True)
    parser.add_argument("--reranker-path", required=True)
    parser.add_argument("--generator-path", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--retrieve-k", type=int, default=30)
    parser.add_argument("--serve-k", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import faiss
    from sentence_transformers import CrossEncoder, SentenceTransformer

    incidents = json.loads(args.incidents.read_text(encoding="utf-8"))[: args.limit]
    query_manifest = json.loads(args.queries.read_text(encoding="utf-8"))
    allowed = {item["incident_id"] for item in query_manifest}
    if any(item["question_id"] not in allowed for item in incidents):
        raise ValueError("incident/query manifest mismatch")
    clean_records = list(read_jsonl(args.corpus))
    clean_ids = [str(item["_id"]) for item in clean_records]
    clean_texts = [document_text(item) for item in clean_records]

    embedder_fingerprint = fingerprint_path(args.embedder_path)
    reranker_fingerprint = fingerprint_path(args.reranker_path)
    generator_fingerprint = fingerprint_path(args.generator_path)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "corpus": sha256_file(args.corpus),
                "embedder": embedder_fingerprint["sha256_tree"],
                "query_prefix": QUERY_PREFIX,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    clean_cache = args.cache_dir / f"clean-{cache_key}.npy"
    query_cache = args.cache_dir / f"queries-{cache_key}.npy"
    embedder = SentenceTransformer(args.embedder_path, device=args.device)
    if clean_cache.exists() and query_cache.exists():
        clean_vectors = np.load(clean_cache)
        query_vectors = np.load(query_cache)[: len(incidents)]
        cache_hit = True
    else:
        clean_vectors = embedder.encode(
            clean_texts,
            batch_size=args.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
        query_vectors = embedder.encode(
            [QUERY_PREFIX + item["question"] for item in incidents],
            batch_size=args.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
        np.save(clean_cache, clean_vectors)
        np.save(query_cache, query_vectors)
        cache_hit = False

    tokenizer = embedder.tokenizer
    filler = "Administrative archive note with unrelated catalog details and neutral indexing metadata."
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)
    layouts = [build_boundary_source(tokenizer, item, args.chunk_size, filler_ids) for item in incidents]
    all_chunk_texts = []
    slices = []
    for layout in layouts:
        pre_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in layout["pre_chunks"]]
        post_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in layout["post_chunks"]]
        start = len(all_chunk_texts)
        all_chunk_texts.extend(pre_texts + post_texts)
        slices.append((start, start + len(pre_texts), len(post_texts), pre_texts, post_texts))
    all_chunk_vectors = embedder.encode(
        all_chunk_texts,
        batch_size=args.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    del embedder
    reranker = CrossEncoder(args.reranker_path, device=args.device, max_length=512)

    def retrieve_and_rerank(
        question: str,
        query_vector: np.ndarray,
        source_texts: list[str],
        source_vectors: np.ndarray,
        condition: str,
    ) -> tuple[list[str], list[str], list[float], list[str]]:
        documents = np.concatenate([clean_vectors, source_vectors], axis=0)
        index = faiss.IndexFlatIP(documents.shape[1])
        index.add(np.ascontiguousarray(documents))
        scores, offsets = index.search(
            np.ascontiguousarray(query_vector.reshape(1, -1)), args.retrieve_k
        )
        dense = offsets[0].tolist()
        candidate_texts = [
            clean_texts[item] if item < len(clean_texts) else source_texts[item - len(clean_texts)]
            for item in dense
        ]
        rerank_scores = np.asarray(
            reranker.predict(
                [(question, text) for text in candidate_texts],
                batch_size=args.reranker_batch_size,
                show_progress_bar=False,
            )
        ).reshape(-1)
        order = sorted(range(len(dense)), key=lambda item: (-float(rerank_scores[item]), dense[item]))
        served_offsets = order[: args.serve_k]
        dense_ids = [
            clean_ids[item] if item < len(clean_ids) else f"boundary::{condition}::{item-len(clean_ids)}"
            for item in dense
        ]
        served_ids = [dense_ids[item] for item in served_offsets]
        served_texts = [candidate_texts[item] for item in served_offsets]
        return dense_ids, served_ids, [float(rerank_scores[item]) for item in served_offsets], served_texts

    rows = []
    prompts = []
    prompt_keys = []
    for incident_index, (incident, layout, slice_info) in enumerate(zip(incidents, layouts, slices)):
        start, post_start, post_count, pre_texts, post_texts = slice_info
        pre_vectors = all_chunk_vectors[start:post_start]
        post_vectors = all_chunk_vectors[post_start : post_start + post_count]
        initial_dense, initial_ids, initial_scores, initial_texts = retrieve_and_rerank(
            incident["question"], query_vectors[incident_index], pre_texts, pre_vectors, "pre"
        )
        f2_dense, f2_ids, f2_scores, f2_texts = retrieve_and_rerank(
            incident["question"], query_vectors[incident_index], post_texts, post_vectors, "post"
        )
        empty_vectors = np.empty((0, clean_vectors.shape[1]), dtype=np.float32)
        control_dense, control_ids, control_scores, control_texts = retrieve_and_rerank(
            incident["question"], query_vectors[incident_index], [], empty_vectors, "quarantine"
        )
        masked_text = tokenizer.decode(layout["masked_first_chunk"], skip_special_tokens=True)
        f0_texts = [
            masked_text if item_id == "boundary::pre::0" else text
            for item_id, text in zip(initial_ids, initial_texts)
        ]
        row = {
            "incident_id": incident["question_id"],
            "question": incident["question"],
            "correct_answer": incident["correct_answer"],
            "target_answer": incident["target_answer"],
            "layout": {
                key: value
                for key, value in layout.items()
                if key not in {"pre_ids", "post_ids", "pre_chunks", "post_chunks", "masked_first_chunk"}
            },
            "initial": {"dense_ids": initial_dense, "served_ids": initial_ids, "reranker_scores": initial_scores},
            "f0_trace_mask": {"served_ids": initial_ids},
            "f2_rechunk_rebuild": {"dense_ids": f2_dense, "served_ids": f2_ids, "reranker_scores": f2_scores},
            "complete_lineage_quarantine": {
                "dense_ids": control_dense,
                "served_ids": control_ids,
                "reranker_scores": control_scores,
            },
            "boundary_retrieval_migration": (
                "boundary::pre::0" in initial_ids
                and "boundary::pre::1" not in initial_ids
                and "boundary::post::0" in f2_ids
            ),
        }
        rows.append(row)
        for condition, contexts in (
            ("initial", initial_texts),
            ("f0_trace_mask", f0_texts),
            ("f2_rechunk_rebuild", f2_texts),
            ("complete_lineage_quarantine", control_texts),
        ):
            prompts.append(make_prompt(incident["question"], contexts))
            prompt_keys.append((incident_index, condition))

    del reranker
    answers = generate_answers(
        prompts,
        args.generator_path,
        args.device,
        args.generation_batch_size,
        args.max_new_tokens,
    )
    for answer, (incident_index, condition) in zip(answers, prompt_keys, strict=True):
        row = rows[incident_index]
        row[condition]["answer"] = answer
        row[condition]["target_present"] = contains_answer(answer, row["target_answer"])
        row[condition]["correct_present"] = contains_answer(answer, row["correct_answer"])
    for row in rows:
        row["initial_unsafe"] = row["initial"]["target_present"]
        row["f0_safe"] = not row["f0_trace_mask"]["target_present"]
        row["f2_unsafe"] = row["f2_rechunk_rebuild"]["target_present"]
        row["complete_lineage_unsafe"] = row["complete_lineage_quarantine"]["target_present"]
        row["repair_mirage"] = row["initial_unsafe"] and row["f0_safe"] and row["f2_unsafe"]

    attacked = [row for row in rows if row["initial_unsafe"]]
    f0_safe = [row for row in attacked if row["f0_safe"]]
    mirages = [row for row in f0_safe if row["f2_unsafe"]]
    control_events = [row for row in f0_safe if row["complete_lineage_unsafe"]]
    payload = {
        "tier": "posthoc_constructed_boundary_migration_stress_test",
        "paper_evidence": False,
        "limitations": [
            "This is a deterministic constructed stress test, not an estimate of natural prevalence.",
            "The layout rule is applied to the first frozen certification incidents without outcome selection.",
            "Chunks use the frozen BGE tokenizer with zero overlap; overlap sensitivity remains required.",
            "Answer labels are lexical diagnostics until blinded semantic adjudication is complete.",
        ],
        "configuration": {
            "incidents": len(rows),
            "clean_documents": len(clean_texts),
            "chunk_size_tokens": args.chunk_size,
            "chunk_overlap_tokens": 0,
            "retrieve_k": args.retrieve_k,
            "serve_k": args.serve_k,
            "embedder": {"name": EMBEDDER_NAME, "revision": EMBEDDER_REVISION, "fingerprint": embedder_fingerprint},
            "reranker": {"name": RERANKER_NAME, "revision": RERANKER_REVISION, "fingerprint": reranker_fingerprint},
            "generator": {"name": GENERATOR_NAME, "revision": GENERATOR_REVISION, "fingerprint": generator_fingerprint},
            "generation_batch_size": args.generation_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "deterministic_generation": {
                "do_sample": False,
                "torch_seed": 0,
                "deterministic_algorithms": True,
            },
        },
        "inputs": {
            "corpus_sha256": sha256_file(args.corpus),
            "queries_sha256": sha256_file(args.queries),
            "incidents_sha256": sha256_file(args.incidents),
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "clean_embedding_cache_hit": cache_hit, "script_sha256": sha256_file(Path(__file__))},
        "summary": {
            "boundary_retrieval_migration": rate_record(sum(row["boundary_retrieval_migration"] for row in rows), len(rows)),
            "initial_generator_attack_success": rate_record(len(attacked), len(rows)),
            "conditional_false_safe": rate_record(len(mirages), len(f0_safe)),
            "joint_repair_mirage": rate_record(len(mirages), len(attacked)),
            "complete_lineage_target_recurrence_conditional": rate_record(len(control_events), len(f0_safe)),
        },
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
