from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from reindexrepair.core import (
    FixedTokenChunker,
    RepairAction,
    SourceDocument,
    apply_repair,
    build_counterfactual_view,
    build_index,
)


MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenBgeEmbedder:
    def __init__(self, batch_size: int, device: str, model_path: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_path,
            revision=MODEL_REVISION if model_path == MODEL_NAME else None,
            device=device,
        )
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()
        self.fingerprint = hashlib.sha256(
            "|".join(
                [
                    "SentenceTransformer:v1",
                    MODEL_NAME,
                    MODEL_REVISION,
                    "pooling=mean",
                    "normalize=true",
                    "dtype=float32",
                ]
            ).encode("utf-8")
        ).hexdigest()

    def encode(self, texts):
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def encode_queries(self, questions):
        return self.encode([QUERY_PREFIX + question for question in questions])


def ranked_ids(query_vector: np.ndarray, chunks, top_k: int) -> list[str]:
    scored = [
        (chunk.chunk_id, float(np.dot(query_vector, chunk.vector))) for chunk in chunks
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [item[0] for item in scored[:top_k]]


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
    parser.add_argument("--output", type=Path, default=Path("artifacts/f1_exactness.json"))
    parser.add_argument("--documents", type=int, default=30)
    parser.add_argument("--contexts-per-document", type=int, default=10)
    parser.add_argument("--edited-documents", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", default=MODEL_NAME)
    args = parser.parse_args()

    if not 0 < args.edited_documents < args.documents:
        raise ValueError("edited-documents must be positive and smaller than documents")
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    selected = raw[: args.documents]
    documents = [
        SourceDocument(
            f"incident-{index}",
            "\n".join(item["context_texts"][: args.contexts_per_document]),
            trusted=False,
        )
        for index, item in enumerate(selected)
    ]
    questions = [item["question"] for item in selected]
    chunker = FixedTokenChunker(args.chunk_size, args.overlap)
    embedder = FrozenBgeEmbedder(args.batch_size, args.device, args.model_path)

    started = time.perf_counter()
    base = build_index(documents, chunker, embedder)
    repaired = list(documents)
    affected: set[str] = set()
    for index in range(args.edited_documents):
        source_id = f"incident-{index}"
        action = RepairAction(
            f"delete-{index}", source_id, "delete_span", 4, 12
        )
        repaired = apply_repair(repaired, action)
        affected.add(source_id)

    f1 = build_counterfactual_view(
        base, documents, repaired, affected, chunker, embedder
    )
    f2 = build_index(repaired, chunker, embedder)
    query_vectors = embedder.encode_queries(questions)
    f1_rankings = [ranked_ids(vector, f1.chunks, 30) for vector in query_vectors]
    f2_rankings = [ranked_ids(vector, f2.chunks, 30) for vector in query_vectors]

    f1_meta = [
        (chunk.chunk_id, chunk.source_id, chunk.start_token, chunk.end_token, chunk.text)
        for chunk in f1.chunks
    ]
    f2_meta = [
        (chunk.chunk_id, chunk.source_id, chunk.start_token, chunk.end_token, chunk.text)
        for chunk in f2.chunks
    ]
    vectors_f1 = np.stack([chunk.vector for chunk in f1.chunks])
    vectors_f2 = np.stack([chunk.vector for chunk in f2.chunks])
    max_vector_error = float(np.max(np.abs(vectors_f1 - vectors_f2)))
    payload = {
        "tier": "implementation_verification",
        "paper_evidence": False,
        "input": {
            "path": str(args.input),
            "sha256": sha256_file(args.input),
            "documents": args.documents,
            "contexts_per_document": args.contexts_per_document,
            "edited_documents": args.edited_documents,
        },
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "load_path": args.model_path,
            "device": args.device,
            "dimension": embedder.dim,
        },
        "chunker": {"chunk_size": args.chunk_size, "overlap": args.overlap},
        "checks": {
            "source_hash_equal": f1.source_hash == f2.source_hash,
            "chunk_metadata_equal": f1_meta == f2_meta,
            "chunk_count_f1": len(f1.chunks),
            "chunk_count_f2": len(f2.chunks),
            "max_absolute_vector_error": max_vector_error,
            "vectors_equal_with_atol_1e-6": bool(
                np.allclose(vectors_f1, vectors_f2, rtol=0.0, atol=1e-6)
            ),
            "top30_rankings_equal": f1_rankings == f2_rankings,
            "ranking_queries": len(questions),
        },
        "runtime": {
            "seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["checks"], indent=2))
    if not all(
        [
            payload["checks"]["source_hash_equal"],
            payload["checks"]["chunk_metadata_equal"],
            payload["checks"]["vectors_equal_with_atol_1e-6"],
            payload["checks"]["top30_rankings_equal"],
        ]
    ):
        raise SystemExit("F1 exactness verification failed")


if __name__ == "__main__":
    main()
