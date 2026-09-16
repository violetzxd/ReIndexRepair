from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import unicodedata
from typing import Iterable, Mapping, Sequence

import numpy as np


TOKEN_RE = re.compile(r"\S+")


def normalize_text(text: str) -> str:
    """Deterministic normalization used by every fidelity level."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def stable_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceDocument:
    source_id: str
    text: str
    trusted: bool = True


@dataclass(frozen=True)
class RepairAction:
    action_id: str
    source_id: str
    kind: str
    start_token: int | None = None
    end_token: int | None = None
    replacement: str = ""

    @property
    def is_document_delete(self) -> bool:
        return self.kind in {"delete_document", "quarantine_document"}


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source_id: str
    start_token: int
    end_token: int
    text: str
    vector: np.ndarray


@dataclass(frozen=True)
class IndexSnapshot:
    source_hash: str
    build_id: str
    pipeline_fingerprint: str
    chunks: tuple[ChunkRecord, ...]
    chunk_size: int
    overlap: int
    embedding_dim: int

    def descendant_ids(self, source_id: str) -> set[str]:
        return {chunk.chunk_id for chunk in self.chunks if chunk.source_id == source_id}


class FixedTokenChunker:
    def __init__(self, chunk_size: int = 256, overlap: int = 32):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: SourceDocument) -> list[tuple[int, int, str]]:
        tokens = tokenize(document.text)
        if not tokens:
            return []
        stride = self.chunk_size - self.overlap
        result: list[tuple[int, int, str]] = []
        for start in range(0, len(tokens), stride):
            end = min(start + self.chunk_size, len(tokens))
            result.append((start, end, " ".join(tokens[start:end])))
            if end == len(tokens):
                break
        return result


class HashingEmbedder:
    """Dependency-light deterministic lexical embedder for tests and smoke pilots.

    This is intentionally not a paper-result model. Production experiments use a
    frozen sentence-transformer revision behind the same encode interface.
    """

    def __init__(self, dim: int = 1024):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.fingerprint = stable_hash(
            f"HashingEmbedder:v1|dim={dim}|normalize=l2|tokenizer=normalize_text_v1"
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                raw = int.from_bytes(digest, "little")
                index = raw % self.dim
                sign = 1.0 if (raw >> 63) == 0 else -1.0
                matrix[row, index] += sign
            norm = float(np.linalg.norm(matrix[row]))
            if norm > 0:
                matrix[row] /= norm
        return matrix


def _snapshot_hash(documents: Sequence[SourceDocument]) -> str:
    payload = "\n".join(
        f"{doc.source_id}\0{normalize_text(doc.text)}\0{int(doc.trusted)}"
        for doc in sorted(documents, key=lambda item: item.source_id)
    )
    return stable_hash(payload)


def _pipeline_fingerprint(chunker: FixedTokenChunker, embedder) -> str:
    embedding_fingerprint = getattr(embedder, "fingerprint", None)
    if not isinstance(embedding_fingerprint, str) or not embedding_fingerprint:
        raise ValueError("embedder must expose a non-empty immutable fingerprint")
    return stable_hash(
        "|".join(
            [
                "pipeline:v1",
                f"chunk_size={chunker.chunk_size}",
                f"overlap={chunker.overlap}",
                f"embedding_dim={embedder.dim}",
                f"embedder={embedding_fingerprint}",
            ]
        )
    )


def _build_chunks(
    documents: Sequence[SourceDocument],
    chunker: FixedTokenChunker,
    embedder: HashingEmbedder,
) -> list[ChunkRecord]:
    raw: list[tuple[str, int, int, str]] = []
    for document in sorted(documents, key=lambda item: item.source_id):
        for start, end, text in chunker.split(document):
            raw.append((document.source_id, start, end, text))
    vectors = embedder.encode([item[3] for item in raw])
    chunks: list[ChunkRecord] = []
    for (source_id, start, end, text), vector in zip(raw, vectors):
        text_digest = stable_hash(text)[:12]
        chunk_id = f"{source_id}:{start}:{end}:{text_digest}"
        chunks.append(ChunkRecord(chunk_id, source_id, start, end, text, vector))
    return chunks


def build_index(
    documents: Sequence[SourceDocument],
    chunker: FixedTokenChunker,
    embedder: HashingEmbedder,
    *,
    build_seed: int = 0,
) -> IndexSnapshot:
    source_hash = _snapshot_hash(documents)
    pipeline_fingerprint = _pipeline_fingerprint(chunker, embedder)
    chunks = tuple(_build_chunks(documents, chunker, embedder))
    build_id = stable_hash(
        f"{source_hash}|{pipeline_fingerprint}|{build_seed}"
    )[:16]
    return IndexSnapshot(
        source_hash=source_hash,
        build_id=build_id,
        pipeline_fingerprint=pipeline_fingerprint,
        chunks=chunks,
        chunk_size=chunker.chunk_size,
        overlap=chunker.overlap,
        embedding_dim=embedder.dim,
    )


def apply_repair(
    documents: Sequence[SourceDocument], action: RepairAction
) -> list[SourceDocument]:
    source_ids = {document.source_id for document in documents}
    if action.source_id not in source_ids:
        raise KeyError(f"unknown source_id: {action.source_id}")
    if action.is_document_delete:
        return [doc for doc in documents if doc.source_id != action.source_id]
    if action.kind not in {"delete_span", "replace_span", "mask_span"}:
        raise ValueError(f"unsupported repair kind: {action.kind}")
    if action.start_token is None or action.end_token is None:
        raise ValueError("span repair requires start_token and end_token")
    repaired: list[SourceDocument] = []
    for document in documents:
        if document.source_id != action.source_id:
            repaired.append(document)
            continue
        tokens = tokenize(document.text)
        if not (0 <= action.start_token <= action.end_token <= len(tokens)):
            raise ValueError("repair span is outside the normalized token sequence")
        if action.kind == "mask_span":
            width = action.end_token - action.start_token
            replacement = ["[REMOVED]"] * width
        elif action.kind == "replace_span":
            replacement = tokenize(action.replacement)
        else:
            replacement = []
        edited = tokens[: action.start_token] + replacement + tokens[action.end_token :]
        repaired.append(SourceDocument(document.source_id, " ".join(edited), document.trusted))
    return repaired


def build_counterfactual_view(
    base: IndexSnapshot,
    base_documents: Sequence[SourceDocument],
    repaired_documents: Sequence[SourceDocument],
    affected_source_ids: Iterable[str],
    chunker: FixedTokenChunker,
    embedder: HashingEmbedder,
) -> IndexSnapshot:
    """F1 exact view: retain unaffected descendants and rederive affected sources."""
    affected = set(affected_source_ids)
    if _snapshot_hash(base_documents) != base.source_hash:
        raise ValueError("base snapshot does not match base_documents")
    expected_pipeline = _pipeline_fingerprint(chunker, embedder)
    if base.pipeline_fingerprint != expected_pipeline:
        raise ValueError("base snapshot pipeline fingerprint does not match configuration")
    base_by_id = {document.source_id: document for document in base_documents}
    repaired_by_id = {document.source_id: document for document in repaired_documents}
    if len(base_by_id) != len(base_documents) or len(repaired_by_id) != len(
        repaired_documents
    ):
        raise ValueError("source_id values must be unique")
    changed = set(base_by_id) ^ set(repaired_by_id)
    for source_id in set(base_by_id) & set(repaired_by_id):
        before = base_by_id[source_id]
        after = repaired_by_id[source_id]
        if (normalize_text(before.text), before.trusted) != (
            normalize_text(after.text),
            after.trusted,
        ):
            changed.add(source_id)
    if changed - affected:
        missing = ", ".join(sorted(changed - affected))
        raise ValueError(f"changed sources omitted from affected_source_ids: {missing}")
    retained = [chunk for chunk in base.chunks if chunk.source_id not in affected]
    changed_docs = [doc for doc in repaired_documents if doc.source_id in affected]
    rebuilt = _build_chunks(changed_docs, chunker, embedder)
    chunks = tuple(
        sorted(
            retained + rebuilt,
            key=lambda item: (
                item.source_id,
                item.start_token,
                item.end_token,
                item.chunk_id,
            ),
        )
    )
    source_hash = _snapshot_hash(repaired_documents)
    build_id = stable_hash(f"f1|{source_hash}|{base.build_id}")[:16]
    return IndexSnapshot(
        source_hash,
        build_id,
        expected_pipeline,
        chunks,
        chunker.chunk_size,
        chunker.overlap,
        embedder.dim,
    )


def retrieve(
    query: str,
    index: IndexSnapshot,
    embedder: HashingEmbedder,
    *,
    top_k: int = 5,
) -> list[tuple[ChunkRecord, float]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not index.chunks:
        return []
    query_vector = embedder.encode([query])[0]
    scored = [(chunk, float(np.dot(query_vector, chunk.vector))) for chunk in index.chunks]
    scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
    return scored[:top_k]


def contains_attack(chunks: Iterable[ChunkRecord], target: str) -> bool:
    marker = f"ATTACK[{target}]"
    return any(marker in chunk.text for chunk in chunks)


def source_token_count(documents: Sequence[SourceDocument], source_id: str) -> int:
    for document in documents:
        if document.source_id == source_id:
            return len(tokenize(document.text))
    return 0
