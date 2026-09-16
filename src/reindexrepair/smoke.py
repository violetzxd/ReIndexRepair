from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

from .core import (
    ChunkRecord,
    FixedTokenChunker,
    HashingEmbedder,
    RepairAction,
    SourceDocument,
    apply_repair,
    build_index,
    contains_attack,
    retrieve,
)
from .metrics import RepairOutcome, summarize_mirage


@dataclass(frozen=True)
class SmokeScenarioResult:
    scenario: str
    incident_id: str
    original_attack: bool
    f0_safe: bool
    f2_safe: bool
    original_source: str | None
    post_rebuild_source: str | None
    mechanism: str


def _top_chunk(
    query: str, documents: Sequence[SourceDocument], chunker: FixedTokenChunker,
    embedder: HashingEmbedder,
) -> ChunkRecord | None:
    index = build_index(documents, chunker, embedder)
    hits = retrieve(query, index, embedder, top_k=1)
    return hits[0][0] if hits else None


def _scenario_rank_substitute(index: int) -> SmokeScenarioResult:
    target = f"wrong-{index}"
    term = f"topic-{index}"
    query = f"{term} {term}"
    documents = [
        SourceDocument("a-primary", f"{term} {term} ATTACK[{target}]"),
        SourceDocument("b-backup", f"{term} ATTACK[{target}]"),
        SourceDocument("z-clean", f"{term} verified"),
    ]
    chunker = FixedTokenChunker(chunk_size=8, overlap=0)
    embedder = HashingEmbedder(dim=4096)
    original = _top_chunk(query, documents, chunker, embedder)
    # F0 drops the sole logged prompt chunk. F2 performs the corresponding
    # source-level quarantine, causing an unseen backup poison to substitute.
    f0_safe = True
    action = RepairAction(f"drop-primary-{index}", "a-primary", "delete_document")
    repaired = apply_repair(documents, action)
    rebuilt = _top_chunk(query, repaired, chunker, embedder)
    return SmokeScenarioResult(
        "rank_substitute",
        f"rank-{index}",
        bool(original and contains_attack([original], target)),
        f0_safe,
        not bool(rebuilt and contains_attack([rebuilt], target)),
        original.source_id if original else None,
        rebuilt.source_id if rebuilt else None,
        "complete source deletion changes the next retrieved candidate",
    )


def _scenario_boundary_migration(index: int) -> SmokeScenarioResult:
    target = f"wrong-{index}"
    term = f"boundary-{index}"
    query = f"{term} {term}"
    # The repair removes the first four-token poisoned region. Re-chunking then
    # promotes the second source region, which was outside the original top-1.
    documents = [
        SourceDocument(
            "a-mixed",
            f"{term} {term} ATTACK[{target}] pad "
            f"{term} {term} ATTACK[{target}] tail verified benign fact",
        ),
        SourceDocument("z-clean", f"{term} verified evidence"),
    ]
    chunker = FixedTokenChunker(chunk_size=4, overlap=0)
    embedder = HashingEmbedder(dim=4096)
    original = _top_chunk(query, documents, chunker, embedder)
    f0_safe = True
    action = RepairAction(f"remove-region-{index}", "a-mixed", "delete_span", 0, 4)
    repaired = apply_repair(documents, action)
    rebuilt = _top_chunk(query, repaired, chunker, embedder)
    return SmokeScenarioResult(
        "boundary_migration",
        f"boundary-{index}",
        bool(original and contains_attack([original], target)),
        f0_safe,
        not bool(rebuilt and contains_attack([rebuilt], target)),
        original.source_id if original else None,
        rebuilt.source_id if rebuilt else None,
        "source edit changes chunk boundaries and promotes a sibling region",
    )


def _scenario_no_mirage(index: int) -> SmokeScenarioResult:
    target = f"wrong-{index}"
    term = f"control-{index}"
    query = f"{term} {term}"
    documents = [
        SourceDocument("a-poison", f"{term} {term} ATTACK[{target}]"),
        SourceDocument("z-clean", f"{term} verified evidence"),
    ]
    chunker = FixedTokenChunker(chunk_size=8, overlap=0)
    embedder = HashingEmbedder(dim=4096)
    original = _top_chunk(query, documents, chunker, embedder)
    repaired = apply_repair(
        documents, RepairAction(f"drop-poison-{index}", "a-poison", "delete_document")
    )
    rebuilt = _top_chunk(query, repaired, chunker, embedder)
    return SmokeScenarioResult(
        "no_mirage_control",
        f"control-{index}",
        bool(original and contains_attack([original], target)),
        True,
        not bool(rebuilt and contains_attack([rebuilt], target)),
        original.source_id if original else None,
        rebuilt.source_id if rebuilt else None,
        "complete lineage deletion with no backup poison",
    )


def run_smoke(incidents_per_scenario: int = 30) -> dict:
    results: list[SmokeScenarioResult] = []
    for index in range(incidents_per_scenario):
        results.extend(
            [
                _scenario_rank_substitute(index),
                _scenario_boundary_migration(index),
                _scenario_no_mirage(index),
            ]
        )
    if not all(item.original_attack for item in results):
        raise AssertionError("smoke corpus failed to trigger every intended original attack")
    outcomes = [
        RepairOutcome(
            item.incident_id,
            "q0",
            item.f0_safe,
            item.f2_safe,
            0.0 if item.f0_safe else 1.0,
            0.0 if item.f2_safe else 1.0,
        )
        for item in results
    ]
    summary = summarize_mirage(outcomes)
    by_scenario: dict[str, dict] = {}
    for scenario in sorted({item.scenario for item in results}):
        subset = [item for item in results if item.scenario == scenario]
        scenario_outcomes = [
            outcome for outcome in outcomes if outcome.incident_id in {item.incident_id for item in subset}
        ]
        by_scenario[scenario] = {
            "count": len(subset),
            "f0_safe": sum(item.f0_safe for item in subset),
            "f2_safe": sum(item.f2_safe for item in subset),
            "metrics": asdict(summarize_mirage(scenario_outcomes)),
        }
    return {
        "tier": "smoke",
        "paper_evidence": False,
        "warning": "Deterministic implementation smoke test; rates are constructed, not empirical prevalence.",
        "incidents_per_scenario": incidents_per_scenario,
        "overall": asdict(summary),
        "by_scenario": by_scenario,
        "records": [asdict(item) for item in results],
    }


def main() -> None:
    output = Path("artifacts/smoke_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = run_smoke()
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "by_scenario": payload["by_scenario"]}, indent=2))


if __name__ == "__main__":
    main()
