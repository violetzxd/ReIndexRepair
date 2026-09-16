# ReIndexRepair

State-consistent evaluation of source repair in poisoned retrieval-augmented
generation (RAG). This repository accompanies an **ICASSP 2027 research draft**;
it is not an accepted paper and does not implement a new deployed poison
locator or semantic repair selector.

## What is new

Repair is evaluated on reconstructed serving state, not just on the evidence
removed from a previously observed trace. The paired experiments isolate
hidden-copy promotion and chunk-boundary migration from stale-vector effects.
Root-lineage closure invalidates old descendants of edited sources; an exact
counterfactual view is checked against a separately rebuilt FAISS index.
An exhaustive, same-question human oracle assesses repair feasibility and
minimum-token cost, explicitly retaining abstentions.

## Frozen evidence

| Evaluation | Result and scope |
|---|---|
| Substitution | 13/71 vs 2/71 target recurrence after initial target adoption and an F0-safe response; Holm p=0.001953125 |
| Boundary migration | 7/18 vs 1/18 under the same conditioning; Holm p=0.03125 |
| Independent exact-view check | 120/120 action certificates; 270/270 original annotated answers byte-identical; maximum vector discrepancy 0 |
| Same-question oracle | 19/30 feasible; 11 abstentions; selected edit median 4 tokens |
| Human judgments | 846 outputs, independent anonymous A/B labels, 67 adjudications, 43 final incomplete outputs |
| Clean vs complete-lineage utility | Both 148/150 hit@30 and 121/150 hit@5; retrieval utility, not answer quality |
| 1M scale check | 49/88 hidden-poison promotion; retrieval-only, post-hoc sensitivity |

The least-favorable assignments of incomplete primary endpoints **do not
retain statistical significance**. Conditioned recurrence, retrieval exposure,
same-question oracle feasibility and utility are distinct endpoints. One corpus,
one generator, synthetic targets and oracle provenance limit generalization.

## Offline reproduction (no GPU or external API)

```sh
python -m venv .venv
# Activate the environment using your operating system's standard command.
python -m pip install -e ".[test]" numpy==1.26.4 matplotlib==3.9.2
python scripts/verify_public_release.py
python -m pytest -q
python scripts/make_paper_figures_submission.py \
  --human-statistics artifacts/paper_statistics_human_final_v1.json \
  --locator-sweep artifacts/locator_error_sweep_100k_v1.json \
  --output results/semantic_results.pdf
```

Verification checks the new public manifest, exact response hashes, all 846
labels, independent agreement, fixed eligibility, seeded paired bootstrap
(10,000 draws, seeds 2027/2028), exact McNemar tests, two-hypothesis Holm
correction, incomplete-endpoint sensitivity, repair costs, mechanism counts
and per-query utility against the frozen final analysis. It additionally binds
all 846 labeled responses to their mechanical artifacts, checks 120 stored
F1/F2 answer/ranking/prompt/source/vector-hash agreements and 270 reference
answers. This checks stored evidence consistency, not physical vector-cache
reexecution (which requires the GPU rerun). It does **not** claim
that sanitized files have the byte hashes of private originals or reproduce the
private annotation-key gate. Private-original commitments are separately marked.

## Source reconstruction and GPU reruns

See [REPRODUCING.md](REPRODUCING.md). Ordered IDs reconstruct both published
snapshots directly from the checksum-verified official NQ corpus, without the
unpublished hard-negative cache used when selecting them. The frozen 470 attack
passages avoid stochastic API regeneration and additional API costs.

The actual recorded GPU environment was Python 3.12.4, PyTorch 2.7.1+cu126,
CUDA 12.6, NumPy 1.26.4, FAISS 1.8.0, on an A100 80GB; runtime details and
model fingerprints are preserved in evaluation artifacts. The mechanical
certificate's old lexical-selector diagnostics (18/30, recall 0.6, and simulated
rebuild counts) are auxiliary, not the final human oracle (19/30, recall 1.0),
and are not measured online savings. Package version
ranges in `pyproject.toml` describe compatibility, not a complete historical
environment lock. Hardware/library changes may change generation bytes;
report reruns separately rather than editing frozen human judgments.

## Repository layout

- `src/`, `scripts/`, `tests/`: original implementation and tests.
- `configs/`: exact model/source revisions.
- `datasets/`: frozen attacks, candidate/query fixtures and ordered snapshot IDs.
- `artifacts/`: de-identified outputs, provenance/certificate evidence and statistics.
- `paper/`: editable IEEE-template LaTeX source and figures.
- `PUBLIC_MANIFEST.json`: SHA-256 for the exported files and private-original commitments.

Original code is [MIT licensed](LICENSE); data and model terms are separate
([DATA_LICENSES.md](DATA_LICENSES.md)). No passwords, API keys, model weights,
private annotation workbooks or server logs should be committed.

## Manuscript status and responsible use

Real authors must independently and substantively compose/revise and verify
the AI-assisted manuscript under the ICASSP 2027 LLM policy; disclosure alone
does not close that requirement. Author information is intentionally blank in
this research draft and must be completed before submission. These experiments
neither certify universal safety nor remove information from model parameters.
