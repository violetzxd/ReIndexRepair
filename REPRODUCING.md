# Reproduction tiers

## 1. Public evidence verification

Run `python scripts/verify_public_release.py` from the repository root.
This uses all 846 canonical records, not just eligible or complete outputs.
It recomputes primary and sensitivity results with the original pure analysis
functions and compares them against the frozen aggregate JSON. Independent
A/B three-class agreement includes Yes/No/Unclear judgments before adjudication.
The original finalizer is unchanged and still requires private freeze inputs;
the public entry point deliberately verifies the **public derivative** instead.

## 2. Reconstruct the exact source snapshots

Download and extract the official ZIP:

https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip

ZIP SHA-256: `2553bf7bfbab47b1436ca00a34bce57320e18e611fd00999a2a3a1b4714be752`.
Corpus SHA-256: `f7e3588dd416772c9830baf67e827d49678f8d5a6ccf6e101b9ce990cc7b208c`.

```sh
python scripts/rebuild_public_snapshot.py --corpus /path/to/nq/corpus.jsonl \
  --spec datasets/snapshots/100k.json --output datasets/nq_certification_snapshot_100k_v1/corpus.jsonl
python scripts/rebuild_public_snapshot.py --corpus /path/to/nq/corpus.jsonl \
  --spec datasets/snapshots/1m.json --output datasets/nq_scale_snapshot_1m_v1/corpus.jsonl
```

The script verifies the source hash, uses ordered IDs and disk-backed SQLite,
serializes each row as `json.dumps(record, ensure_ascii=False, sort_keys=True)`
plus an LF newline, then verifies the exact final corpus hash. Allow several
GB of disk space. It refuses to overwrite an existing output. Corpus rebuilding
requires no model, unpublished retrieval cache, random resampling or API call.

## 3. Actual full-pipeline reruns on a GPU

Install `.[real]` in a compatible GPU environment. Download the three exact
model revisions from `configs/sources.lock.yaml` to local directories. Set
`PYTHONPATH=src` and use explicit paper settings; runner defaults are sometimes
development settings, especially generation length.

For substitution, run `scripts/run_full_pipeline_dev.py` twice with common:

```text
--corpus datasets/nq_certification_snapshot_100k_v1/corpus.jsonl
--queries datasets/nq_certification_queries_v1.json
--incidents datasets/nq_certification_incidents_v1.json
--embedder-path <BGE embedding directory> --reranker-path <BGE reranker directory>
--generator-path <Qwen directory> --cache-dir <new local cache>
--device cuda --retrieve-k 30 --serve-k 5 --poison-multiplicity 5
--limit 94 --max-new-tokens 96 --generation-batch-size 1
--materialize-f2-faiss
--f2-delete-policy visible   [first run]
--f2-delete-policy all_owner [second run]
--output <new output for each run>
```

Do not pass `--paper-evidence` to imply that a new unannotated output is already
human-certified. Re-label new responses if generation differs. For boundary
and oracle runners use `--chunk-size 128`, `--retrieve-k 30`, `--serve-k 5`,
`--max-new-tokens 96`, `--generation-batch-size 1`, limits 50 and 30 respectively.
Every runner provides `--help`. The independent-certificate runner additionally
requires the clean and query vector caches created by preceding runs and the
frozen reference lattice artifact. `verify_repair_lattice_independent_cert.py`
checks physical input/model/cache fingerprints on actual reruns; sanitized
artifact paths are provenance descriptions, not live local paths.

No claimed deployed selector is supplied: the repair choice in this study is an
exhaustive human-label oracle on the same triggering question. To extend this
work, introduce a held-out semantic selector and an independent test set rather
than interpreting the existing oracle's zero regret as generalization.
