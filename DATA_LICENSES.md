# Data provenance and licensing

The root MIT license applies to original project software and documentation,
not to third-party corpora, model weights, or paper-template files.

## Natural Questions and BEIR

The experiments use the Natural Questions (NQ) corpus as distributed by BEIR.
Natural Questions derives from Wikipedia and is distributed under CC BY-SA 3.0;
retain its source attribution and share-alike terms for NQ-derived questions,
answers, titles and passages in research fixtures and evaluation records.
This release does not claim to relicense those fields under MIT.

- Kwiatkowski et al., *Natural Questions: A Benchmark for Question Answering
  Research*, TACL 2019. [Official source and terms](https://ai.google.com/research/NaturalQuestions/download).
- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of
  Information Retrieval Models*, NeurIPS 2021.
  [Dataset provenance](https://github.com/beir-cellar/beir/wiki/Datasets-available).
- [Wikipedia attribution/history](https://en.wikipedia.org/): document IDs and
  titles in the officially downloaded BEIR corpus identify upstream articles.
- [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

Full 100k/1M source passages are not uploaded. `datasets/snapshots/` contains
only ordered document IDs and hash-bound reconstruction specifications.
`datasets/nq_certification_incidents_v1.json` contains 94 incidents, each with
five frozen DeepSeek-generated attack passages (470 total), not 470 copied NQ
corpus passages. Their deliberately false claims are controlled research
fixtures, not factual answers. The combined NQ-derived research-data bundle is
provided under CC BY-SA 3.0, with the attribution above; model-output fields are
clearly identified as synthetic outputs, without a claim of exclusive rights.
Do not use these fixtures as reliable reference material.

## Models and dependencies

Model identifiers and exact revisions are in `configs/sources.lock.yaml`.
Download BAAI/bge-base-en-v1.5, BAAI/bge-reranker-base and
Qwen/Qwen2.5-7B-Instruct from their official repositories and follow their
respective model-card licenses. No model weights, external repository copies,
API credentials, or retrieval caches are included here.

## Human evaluation and privacy

`artifacts/human_labels_public.json` retains canonical statistical item IDs,
the identical generated response, final three Boolean decisions, anonymous
independent A/B decisions, and an adjudication flag. It removes annotator
names, confidence, free-text notes, workbook coordinates, blind item codes,
private annotation keys and server/user paths. IDs identify experimental
conditions, not annotators. The original private freeze and verification gate
remain unchanged: this public derivative has its own manifest and hashes.

## Paper template

IEEE `spconf.sty`, `IEEEbib.bst`, and other template files retain their upstream
notices and are not covered by this project's MIT license. The manuscript is
an AI-assisted research draft, not an accepted ICASSP paper.
