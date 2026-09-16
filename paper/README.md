# Editable ICASSP 2027 LaTeX draft

Compile from this directory with a standard TeX installation:

```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The archive is self-contained: official spconf style, IEEEbib bibliography
style, bibliography, frozen generated result macros, TikZ method figure,
vector result figure, and the checked five-page PDF are included. The first
four pages hold technical content and AI/ethics disclosures; page five holds
24 references. Template notices retain their upstream terms.

Public code and de-identified evidence: https://github.com/violetzxd/ReIndexRepair
Original code is MIT; third-party data/model terms are separate.

This is an AI-assisted research draft, not an accepted or unconditionally
submission-ready paper. Real authors must substantively compose/revise and
verify it under the ICASSP 2027 LLM policy and complete names, affiliations
and ORCIDs. Filling authors or revising text requires a fresh page-limit,
font-embedding and official upload-time PDF check. Do not remove a truthful
AI disclosure as a substitute for substantive author work.
