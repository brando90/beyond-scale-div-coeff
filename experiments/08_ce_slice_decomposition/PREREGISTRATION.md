# Experiment 08 — Pre-registration: Cross-Entropy Slice Decomposition

**Committed BEFORE computing any results (git timestamp is the proof). Targets the standing ICLR-2025 confounder objection (reviewer N6rW / meta-review): "does the mix win on C4/OWT2 CE because of diversity, or because it is more similar to the eval data?"**

**Date:** 2026-07-21

## Design

Score all 27 UDACA checkpoints' per-token cross-entropy on evaluation documents partitioned into three fixed slices, then test where the mix model's CE advantage lives.

- **Predictions:** *Diversity* predicts the mix's advantage over each component is present on the NEITHER slice (general capability). *Alignment* predicts the advantage concentrates on the slice matching the added component (e.g., mix beats USPTO mainly on MED docs) and vanishes on NEITHER.

## Evaluation data

- C4 (en) validation split (primary; same eval set as the paper) and OpenWebText2 validation if available on DFS/HF (secondary).
- Sample size: up to 20,000 documents per eval set (fixed seed 42), truncated/packed at sequence length 1024, GPT-2 tokenizer for GPT-2 models and LLaMA tokenizer for LLaMA models.

## Slices (document-level, fixed keyword lexicons, decided a priori)

Case-insensitive word-boundary matches per 1,000 whitespace tokens:

- **MED lexicon:** patient(s), clinical, medical, diagnosis, treatment, disease(s), syndrome, therapy, physician, symptom(s), dose, tumor, infection, chronic, cardiovascular
- **PATENT lexicon:** embodiment(s), "prior art", wherein, apparatus, invention, claim(s), "disclosed herein", patentable, assignee, "field of the invention"
- **Rule:** doc → MED if ≥3 distinct MED-lexicon hits and MED hits > PATENT hits; doc → PATENT symmetric; else NEITHER. If a matched slice has <500 docs in an eval set, report it descriptively and exclude it from confirmatory tests (NEITHER will dominate by construction; that is fine — it is the slice that matters).

## Metric

Mean per-token cross-entropy (natural log) per (model, slice), computed with non-overlapping 1024-token windows, bf16 on A100.

## Confirmatory tests (exact paired sign tests across family triplets, as in Exp 07)

- **S1 — Diversity:** On the NEITHER slice of C4-val: mix vs USPTO and mix vs PubMed, one-sided (diversity predicts mix better on both).
- **S2 — Alignment-concentration check (descriptive-confirmatory):** For each component contrast, compare the mix's CE advantage on the matched slice vs the NEITHER slice; alignment predicts matched ≫ NEITHER, diversity predicts comparable advantages.

## Uncertainty

Per-document bootstrap (10,000 resamples) → 95% CIs per (model, slice) mean CE; family-level sign tests as primary inference.

## Interpretation rules (stated in advance)

- S1 pass: the mix's advantage exists on documents matching NEITHER training domain → directly rebuts the similarity-confounder objection; add to paper.
- S1 fail with advantage concentrated on matched slices (S2 alignment-pattern): honest update — the paper's domain-control argument weakens; report faithfully and adjust the discussion's confounder paragraph.
- Mixed/uncertain: report descriptively with CIs; no claim upgrade.
