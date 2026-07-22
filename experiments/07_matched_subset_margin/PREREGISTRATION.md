# Experiment 07 — Pre-registration: Matched-Subset Per-Choice Margin Analysis

**Committed BEFORE computing any results (git timestamp is the proof). Confirmatory tests and subject groups are fixed here; anything not listed is exploratory and will be labeled as such.**

**Date:** 2026-07-21
**Data (already collected, never analyzed at subject level):** existing lm-eval-harness per-sample JSONLs on SNAP DFS `/dfs/scratch0/brando9/data/beyond_scale/eval_results/` — MMLU per-subject runs (2024-09-30) and ARC-Easy/HellaSwag/WinoGrande/LAMBADA runs (2026-04-04→06) for the 27 UDACA checkpoints (GPT-2 51M–1.5B families + LLaMA-2 7B), trained on USPTO (div 0.158), PubMed Abstracts (0.168), USPTO+PubMed (0.195). No new model evaluations are run; this is pure re-scoring.

## Metrics (per question)

From `filtered_resps` per-choice log-likelihoods and `target` gold index:

- **logpcc** = logP(correct choice)
- **margin** = logpcc − mean over incorrect choices of logP(incorrect)  [PRIMARY]
- Secondary: length-normalized variants (each choice log-likelihood divided by its character length, as in lm-eval `acc_norm`).

Primary decides; secondary reported for robustness.

## Subject groups (fixed a priori)

- **MED** (PubMed-aligned MMLU subjects): `clinical_knowledge, college_medicine, professional_medicine, college_biology, medical_genetics, anatomy`
- **GEN**: all 57 MMLU subjects.
- No confirmatory USPTO-matched group exists (no good patent-domain MMLU subjects); legal-adjacent subjects (`jurisprudence, international_law, professional_law`) are exploratory only.

## Units of pairing

Within-family triplets (USPTO, PubMed, mix), aggregating LLaMA-2 checkpoints by condition mean. MMLU: 6 complete GPT-2 triplets (117M, 204M, 345M, 51M-1.31B, 51M-557M, 1.5B; 810M lacks a USPTO MMLU run) + 1 LLaMA triplet = 7. Downstream: all complete triplets found in the data (expected 7 GPT-2 + 1 LLaMA = 8). Family counts will be reported exactly as found; if a family is incomplete it is dropped from pairing (not imputed).

## Confirmatory tests (all exact paired sign tests across family triplets; report all outcomes regardless of direction)

- **T1 — Instrument positive control (alignment):** On MED margin, PubMed vs USPTO, one-sided (alignment predicts PubMed > USPTO). If T1 fails, the instrument has no detectable power at this scale and T2/T3 are interpreted accordingly.
- **T2 — Diversity vs alignment discrimination:** On MED margin, mix vs PubMed, two-sided. Diversity predicts mix > PubMed (0.195 > 0.168); alignment predicts PubMed > mix (100% vs 50% medical).
- **T3 — Diversity on the general pool:** On GEN margin, (a) mix vs USPTO and (b) mix vs PubMed, one-sided (diversity predicts mix wins both).
- **T4 — Diversity on downstream benchmarks:** margin per benchmark (ARC-Easy, HellaSwag, WinoGrande; LAMBADA has no incorrect choices → logpcc only), mix vs each component, one-sided as in T3.

## Uncertainty

Per-question nonparametric bootstrap (10,000 resamples) → 95% CIs on each model's group-mean margin. Sign tests as above at family level. Rank correlations (Spearman across all models) reported for continuity with the paper but are NOT confirmatory (only 3 predictor levels).

## Hygiene items bundled in

- Rebuild MMLU metric table directly from raw JSONLs (replaces the copy-suspect `mmlu_results.csv`; resolves the duplicated-row issue flagged in EXPERIMENT_STATE.md §6).
- Audit LLaMA LAMBADA = 0.000 from raw samples.

## Interpretation rules (stated in advance)

- T1 pass + T3 fail: instrument works; diversity effect not detectable on question-level metrics at this scale → keep the paper's current framing, add positive control.
- T1 pass + T3 pass: first question-level evidence for the diversity claim → add to paper as supporting evidence with CIs.
- T1 fail: benchmarks carry no signal for these models even in-domain with continuous metrics → strengthens the floor-effect explanation; report as such.
- T2 is reported descriptively as discriminating evidence either way (diversity- vs alignment-consistent direction).
