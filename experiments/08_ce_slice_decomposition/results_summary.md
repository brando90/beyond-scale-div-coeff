# Exp 08 — CE-Slice Decomposition: Results (INCONCLUSIVE — not added to paper)

**TL;DR:** The pre-registered decomposition **cannot answer the confounder question on C4** because C4's validation set is **98% general web text** (of 20,000 docs: NEITHER=19,567, MED=427, PATENT=6) — there are too few domain-matched documents to test whether the mix's cross-entropy advantage concentrates on matched vs unmatched slices. On the NEITHER slice the mix is directionally better than both components (paired family sign test: mix < USPTO in 5/7 families p=0.23; mix < PubMed in 6/8 p=0.14) — consistent with the paper's finding and with a diversity (not similarity) reading, but **not statistically significant**, and by my pre-registered rule ("mixed/uncertain → report descriptively, no claim upgrade") **this experiment is not added to the paper.**

Pre-registration: `PREREGISTRATION.md` (committed 8fbdb75, before results). New CE computation on 20,000 C4-val docs (first 1024 tokens), all 27 UDACA checkpoints, one A100 each (parallelized across 6 GPUs).

## Confirmatory tests (pre-registered S1; paired family sign tests)

| Slice | Contrast | wins/n | p (1-sided) | Note |
|---|---|---|---|---|
| **NEITHER** (n=19,567) | mix < USPTO | 5/7 | 0.23 | directional, ns |
| **NEITHER** | mix < PubMed | 6/8 | 0.14 | directional, ns |
| MED (n=427) | mix < USPTO | 5/7 | 0.23 | small slice |
| MED | mix < PubMed | 4/8 | 0.64 | null |
| PATENT (n=6) | — | — | — | **excluded** (<500-doc pre-reg threshold; 6 docs is noise) |

## Why this is inconclusive (honest limitations)

1. **The decomposition premise fails on C4.** The test needs eval documents that match each training domain to check whether the mix's advantage concentrates there (similarity) or is broad (diversity). C4-val has essentially no patent docs (6) and few medical docs (427), so "concentration on matched slices" cannot be measured — NEITHER ≈ the whole eval set.
2. **Per-condition mean CE is composition-confounded — do NOT read it as a result.** Mean CE over models is USPTO 6.61 < PubMed 6.94 < mix 7.68 on NEITHER, which *looks* like mix is worst, but this is purely because the mix condition contains more short-budget (6M-token) LLaMA-2 checkpoints (high CE) than the USPTO condition. Within families, the mix is better in most cases (5/7, 6/8) — this is why the pre-registered test is **paired within family**, not a pooled mean.
3. **Protocol differs from the paper's headline figure.** This is a rough 20k-doc / 1024-token re-scoring; the paper's Figure~\ref{fig:div_vs_val_ce} uses the full validation sets and the >150M-token GPT-2 subset. The two are not directly comparable, and the two families where mix loses to PubMed here (204M, 810M) reflect that plus real run-to-run noise the reviewers already noted.

## What to take away

- **For the paper:** nothing added (per pre-registered rule). The existing vocabulary-overlap ablation (frequency-weighted Jaccard *decreases* for the mix) remains the paper's confounder rebuttal.
- **Methodological finding worth keeping:** a domain-slice CE decomposition needs eval sets with substantial matched-domain subsets; C4/OpenWebText2 are ~98% general web text and cannot support it. A cleaner future test would evaluate on a benchmark with explicit medical/legal/general partitions (e.g., MMLU-by-subject CE, or domain-labeled corpora), where slice sizes are balanced.

## Artifacts (`expt_results/`)

`ce_slices.csv` (per model×slice mean CE + bootstrap 95% CIs), `sign_tests.json`, `slice_counts.json`.
