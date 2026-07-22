# Experiment 01: Addressing OpenReview Feedback — Master Plan

**TL;DR:** Consolidated plan to address all reviewer concerns from ICLR 2024 (reject), ICLR 2025 (reject), and workshop acceptances (ICML 2023, DMLR@ICLR 2024). 10 tasks total, 5 critical experiments + 3 high-priority writing fixes + 2 medium improvements.

---

## Submission History

| Venue | Decision | Forum |
|-------|----------|-------|
| ICML 2023 Workshop (DeployableGenerativeAI) | Accept | https://openreview.net/forum?id=oCYjN48axE |
| DMLR @ ICLR 2024 Workshop | Accept (Poster) | https://openreview.net/forum?id=tgkWxsOapD |
| ICLR 2024 Main Track | Reject | https://openreview.net/forum?id=506Sxc0Adp |
| ICLR 2025 Main Track | Reject | https://openreview.net/forum?id=kDakBhOaBV |

---

## Sub-experiments

| # | Experiment | Priority | Status |
|---|-----------|----------|--------|
| 02 | Baseline diversity metrics (Vendi, N-gram, embedding) | CRITICAL | Scripts written |
| 03 | Downstream benchmarks (ARC, HellaSwag, WinoGrande, LAMBADA) | CRITICAL | Scripts written |
| 04 | New datasets div coeff (FineWeb, Dolma, RedPajama) | CRITICAL | Scaffold ready |
| 05 | Confounding ablations (size, domain, vocab overlap) | CRITICAL | Scaffold ready |
| 06 | GPT-4 annotation validation | MEDIUM | Scaffold ready |

## Writing tasks (no separate experiment folder needed)

| # | Task | Priority |
|---|------|----------|
| 6 | Improve Task2Vec methodology + pseudocode in 02_method.tex | HIGH |
| 7 | Update related work with 10+ missing citations | HIGH |
| 8 | Tone down overclaiming, remove "paradigm shift" | HIGH |
| 10 | Fix 18+ presentation issues | MEDIUM |
| 4 | Address inconsistent diversity→performance in 05_discussion.tex | CRITICAL |

---

## Execution order

1. **Expt 03** (downstream benchmarks) — run overnight, mostly automated lm_eval
2. **Expt 02** (baseline metrics) — run in parallel on different GPU
3. **Writing tasks 6, 7, 8** — can be done while experiments run
4. **Expt 04** (new datasets) — straightforward, uses existing API
5. **Expt 05** (confounders) — requires some new training
6. **Writing task 4** (failure analysis) — needs experiment results first
7. **Expt 06** (GPT-4 annotation) — lower priority, quick to run
8. **Writing task 10** (formatting) — do last, after all content changes

---

## 2026-07-21 update — reviewer-complaint closure pass (DMLR 2026 draft)

Paper edits in `paper_latex/DMLR_2026_BeyondScale/` addressing the remaining complaints:

1. **Benchmark evals (ICLR 2025 JTBn/N6rW/Bhvz + meta-review):** Added Appendix "Downstream Benchmark Evaluations Beyond Cross-Entropy" (`appendix:downstream_benchmarks`) presenting Experiment 03 results honestly: 27 models × {ARC-Easy, HellaSwag, WinoGrande, LAMBADA}, low absolute accuracy, and no significant Spearman correlation. The paper reports this as a null result and presents a floor effect as a plausible explanation rather than an established cause. The Discussion limitation paragraph now cites it.
2. **Shannon entropy (preemptive):** Added Appendix "Why Shannon Token Entropy is Unsuitable as a Measure of Training-Data Diversity" (`appendix:entropy_not_diversity`): (a) a uniform marginal maximizes token entropy, and i.i.d. uniform sampling gives the paper's own degenerate upper-bound dataset; (b) token entropy is permutation-invariant and blind to sequence structure; and (c) practical entropy-rate estimates require a predictive model or compressor. The upper-bound recipe in `02_method.tex` cross-references the discussion.
3. **Baseline comparison (all reviewers):** Added Appendix "Empirical Comparison with Simpler Diversity Baselines" (`appendix:baseline_metrics`) with Experiment 02 results: n-gram diversity agrees with the diversity coefficient on the extremes (ρ=0.60), but the tested GPT-2-kernel Vendi Score (ρ=0.03) ranks USPTO above C4 and GPT-2 mean-embedding cosine (ρ=0.26) ranks C4 lowest. Fixed the dangling `\ref{sec:experiments}` in related work and the "Vendi comparison is future work" contradiction.
4. **Embedding backbone:** The Experiment 02 raw run and implementation use pretrained GPT-2 embeddings, although its results-summary file incorrectly says all-MiniLM-L6-v2. The paper now states the actual GPT-2 backbone and cites SBERT (`reimers2019sentence`) only as an alternative sentence-embedding approach; a true SBERT baseline would require a new experiment.

5. **Benchmark metric clarified (Brando's question):** The Exp 03 numbers are **accuracy** (`acc_norm` for ARC-Easy/HellaSwag, `acc` for WinoGrande/LAMBADA), *not* continuous logprob(correct)-style metrics. The appendix now states this explicitly and cites Schaeffer et al. 2024 ("elusive benchmarks", `schaeffer2024elusive`): accuracy's argmax discretization destroys small-scale signal that continuous log-likelihood metrics (our CE evaluation) retain. Re-scoring the four benchmarks with continuous per-choice metrics is flagged as future work — a cheap, high-value pre-submission strengthening if desired (lm-eval already logs per-choice loglikelihoods).
6. **OpenReview sweep (2026-07-21):** Exhaustive web sweep confirms the 4 known forums are the complete set (ICML23 wkshp, DMLR@ICLR24 wkshp, ICLR24, ICLR25); no public TMLR submission exists; one extra non-OpenReview acceptance (DMLR wkshp @ ICML 2023, dmlr.ai entry 113). Across all 8 reviews: zero SBERT/entropy-baseline mentions; embedding-baseline asks were generic ("other textual embedding methods" — v5Te; "mean GPT-2 last layer embedding" — N6rW, which is exactly what Exp 02 ran).

Also fixed pre-existing bibtex build error (duplicate `\bibliographystyle` in `00_dmlr_beyond_scale.tex`; the class file already sets it), and corrected the Exp 02 results-summary config table (said all-MiniLM-L6-v2; actual backbone is GPT-2 per `compute_baseline_metrics.py:353` and the run log).

## Reference files

```
experiments/01_addressing_openreviews/deep_research.md         # ChatGPT + Gemini analysis
experiments/01_addressing_openreviews/iclr2024_reviews.md      # ICLR 2024 reviews
experiments/01_addressing_openreviews/iclr2025_reviews.md      # ICLR 2025 reviews
experiments/01_addressing_openreviews/00_master_suggested_prompts.md  # original prompt list
```
