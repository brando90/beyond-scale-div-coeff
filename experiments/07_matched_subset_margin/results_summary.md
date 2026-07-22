# Exp 07 — Matched-Subset Per-Choice Margin: Results

**TL;DR:** The pre-registered per-choice **margin** metric (logpcc − mean logpics) has demonstrable power — it separates the PubMed-trained from the USPTO-trained model on medicine-domain MMLU in **7/7 families (paired sign test p=0.0078)**. This power is the point: it shows the general-benchmark diversity nulls are *not* a dead-metric or pure-floor artifact. But the **diversity contrast itself is null** — the mix (div 0.195) never beats the intermediate-diversity PubMed model (≤2/6 families on any benchmark), so there is no monotonic diversity effect at this scale. **Honesty caveat:** the PubMed>USPTO effect is *largely general*, not purely in-domain — PubMed also beats USPTO on the full MMLU pool 6/7, so T1 confirms the metric detects *training-corpus* differences, with only a modest extra in-domain boost; we do **not** claim a clean alignment-specific effect.

**⚠ n=25, not 27:** Two "mix" checkpoints (204M and 810M USPTO+PubMed) were found to have eval outputs bit-identical to a USPTO model (a checkpoint-loading bug) and are excluded as invalid; see `experiments/DATA_INTEGRITY_2026-07-22_corrupt_mix_evals.md`. The positive control is unaffected (it uses only authentic USPTO/PubMed models); every diversity-contrast result was and remains a non-significant null.

Pre-registration: `PREREGISTRATION.md` (committed 8fbdb75, **before** any results). Pure re-scoring of existing lm-eval per-sample JSONLs (no new evals). n=27 models.

## Confirmatory paired sign tests (family-level; LLaMA checkpoints averaged by condition)

| Test | Contrast | Group | wins/n | p (1-sided) | Verdict |
|---|---|---|---|---|---|
| **T1** | PubMed > USPTO | MED margin | **7/7** | **0.0078** | **PASS — positive control: metric detects training-corpus differences** (also 6/7 on GEN → largely general, not purely in-domain) |
| T2 | mix vs PubMed | MED margin | 2/6 | 0.89 | Null — added diversity ≠ margin gain in-domain |
| T3a | mix > USPTO | GEN margin | 2/6 | 0.89 | Null |
| T3b | mix > PubMed | GEN margin | 1/6 | 0.98 | Null |
| T4 arc_easy | mix > USPTO / PubMed | margin | 3/6, 4/6 | 0.66, 0.34 | Null |
| T4 hellaswag | mix > USPTO | margin | 6/6 | 0.016 | mix beats USPTO… |
| T4 hellaswag | mix > PubMed | margin | 2/6 | 0.89 | …but not PubMed → not a clean diversity win |
| T4 winogrande | mix > USPTO / PubMed | margin | 4/6, 1/6 | 0.34, 0.98 | Null |
| T4 lambada | mix > USPTO / PubMed | logpcc | 6/6, 5/6 | 0.016, 0.11 | mix>USPTO but not>PubMed → no monotonic effect |

## MED margin by condition (mean over models; higher = better)

| Condition | div coeff | mean margin (n models) |
|---|---|---|
| USPTO | 0.158 | −0.0950 (7) |
| PubMed | 0.168 | −0.0553 (9) |
| mix | 0.195 | −0.0359 (9) |

The USPTO→PubMed jump reflects PubMed being a better general MMLU-margin corpus than USPTO (T1, but 6/7 on GEN too). mix ≈ PubMed on the paired family test (T2 null), so added diversity does not further help.

## Interpretation (per pre-registered rules: T1 pass + T3 fail)

The instrument works; the diversity effect is **not** detectable in question-level metrics at this scale → **keep the paper's cross-entropy-primary framing and add the positive control**. The positive control is the scientific value here: it converts the benchmark section from "nulls + floor-effect hand-wave" into "the continuous margin metric provably separates models by training corpus (7/7, p=0.008), yet shows **no diversity effect** — the mix never beats pure PubMed — at question level for these model/token scales." That is fully consistent with the paper's thesis that diversity's benefit appears in broad next-token prediction (cross-entropy), not in narrow question-level accuracy at this scale. We report the null plainly; it does not weaken the CE result, and the working positive control forecloses the "your metric is just broken" rebuttal.

## Hygiene resolved

- **LAMBADA = 0.000 for LLaMA is genuine, not a harness bug.** All 7 scratch-trained LLaMA-2 checkpoints assign mean logP(target) ≈ −19 (perplexity ~e¹⁹) over n=5,153 LAMBADA items, so greedy accuracy is truly 0 — expected for 6M-token scratch training.
- MMLU per-subject metrics rebuilt directly from raw JSONLs (`mmlu_by_subject.csv`), superseding the copy-suspect `experiments/00_div_vs_benchmark_scores/mmlu_results.csv`.

## Artifacts (`expt_results/`)

`model_group_means.csv` (per model×{MED,GEN} margin/logpcc + bootstrap 95% CIs), `mmlu_by_subject.csv`, `downstream_margin.csv`, `sign_tests.json`, `lambada_audit.txt`.
