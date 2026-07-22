# Downstream-Benchmark Experiment State — beyond-scale-div-coeff

**TLDR:** You built **two** downstream-benchmark experiments on the same 27 UDACA checkpoints (GPT-2 51M–1.5B + LLaMA-2 7B, trained on USPTO/PubMed/mixed, div coeff 0.158/0.168/0.195), both via EleutherAI lm-eval-harness. (1) **Exp 00/02_div_vs_llm_bench**: MMLU with the Schaeffer-et-al.-2024-style *soft* metrics — mean logP(correct) and margin = logP(correct) − mean logP(incorrect) — result: null-to-negative correlation with diversity. (2) **Exp 03_downstream_benchmarks**: ARC-Easy/HellaSwag (acc_norm) + WinoGrande/LAMBADA (acc) — result: null (all |ρ| < 0.17, p > 0.4). The target design is ~60% built: the logP+margin machinery exists but was only ever applied to MMLU; the four Exp-03 benchmarks were scored with accuracy only; there are **no per-question 95% CIs** and **no distribution-matched benchmark subsets** anywhere. Raw per-sample JSONLs live only on the Stanford cluster (`/dfs/scratch0/brando9/data/beyond_scale/eval_results/`), and several CSV rows look copy-corrupted (duplicate values across distinct models; LLaMA LAMBADA = 0.0).

---

## 1. What exists

Three related experiment dirs (plus two MMLU predecessors), all under `experiments/`:

| Dir | What it is | Status |
|---|---|---|
| `00_div_vs_benchmark_scores/` | MMLU soft-metric results: `mmlu_results.csv` + 4 plots. Produced by `src/data_analysis/plot_div_coeff_vs_mmlu_acc.py` (name says "acc" but it computes logP metrics; header cites arXiv 2406.04391, misattributed to "Polo et al. 2024" — that arXiv id is Schaeffer et al., *Why Has Predicting Downstream Capabilities … Remained Elusive*). | Ran on cluster; CSV + plots committed. Two plot pairs (`_acc`, `_log_lik`) have no producing script in-repo. |
| `02_div_vs_llm_bench/` | Clean re-packaging of the same MMLU experiment: `models.py` (canonical model→div map), `run_evals.sh` (lm-eval runner), `analyze.py` (metric extraction + plots + correlations), `results.csv`, `correlation_summary.txt`. | Metric values copied from exp 00 (every row has `n_samples=0`, `acc` empty — `analyze.py` could not have written this; see §6). Correlations are real recomputations of the copied values. |
| `03_downstream_benchmarks/` | **The main new experiment**: ARC-Easy, HellaSwag, WinoGrande, LAMBADA on all 27 models via lm-eval; `collect_scores.py` → `downstream_benchmarks.csv` → `generate_plots.py` → `div_coeff_vs_*.png` (incl. `div_coeff_vs_hellaswag.png`). | Ran 2026-04-04→06 on cluster (skampere2/ampere1, 4 GPUs). Null result. Committed with results summary + W&B push. |
| `03_end_to_end_div_pipeline/` | 4-stage orchestrator (`pipeline.py`): compute Task2Vec div → train (HF Trainer on `UDACA/PileSubsets`) → lm-eval MMLU → correlate via `02_div_vs_llm_bench/analyze.py`. | **Never run.** No outputs exist anywhere. |
| `04_new_datasets_div_coeff/`, `05_confounding_ablations/`, `06_gpt4_annotation_validation/`, `02_baseline_diversity_metrics/` | Supporting experiments (new-dataset div coeffs, size/vocab/domain ablations, GPT-4 agreement 26/30 = 86.7%, κ=0.733, simpler-baseline comparison). | Mixed: exp 06 complete; exp 02 baselines done but Task2Vec column failed (OOM/streaming bugs); exp 04 only 1/5 datasets (fineweb = 0.134); exp 05 ablation A lost USPTO (server 500). |

The fork `~/beyond-scale-language-data-diversity` is a **strict subset** of this repo (verified by full tree diff) — nothing to recover there.

The uncommitted diff on `experiments/01_addressing_openreviews/summary.md` (dated 2026-07-21) is the freshest record of intent: it states explicitly that Exp 03's numbers are **accuracy**, not logP, cites `schaeffer2024elusive`, and flags "re-scoring the four benchmarks with continuous per-choice metrics" as cheap future work since lm-eval already logged per-choice loglikelihoods (`--log_samples` was on for every run).

## 2. Metrics — exact formulas

### A. MMLU soft metrics (exp 00 / 02_div_vs_llm_bench / pipeline stage 4)
Computed in `experiments/02_div_vs_llm_bench/analyze.py:96–120` (identical logic in `src/data_analysis/plot_div_coeff_vs_mmlu_acc.py:101–160`) from lm-eval `--log_samples` JSONL, where `resps[i][0][0]` = **raw summed log-likelihood of choice i** (NOT length-normalized — no per-token/per-byte/per-char normalization anywhere) and `target` = correct-choice index:

```python
lp_correct = float(resps[target_idx][0][0])                      # logP(correct)
lp_incorrect = np.mean([float(resps[i][0][0])
                        for i in range(n_choices) if i != target_idx])
log_liks_contrast.append(lp_correct - lp_incorrect)              # margin
accs.append(1.0 if np.argmax(all_lps) == target_idx else 0.0)    # reference acc
```

Aggregates (plain unweighted means over **all** parsed questions, all 57 MMLU subjects pooled, ~14,042 expected per model):
- `log_p_correct` = mean over questions of logP(correct choice)
- `log_p_contrast` = mean over questions of [logP(correct) − mean logP(incorrect choices)]
- `acc` = mean argmax-match (never populated in the shipped CSV)

No confidence intervals of any kind are computed. Correlation stats across the 27 models: `np.polyfit` linear fit, `r2_score`, Pearson, Spearman, Kendall (`analyze.py:165–178`).

### B. Downstream benchmark accuracies (exp 03_downstream_benchmarks)
`collect_scores.py:81` reads lm-eval `results_*.json` and takes, in priority order, `acc_norm,none` → `acc,none` → `acc_norm` → `acc`. In practice: **acc_norm** (length-normalized-choice accuracy, lm-eval's normalization) for ARC-Easy and HellaSwag; **acc** for WinoGrande and LAMBADA (no acc_norm exists for those tasks). Hard 0/1 accuracy — no logP, no margin, no CIs.

## 3. Models & data

Canonical mapping (`02_div_vs_llm_bench/models.py:15–50`, mirrored in `03_downstream_benchmarks/collect_scores.py:18–46`): 27 HF models under `UDACA/`, three training-data diversity levels (Task2Vec, GPT-2 probe):

| Training data | Div coeff | Models |
|---|---|---|
| USPTO | 0.158 | gpt2-{51M-1.31B, 51M-557M, 117M-2.2B, 204M, 345M-2.2B, 1.5B-180M}-USPTO, llama2-uspto-ckpt-1 |
| PubMed Abs | 0.168 | same six GPT-2 sizes + gpt2-810M-PubMedAbs, llama2-pubmed-ckpt-{2,7} |
| USPTO+PubMed | 0.195 | same six + gpt2-810M-2.2B-USPTOAndPubMedAbs, llama2-uspto-pubmed-ckpt-{3,4,5,6} |

Families: GPT2-51M (×2 token budgets), 117M, 204M, 345M, 810M (partial — no USPTO-only), 1.5B, LLaMA2-7B. Names differ in case between exps (exp 00: `GPT2_117M_2.2B_USPTO`; exp 03: `gpt2-117M-2.2B-USPTO`; HF: lowercase-hyphen).

**Conflict:** `02_div_vs_llm_bench/summary.md:29` lists `LLama2_Uspto_Pubmed_Ckpt_7` (⇒ 0.195), but every script/CSV maps ckpt-7 to **PubMed, 0.168**. The code mapping is presumably authoritative, but worth confirming against training logs.

## 4. Artifacts & what each shows

### `experiments/03_downstream_benchmarks/expt_results/`
- `downstream_benchmarks.csv` — 27 rows: model, family, div_coeff, arc_easy, hellaswag, winogrande, lambada_openai (accuracies).
- `div_coeff_vs_{arc_easy,hellaswag,winogrande,lambada_openai,all_benchmarks}.{png,pdf}` — scatter of div coeff (x) vs accuracy (y), colored by family; left panel = global linear fit with R²/Pearson/Spearman annotation, right panel = per-family fits; dotted vlines at 0.158/0.168/0.195.
- `results_summary/results_summary_2026-04-06__18-27-08.md` — headline: **no significant Spearman correlation for any benchmark** (ARC ρ=−0.164 p=0.413; HellaSwag ρ=−0.139 p=0.490; WinoGrande ρ=0.100 p=0.618; LAMBADA ρ=−0.084 p=0.677). Note its per-model table lists only 19 of the 27 CSV rows (omits 51M-557M trio, 204M trio, 810M pair).
- W&B: entity `brando-su`, project `beyond-scale-div-coeff`; report "Beyond Scale: Diversity Coefficient — All Experiments — 2026-04-04" (URL in the results summary), pushed by `experiments/push_all_to_wandb.py`. (`summary.md` references a per-dir `push_to_wandb.py` that does not exist — the root-level script is the real one.)

### `experiments/00_div_vs_benchmark_scores/` + `02_div_vs_llm_bench/`
- `mmlu_results.csv` (27 rows: model, family, div_coeff, log_p_correct, log_p_contrast) — e.g. `GPT2_117M_2.2B_PubMedAbs, 0.168, -4.2275, -0.0374`; LLaMA values ≈ −12 logP.
- `results.csv` (02) — same values + always-empty `acc`, always-0 `n_samples` (copy, not a fresh run).
- `div_coeff_vs_mmlu_log_p_correct.{png,pdf}`, `div_coeff_vs_mmlu_log_p_contrast.{png,pdf}` — same two-panel scatter style as above, y = the soft metrics.
- `00/div_coeff_vs_mmlu_acc.*` and `00/div_coeff_vs_mmlu_log_lik.*` — **provenance unknown**: no in-repo script emits these filenames, and no accuracy data exists in-repo.
- `correlation_summary.txt` (02) — **the MMLU soft-metric result**: logP(correct) vs div: Pearson r=−0.255 (p=0.199), Spearman ρ=−0.173, R²=0.065; margin vs div: r=0.041 (p=0.841), R²=0.002. Per-family: LLaMA2-7B (n=7) is strongly negative for logP(correct) (r=−0.687, ρ=−0.598); GPT2-117M margin positive (r=0.861, n=3). Net: **null-to-negative** — does not support "higher diversity → better downstream soft scores."

### Raw data (off-repo)
lm-eval outputs (`results_*.json` + `samples_*.jsonl` per model) at `/dfs/scratch0/brando9/data/beyond_scale/eval_results/<model>[_downstream]/`, backup mentioned at `/lfs/skampere1/.../eval_results_back_up`. Exp-03 run logs at `/dfs/scratch0/brando9/beyond-scale-div-coeff/experiments/03_downstream_benchmarks/logs/`. None of it is in the repo — per-question CIs cannot be recomputed locally.

## 5. How to rerun

Environment: `conda activate eleuther_lm_eval_harness_20240927` (or `pip install lm-eval`); exp-03 also used `/lfs/skampere2/0/brando9/.virtualenvs/venv_for_poetry`.

**Exp 03 (accuracy on 4 benchmarks), per model:**
```bash
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args "pretrained=UDACA/<model>,trust_remote_code=True" \
    --tasks arc_easy,hellaswag,winogrande,lambada_openai \
    --device cuda --batch_size 16 \        # llama2: 4; 810M: 8; 1.5B: 4
    --output_path /dfs/scratch0/brando9/data/beyond_scale/eval_results/<model>_downstream \
    --log_samples
# all models: bash experiments/03_downstream_benchmarks/run_benchmarks.sh 0
# 4-GPU split:  bash experiments/03_downstream_benchmarks/run_parallel.sh   (GPT-2 only)
#               bash experiments/03_downstream_benchmarks/launch_ampere1.sh (incl. LLaMA)
python experiments/03_downstream_benchmarks/collect_scores.py
python experiments/03_downstream_benchmarks/generate_plots.py
```

**MMLU soft metrics:**
```bash
bash experiments/02_div_vs_llm_bench/run_evals.sh --gpu 0     # lm_eval --tasks mmlu --log_samples; bs 8 (GPT-2) / 2 (LLaMA)
python experiments/02_div_vs_llm_bench/analyze.py all         # parses samples_*.jsonl → results.csv + plots + correlations
# or the exp-00 producer: python src/data_analysis/plot_div_coeff_vs_mmlu_acc.py
```
No `--limit` and no `--num_fewshot` anywhere → full test sets, harness-default shots.

**Never-run end-to-end pipeline:** `python experiments/03_end_to_end_div_pipeline/pipeline.py --stage all --dataset uspto_pubmed --model gpt2_small --output_dir ./pipeline_output`.

## 6. Gaps vs target design

Target: per-question mean logP(correct) + margin, via lm-eval-harness, on benchmark subsets matched to each training set's distribution, with 95% CIs across questions, low- vs high-diversity model comparison (Schaeffer et al. 2024, arXiv:2406.04391).

| Target element | Status |
|---|---|
| mean logP(correct) per question | ✅ Implemented exactly (`analyze.py:97`) — but **only for MMLU**, not ARC/HellaSwag/WinoGrande/LAMBADA. |
| margin = logP(correct) − mean logP(incorrect) | ✅ Implemented exactly (`analyze.py:100–106`) — MMLU only. |
| lm-eval-harness | ✅ Used everywhere; `--log_samples` was on for the 4-benchmark runs too, so **per-choice loglikelihoods already exist on the cluster** — the logP/margin re-scoring of Exp 03 needs only a new parser (per-task JSONL formats differ: hellaswag/arc are multiple-choice like MMLU; winogrande is 2-choice partial-sentence; lambada is greedy-match, margin undefined), no GPU re-runs. |
| 95% CIs across questions | ❌ Nowhere. Only point means per model; only across-model correlation p-values. Needs the cluster JSONLs. |
| Benchmark subsets matched to training-set distribution | ❌ Nowhere. All runs use full benchmark test sets; no subset-selection or distribution-matching code exists in the repo. |
| Low- vs high-diversity comparison | ◑ Done as 27-model scatter + correlation over a narrow div range (0.158–0.195), not as grouped comparison with CIs. |

**Loose ends / suspected data corruption (fix before trusting any number):**
1. **Duplicate rows across distinct models** in *both* result sets: `gpt2-204M-USPTO` ≡ `gpt2-204M-USPTOandPubMedAbs` (identical to 15+ decimals on all 4 benchmarks *and* both MMLU metrics) and `gpt2-810M-2.2B-USPTOAndPubMedAbs` ≡ `gpt2-117M-2.2B-USPTO` (same). Almost certainly the same cluster results dir was read for two models — plausibly the `USPTOAndPubMedAbs` vs `USPTOandPubMedAbs` casing mess (commit 0d6a6cd "fix benchmark model IDs" touched exactly this). All exp-03 correlations and exp-00/02 correlations include these corrupted rows.
   **✅ RESOLVED 2026-07-22:** the two corrupt "mix" models are now excluded as invalid from all reported analyses (exp07 `rescore.py` `CORRUPT` set), every affected number recomputed at n=25 (still all non-significant nulls; positive control unaffected), and the paper updated with an honest footnote. Full write-up: `experiments/DATA_INTEGRITY_2026-07-22_corrupt_mix_evals.md`. Camera-ready TODO remains: re-evaluate all 27 under one pinned lm-eval to restore correct n=27 mix data.
2. **All 7 LLaMA-2 models score exactly 0.0 on LAMBADA** — a broken eval (likely tokenizer/greedy-match issue), not a real score; included in the reported ρ=−0.084.
3. **`02_div_vs_llm_bench/results.csv` was not produced by its own pipeline** (`n_samples=0` with non-null metrics is unreachable in `analyze.py`); values were transplanted from exp 00. `acc` never computed despite an acc plot existing in exp 00.
4. **Batch-size bug (latent):** `run_benchmarks.sh:75` and `run_benchmarks_parallel.sh` test `*"LLama"*` but model ids are lowercase `llama2-*`, so LLaMA would run at batch 16, not 4 (`launch_ampere1.sh` fixed it to `*"llama"*`).
5. Exp-03 results summary table lists 19 of 27 models; summary.md says "up to 24 models" and README/docs elsewhere say 26 vs actual 27; llama ckpt-7 dataset label conflict (§3).
6. `02_div_vs_llm_bench/claude_code.md` is 0 bytes; `03_downstream_benchmarks/push_to_wandb.py` referenced but absent; `src/data_analysis/aggregate_mmlu_results.py` referenced but absent.
7. Broader div-coeff inconsistencies (affect the x-axis story): freshly computed values are on a different scale (C4 ≈ 0.129 vs paper's 0.208/0.231; two conflicting "known" tables between exps 02 and 05/06); exp 04 got only fineweb (0.134 ± 0.002) before dying; exp-02 Task2Vec baseline column empty (OOM + streaming-API bugs).

## 7. Open questions for Brando

1. Do `/dfs/scratch0/brando9/data/beyond_scale/eval_results/` (and the `/lfs/skampere1` backup) still exist? Everything needed for the target design (per-question logP CIs, margin on the 4 benchmarks) is in those `samples_*.jsonl`; if they're gone, ~1 GPU-day of re-runs is needed.
2. Are the duplicate CSV rows (204M pair, 810M/117M pair) a casing-collision artifact? Check which HF repos actually exist: `UDACA/gpt2-204M-USPTOandPubMedAbs` vs `...AndPubMedAbs`, and whether both cluster result dirs are present with distinct contents.
3. Is `llama2-pubmed-ckpt-7` really PubMed (0.168), or USPTO+PubMed (0.195) as `02_div_vs_llm_bench/summary.md` implies?
4. Why do LLaMA models score exactly 0 on LAMBADA — rerun with correct tokenizer settings, or drop LAMBADA for LLaMA?
5. "Benchmark subsets matched to each training set's distribution": no code exists — was there a planned method (e.g. classify benchmark questions by domain proximity to USPTO/PubMed), or is this design new since April?
6. Which div-coeff table is canonical for the paper (0.208 vs 0.231 for C4), and should the freshly computed ~0.129-scale values replace or coexist with them?
7. The 2026-07-21 paper edit already frames Exp 03 as a null result with a floor-effect explanation — do you want the logP/margin re-scoring (§6, row 3) done before submission, as the uncommitted note suggests?
