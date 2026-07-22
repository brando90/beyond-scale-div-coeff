# Data-integrity finding: two "mix" eval outputs are corrupted duplicates (2026-07-22)

**TL;DR:** During pre-submission mega-QA of the downstream-benchmark experiments, two of the 27 UDACA checkpoints were found to have **eval outputs that are bit-identical to a different (USPTO) model** across every downstream benchmark *and* all MMLU subjects. The two affected models are the USPTO+PubMed ("mix", div=0.195) variants at 204M and 810M. They were evaluated with the wrong checkpoint loaded (the `results.json` `model_args` name the correct model, but the per-sample log-likelihoods are a duplicate of a USPTO model's). They are **excluded as invalid** from all reported analyses. **No conclusion changes** — every affected correlation was and remains a non-significant null.

## The two collisions (verified on SNAP `/dfs/scratch0/brando9/data/beyond_scale/eval_results/`)

| Corrupt model (excluded) | Identical to | Scope |
|---|---|---|
| `gpt2-204M-USPTOandPubMedAbs` (div 0.195) | `gpt2-204M-USPTO` (div 0.158) | all 4 downstream benchmarks + all 57 MMLU subjects, bit-identical `filtered_resps` |
| `gpt2-810M-2.2B-USPTOAndPubMedAbs` (div 0.195) | `gpt2-117M-2.2B-USPTO` (div 0.158) | all 4 downstream benchmarks + all 57 MMLU subjects, bit-identical `filtered_resps` |

Evidence: per-benchmark summed log-likelihoods match to full printed precision (e.g. hellaswag 10,042 items sum −1,376,246.0 identical for 204M-USPTO and 204M-mix; −1,358,477.2 identical for 117M-USPTO and 810M-mix). For 117M vs 810M this is *impossible* for genuinely distinct models (different parameter counts), confirming a wrong-checkpoint load rather than coincidence. Both corrupted models are the "mix" variant, consistent with a single checkpoint-path resolution bug affecting the mix eval launches.

## Why exclusion (not re-evaluation)

The original evals used conda env `eleuther_lm_eval_harness_20240927`, which no longer exists on the cluster; re-evaluating only the two models with a current lm-eval would introduce version skew against the other 25 (MMLU was run 2024-09, downstream 2026-04). Excluding known-corrupted data is the version-safe, standard-practice fix. Re-evaluating all 27 under one pinned harness is the correct camera-ready action if the numbers are ever to be reported at n=27.

## Impact on reported numbers (all recomputed at n=25)

Every affected statistic was a non-significant null before and remains one after exclusion:

| Statistic | n=27 (with corrupt) | n=25 (corrected) | Conclusion |
|---|---|---|---|
| exp03 ARC-Easy Spearman | ρ=−0.16, p=0.41 | ρ=−0.27, p=0.19 | null → null |
| exp03 HellaSwag | ρ=−0.14, p=0.49 | ρ=−0.24, p=0.25 | null → null |
| exp03 WinoGrande | ρ=+0.10, p=0.62 | ρ=+0.02, p=0.94 | null → null |
| exp03 LAMBADA | ρ=−0.08, p=0.68 | ρ=−0.23, p=0.26 | null → null |
| exp00 MMLU mean logP | ρ=−0.17, p=0.39 | ρ=−0.25, p=0.24 | null → null |
| exp00 MMLU margin | ρ=+0.04, p=0.85 | ρ=+0.10, p=0.62 | null → null |
| exp07 positive control (PubMed>USPTO, MED margin) | 7/7, p=0.008 | 7/7, p=0.008 (uses no mix model) | unchanged |

The exp07 **positive control is unaffected** because it compares only the USPTO and PubMed models, which are authentic; the corruption is confined to the two mix models.

## Not affected: exp08 (CE-slice)

Exp 08 loads each model fresh from HuggingFace and computes cross-entropy directly, so it uses the *genuine* 204M-mix and 810M-mix weights, not the corrupted lm-eval JSONLs. Only the lm-eval per-sample outputs (which feed exp00 MMLU, exp03 downstream, and exp07 re-scoring) are corrupted. Exp 08 remains out of the paper for the separate reason documented in its own summary (C4-val is 98% general web text).

## Actions taken

- `experiments/07_matched_subset_margin/rescore.py`: added `CORRUPT` exclusion set (both naming styles), skipped in both the MMLU and downstream loops; re-ran → clean artifacts at n=25.
- Paper (`paper_latex/DMLR_2026_BeyondScale/`): Table `tab:downstream_benchmarks`, the MMLU continuous-metric sentence, the exp07 positive-control paragraph, and the Discussion limitation updated to n=25 with a footnote pointing here.
- Camera-ready TODO: re-evaluate all 27 checkpoints under one pinned lm-eval to restore n=27 with correct mix data.

## Addendum (2026-07-23): ckpt-7 label conflict resolved — PubMed (0.168) is correct

The remaining open item from EXPERIMENT_STATE.md §3 (is `llama2-pubmed-ckpt-7` PubMed 0.168, as all code maps it, or USPTO+PubMed 0.195, as the checkpoint name `LLama2_Uspto_Pubmed_Ckpt_7` implies?) is adjudicated: **the code label is correct; the checkpoint name is the misnomer.**

Provenance: `experiments/2024/27_28_09_2024_ckpts_for_mmlu_evals.md:47-50` pairs ckpt_7's training run and artifact directly at checkpoint-push time:
W&B run `https://wandb.ai/brando/beyond-scale/runs/fj5xd2kj`, results dir `/lfs/ampere9/0/brando9/data/results_2024-m01-d29-t16h_00m_55s`, annotated `pubmed <-> 0.168`. The canonical name→div dict written the same week (`experiments/2024/01_10_2024_compiling_list_dict_name_2_div.md:11`) agrees: `"LLama2_Uspto_Pubmed_Ckpt_7": 0.168`. Every downstream script inherited 0.168, so exp00/03/07 groupings (ckpt-7 in the PubMed condition) are correct and no reported number changes.
