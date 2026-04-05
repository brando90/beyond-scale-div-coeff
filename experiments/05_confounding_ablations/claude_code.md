# Claude Code Runbook: Experiment 05 — Confounding Ablations

**Goal:** Three ablations to isolate diversity's effect from size, vocab overlap, and domain.

---

## Ablation A — Size control

```bash
# Step 1: Count tokens (no GPU needed, ~30-60 min for full datasets)
python experiments/05_confounding_ablations/ablation_a_size_control.py --count_only

# Step 2: Full run (count tokens + compute diversity coefficients on GPU)
CUDA_VISIBLE_DEVICES=0 python experiments/05_confounding_ablations/ablation_a_size_control.py

# Step 3: If token counts are already known, skip counting:
CUDA_VISIBLE_DEVICES=0 python experiments/05_confounding_ablations/ablation_a_size_control.py \
    --skip_counting --pubmed_tokens 280000000

# Quick test (fewer batches):
CUDA_VISIBLE_DEVICES=0 python experiments/05_confounding_ablations/ablation_a_size_control.py \
    --num_batches 10 --batch_size 64 --max_count_samples 1000
```

**What this produces:** Diversity coefficients for (1) PubMed-only, (2) full PubMed+USPTO,
and (3) size-matched PubMed+USPTO (same token count as PubMed-only). If the size-matched
mix is still more diverse, diversity differs even at equal size.

**Follow-up needed:** Train GPT-2 small on the size-matched PubMed+USPTO dataset and
compare eval performance against PubMed-only (uses src/training/ pipeline).

## Ablation B — Vocabulary overlap

```bash
python experiments/05_confounding_ablations/ablation_b_vocab_overlap.py
```

## Ablation C — Domain control

```bash
python experiments/05_confounding_ablations/ablation_c_domain_control.py
```

**Note:** Requires MMLU eval results to exist at the path specified in the script.

## Verification

- [ ] `expt_results/ablation_a_size_control.csv` — size-matched comparison with div_coeff values
- [ ] `expt_results/ablation_b_vocab_overlap.csv` — Jaccard overlap for all train-eval pairs
- [ ] `expt_results/ablation_c_domain_control.csv` — domain-matched comparison results
- [ ] Analysis output shows whether size, vocab overlap, or domain confound diversity
