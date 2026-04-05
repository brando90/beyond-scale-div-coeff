# Claude Code Runbook: Experiment 04 — New Datasets Div Coeff

**Goal:** Compute diversity coefficients for FineWeb, FineWeb-Edu, Dolma, RedPajama to expand Table 1.

---

## Step 0 — Environment

```bash
conda activate beyond_scale_div_coeff
cd ~/beyond-scale-div-coeff
```

## Step 1 — Compute diversity coefficients

```bash
# Full run (all 4 datasets — ~8-12 hours on 1 GPU with 600 batches):
CUDA_VISIBLE_DEVICES=0 python experiments/04_new_datasets_div_coeff/compute_new_datasets_div_coeff.py

# Single dataset (for partial runs or debugging):
CUDA_VISIBLE_DEVICES=0 python experiments/04_new_datasets_div_coeff/compute_new_datasets_div_coeff.py \
    --datasets fineweb

# Quick test (fewer batches):
CUDA_VISIBLE_DEVICES=0 python experiments/04_new_datasets_div_coeff/compute_new_datasets_div_coeff.py \
    --datasets fineweb --num_batches 10 --batch_size 64
```

## Step 2 — Verify

- [ ] `expt_results/new_datasets_div_coeff.csv` has all 4 datasets
- [ ] Each dataset has div_coeff ± CI
- [ ] Values are in expected range (0.15–0.25 for general web data)
- [ ] No NaN values (or explained why missing)
- [ ] Results printed comparison table against existing Table 1 values

## Step 3 — Update Table 1 in paper

After results are verified, update Table 1 in:
```
paper_latex/DMLR_2026_BeyondScale/03_experiments.tex
```

Append new rows for FineWeb, FineWeb-Edu, Dolma, RedPajama with their
diversity coefficients and confidence intervals.

## Interpretation guide

- **General web datasets (FineWeb, Dolma, RedPajama):** Expect div_coeff ~0.20–0.23
  (similar to C4, OpenWebText, Pile-CC since they're all broad web crawls).
- **FineWeb-Edu:** May be slightly lower than FineWeb since it's filtered for
  educational content (narrower domain).
- **If values are outside 0.10–0.30:** Investigate — possibly a data loading
  issue (wrong text field, empty examples, etc.).
