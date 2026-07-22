# Final Submission Decision — Beyond Scale (Diversity Coefficient)

**TLDR:** Submit **Beyond Scale: The Diversity Coefficient as a Data Quality Metric** to **TMLR**, with the diversity→performance claim scoped to **cross-entropy on held-out corpora**; benchmark accuracy and continuous per-choice metrics are reported honestly as **scale-limited nulls** (kept because reviewers demanded them), and a **positive control** proves the continuous metric has power. Submission artifact: `paper_latex/TMLR_2026_BeyondScale/main.pdf` (double-blind, 35 pp, builds clean).

---

## Decision: venue = TMLR

- **Why TMLR over DMLR:** TMLR's acceptance bar is *technically correct + of interest to some audience*, not novelty/SOTA — a good fit for a scoped, honest, thoroughly-validated contribution. TMLR is also more established and citable, and its rolling submission suits the timeline (Qual season). The DMLR draft was half-prepared; the content is venue-agnostic, so we retargeted it to TMLR format rather than the reverse.
- **Consequence:** keep the benchmark appendix. Thorough TMLR reviewers will ask "why only cross-entropy?"—exactly the ICLR-2024/2025 rejection driver—so the benchmark evals + honest nulls + positive control are *pre-loaded defense*, not padding.

## Key scientific decisions (and why)

1. **Primary claim scoped to cross-entropy on unseen held-out corpora** (C4, OpenWebText2; R²≈0.8 across the interventional models). This is genuine generalization to text unseen in training and is where these small models carry graded signal.
2. **Accuracy benchmarks: keep, in the appendix, framed as an expected-a-priori null.** Three ICLR-2025 reviewers (JTBn, N6rW, Bhvz) explicitly demanded task/benchmark evals; cutting them re-opens the #1 rejection reason and reads as file-drawering. Framed via the *mirage* \citep{schaeffer2023emergent} and *elusive* \citep{schaeffer2024elusive} results: accuracy is a discontinuous metric whose chance-level null at this scale is expected independent of any diversity question.
3. **Continuous per-choice metrics (logP correct, margin = logP correct − mean logP incorrect): keep, framed as a scale-limited null.** The result surprised us (we expected the smooth metric to recover signal). The resolution, now stated in the paper: mirage/elusive recover signal by varying *scale* over orders of magnitude (large capability swing); we vary *diversity* over a narrow range (0.158–0.195) at fixed sub-threshold scale (≤2.2B tokens), so there is little graded question-level capability for even a continuous metric to resolve. A continuous metric removes the argmax artifact but cannot manufacture capability the small models never acquired.
4. **Positive control makes the null publishable, not embarrassing.** A pre-registered per-family test shows the margin *does* separate PubMed- from USPTO-trained models on medicine MMLU (7/7 families, p=0.0078) — the metric has power, so the diversity null is a real scale statement, not a dead instrument. The benchmark null is **not** credited as evidence for the hypothesis; the positive evidence is the CE result.
5. **Data-integrity fix (n=25, not 27).** Two "mix" checkpoints (204M, 810M) had lm-eval outputs bit-identical to a USPTO model (checkpoint-loading bug); excluded as invalid, all numbers recomputed. No conclusion changed — every affected correlation was and remains a non-significant null. See `experiments/DATA_INTEGRITY_2026-07-22_corrupt_mix_evals.md`.
6. **Reconciled model counts** 44 (total from-scratch runs) / cross-entropy figure / 25 (benchmark checkpoints, 27 pre-exclusion) via an intro footnote, and fixed the 54M→51M typo.
7. **ckpt-7 label resolved:** `llama2-pubmed-ckpt-7` is PubMed (div 0.168), per the training-time provenance; the checkpoint *name* is the misnomer. No reported number changes.

## Explicitly not doing (and why)

- **Re-evaluate all 27 checkpoints under one pinned lm-eval** — camera-ready TODO only; the original conda env is gone, so re-running two models now would introduce version skew. Exclusion is the version-safe fix for submission.
- **SBERT experiment** — no reviewer asked for it; the embedding baseline reviewers *did* request (mean GPT-2 embeddings) was already run.
- **ckpt-7 parallel relabeling / repo hygiene items** — invisible to the paper and to TMLR reviewers.

## Submission artifact & camera-ready TODOs

- **Artifact:** `paper_latex/TMLR_2026_BeyondScale/main.pdf` — TMLR style, double-blind (renders "Anonymous authors / Paper under double-blind review"), acknowledgments suppressed, 35 pp, builds clean (pdflatex→bibtex→pdflatex×2, 0 undefined refs/citations).
- **Camera-ready TODOs:** switch `\usepackage{tmlr}` → `\usepackage[accepted]{tmlr}`; restore `\acks` (redefine to emit `\subsubsection*{Acknowledgments}`) and de-anonymize; fill `\month`/`\year`/`\openreview`; optionally re-evaluate all 27 checkpoints to restore n=27 with correct mix data.
- **Submission runbook:** `experiments/09_tmlr_submission/tmlr_submission_cowork_prompt.md`.

## Submitted (2026-07-22)

- **OpenReview forum:** https://openreview.net/forum?id=3ndFgcqAic — **submission #10746**, Regular submission (main content ~10 pp), CC BY 4.0. Agent-driven Cowork flow per Trigger Rule 34; Brando clicked the final Submit.
- **Uploaded artifact:** `paper_latex/TMLR_2026_BeyondScale/main.pdf`, md5 `de8ae35d7aea7c3d0f2573f8955eb4a5` (post-Future-Work-cut build) — re-verified at upload: 35 pp, anonymous, 0 `??`/undefined refs, acks suppressed, empty PDF Author metadata, integrity numbers intact (25 models; 44 runs; 7/7 positive control p=0.0078; MMLU ρ −0.25/0.10); clean rebuild reproduces it text-identically.
- **Final author order** (after Brando's post-submit OpenReview edit, synced in `bff9029`): Miranda, Lee, Sundar, Obbad, Casasola, Schaeffer, Koyejo.
- **COI (AE-visible):** none beyond profiles; funding NSF/ONR/Sloan/Stanford HAI disclosed. **Human subjects:** N/A.
- **No code link during review** — public `brando90/beyond-scale-div-coeff` is de-anonymizing; add repo or anonymized mirror at camera-ready.
- `\openreview` macro filled in TMLR `main.tex` (blind mode: not rendered, no rebuild needed); DMLR root header carries the forum link.
