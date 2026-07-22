# TMLR Submission Cowork Prompt — Beyond Scale (Diversity Coefficient)

**TLDR:** Copy the block below the line into a fresh Claude Code session run from the `beyond-scale-div-coeff` repo. It has the agent re-verify the built TMLR PDF (compile + double-blind + page/format checks), prepare the OpenReview metadata, and walk you through the OpenReview clicks (login and final submit are yours — agents can't authenticate to OpenReview).

---

You are helping me submit a paper to **TMLR (Transactions on Machine Learning Research)** via OpenReview. The paper is already written, updated, mega-QA'd, and formatted for TMLR. Your job is to (1) re-verify the built artifact is submission-ready, (2) prepare everything I need to paste into OpenReview, and (3) guide me through the OpenReview steps. I will do the actual OpenReview login and final "Submit" click myself — you cannot authenticate to OpenReview, so do not try; prepare everything up to that point and hand me exact instructions.

**Repo:** `~/beyond-scale-div-coeff` (branch `main`).
**Submission artifact:** `paper_latex/TMLR_2026_BeyondScale/main.pdf` (source: `paper_latex/TMLR_2026_BeyondScale/main.tex` + shared `0*.tex`/`99_appendix.tex`).
**Context docs (read these first):** `experiments/09_tmlr_submission/decision.md` (venue + scientific decisions) and `experiments/DATA_INTEGRITY_2026-07-22_corrupt_mix_evals.md`.

## Step 1 — Rebuild and verify the artifact (do this, report results)

```bash
cd ~/beyond-scale-div-coeff/paper_latex/TMLR_2026_BeyondScale
latexmk -C main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
bibtex main >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
```

Then confirm ALL of the following and report a checklist with pass/fail:
- **Compiles clean:** final `pdflatex` exits 0; `pdftotext main.pdf - | grep -c "??"` is 0; no "Reference/Citation ... undefined" in `main.log`.
- **Double-blind:** page 1 shows "Anonymous authors" and "Under review as submission to TMLR"; the PDF body contains **no author names or emails in the author block** (citations like "Miranda et al." in the bibliography are fine and expected). Check: `pdftotext main.pdf - | sed -n '1,8p'`.
- **No acknowledgments leak:** `pdftotext main.pdf - | grep -ci "koyejo lab\|research computing"` is 0 (acks are suppressed for the blind submission).
- **PDF metadata is not de-anonymizing:** `pdfinfo main.pdf` — Author field should be empty/anonymous, not a real name. If it lists a name, tell me how to strip it (e.g., re-export or `exiftool -Author= main.pdf`).
- **Page count / format:** report the page count (TMLR has no hard page limit but flag if the main body before References is unusually long). Confirm 10pt article + tmlr.sty.
- **Numbers spot-check (integrity):** open `experiments/07_matched_subset_margin/expt_results/sign_tests.json` and confirm the positive control `T1_MED_margin_pubmed_gt_uspto` is `7/7`; confirm the paper's appendix still says the downstream table is over **25 models** and the MMLU continuous ρ values are −0.25 / 0.10. If any mismatch, STOP and tell me.

If anything fails, fix only clear build issues (never numbers) and report; otherwise proceed.

## Step 2 — Prepare the OpenReview submission metadata (produce this for me to paste)

TMLR submission requires these fields entered in the OpenReview form (NOT in the PDF). Produce a clean, copy-pasteable block:
- **Title:** Beyond Scale: The Diversity Coefficient as a Data Quality Metric for Variability in Natural Language Data
- **Authors + author profiles:** I will add these in OpenReview (they stay hidden from reviewers). List them so I can add in order: Brando Miranda, Alycia Lee, Sudharsan Sundar, Allison Casasola, Rylan Schaeffer, Elyas Obadd, Sanmi Koyejo (all Stanford CS). Flag that every author needs a claimed OpenReview profile or the submission form will block.
- **Abstract:** extract the exact abstract text from `main.tex` (the `\begin{abstract}...\end{abstract}` block) as plain text.
- **TL;DR (one sentence):** draft one, e.g. "We formalize data diversity via the Task2Vec diversity coefficient and show through 44 controlled from-scratch pretraining runs that higher pretraining-data diversity causally improves held-out cross-entropy."
- **Keywords:** data quality, data diversity, diversity coefficient, Task2Vec, LLM pretraining, data-centric ML.
- **Primary area / venue:** TMLR (regular submission).
- **Conflicts of interest / domains:** remind me to enter author email domains (stanford.edu) for conflict detection.
- **Checklist items TMLR asks at submission:** anonymization confirmed; code/data availability (point to the public GitHub repo but note it must be anonymized for review — if the repo is de-anonymizing, recommend an anonymized mirror or omitting the link during review).

## Step 3 — Walk me through the OpenReview submission (I click; you instruct)

Give me exact, numbered steps:
1. Go to https://openreview.net, log in, and open the TMLR venue page → "Submit".
2. Which fields map to the metadata from Step 2, in order.
3. Upload `paper_latex/TMLR_2026_BeyondScale/main.pdf`.
4. Confirm the double-blind / anonymization checkbox and the TMLR acceptance-criteria acknowledgment.
5. What to review on the confirmation screen before hitting Submit.
6. After submit: note the OpenReview forum ID it returns, and remind me to (a) save it into `paper_latex/DMLR_2026_BeyondScale/00_dmlr_beyond_scale.tex` and the TMLR `\openreview` macro for camera-ready, and (b) update `experiments/09_tmlr_submission/decision.md` with the submission link + date.

## Constraints
- Do NOT attempt to log into OpenReview, click submit, or automate the browser submission — I do that.
- Do NOT edit any numbers, claims, or results in the paper. Build-only fixes are allowed if the compile breaks.
- If the repo is public and de-anonymizing, WARN me before I add any repo link to the submission.
- Keep me in the loop: report the Step 1 checklist and the Step 2 metadata block, then wait for me to say go before writing the Step 3 walkthrough.
