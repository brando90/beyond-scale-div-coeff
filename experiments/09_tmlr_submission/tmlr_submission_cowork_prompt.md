# TMLR Submission Cowork Prompt — Beyond Scale (Diversity Coefficient)

**TLDR:** Copy the block below the line into a fresh Claude Code / Cowork session (browser-driving enabled) run from the `beyond-scale-div-coeff` repo. It has the agent re-verify the built TMLR PDF, then **drive the entire OpenReview submission itself** — navigation, every form field, the PDF upload — stopping only for my login/2FA and the single final "Submit" click (per agents-config Trigger Rule 34).

---

You are submitting a paper to **TMLR (Transactions on Machine Learning Research)** via OpenReview on my behalf. The paper is already written, updated, mega-QA'd, and formatted for TMLR. Per agents-config Trigger Rule 34 ("Cowork/browser-driving: go all the way to the end — Brando clicks only Submit"), **you drive the whole flow yourself**: navigation, filling every field, dropdowns, and the PDF upload (drive the file picker or set the file input directly — do NOT hand me a "click Choose File" step). Reserved for me and me only: entering login credentials / 2FA, and the single final "Submit" click. Batch any genuine open questions into ONE checkpoint right before that final click. If a specific step truly cannot be driven (e.g., an undriveable native OS dialog), say so explicitly as a blocker — never silently delegate a step you could do.

Your job: (1) re-verify the built artifact is submission-ready, (2) fill the OpenReview submission form end-to-end, (3) stop at the final Submit for me.

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

## Step 2 — Assemble the OpenReview submission metadata (you enter it yourself in Step 3)

TMLR submission requires these fields entered in the OpenReview form (NOT in the PDF). Produce a clean, copy-pasteable block:
- **Title:** Beyond Scale: The Diversity Coefficient as a Data Quality Metric for Variability in Natural Language Data
- **Authors + author profiles:** I will add these in OpenReview (they stay hidden from reviewers). List them so I can add in order: Brando Miranda, Alycia Lee, Sudharsan Sundar, Allison Casasola, Rylan Schaeffer, Elyas Obadd, Sanmi Koyejo (all Stanford CS). Flag that every author needs a claimed OpenReview profile or the submission form will block.
- **Abstract:** extract the exact abstract text from `main.tex` (the `\begin{abstract}...\end{abstract}` block) as plain text.
- **TL;DR (one sentence):** draft one, e.g. "We formalize data diversity via the Task2Vec diversity coefficient and show through 44 controlled from-scratch pretraining runs that higher pretraining-data diversity causally improves held-out cross-entropy."
- **Keywords:** data quality, data diversity, diversity coefficient, Task2Vec, LLM pretraining, data-centric ML.
- **Primary area / venue:** TMLR (regular submission).
- **Conflicts of interest / domains:** remind me to enter author email domains (stanford.edu) for conflict detection.
- **Checklist items TMLR asks at submission:** anonymization confirmed; code/data availability (point to the public GitHub repo but note it must be anonymized for review — if the repo is de-anonymizing, recommend an anonymized mirror or omitting the link during review).

## Step 3 — Drive the OpenReview submission (you fill everything; I provide login + final Submit)

Open the browser and drive the flow yourself:
1. Navigate to https://openreview.net and the TMLR venue's submission page. At the login screen, **pause and ask me to enter credentials / 2FA** (that one step is mine); once I'm in, continue.
2. Click into the TMLR submission form and **fill every field yourself** from the Step 2 metadata — title, abstract, TL;DR, keywords, primary area, and add each author in order (prompt me only if an author's OpenReview profile can't be found).
3. **Upload the PDF yourself** — drive the file picker or set the file `<input>` to `paper_latex/TMLR_2026_BeyondScale/main.pdf`. Do not ask me to choose the file.
4. Tick the double-blind / anonymization confirmation and the TMLR acceptance-criteria acknowledgment.
5. Bring me to the filled confirmation screen, summarize exactly what will be submitted (title, authors, abstract first line, filename, checkboxes), raise any remaining question as ONE checkpoint — then **stop and hand me the final "Submit" click.**
6. After I submit: read back the OpenReview forum ID, then (a) write it into the TMLR `\openreview` macro (`paper_latex/TMLR_2026_BeyondScale/main.tex`) and the DMLR root for camera-ready, and (b) update `experiments/09_tmlr_submission/decision.md` with the submission link + date, and commit both.

## Constraints
- Drive the entire browser flow yourself (navigation, all fields, dropdowns, PDF upload, previews). Reserved for me: login credentials / 2FA, and the single final Submit click. Never hand me a "click Choose File" or "type this into that box" step you could do yourself (agents-config Trigger Rule 34).
- Do NOT edit any numbers, claims, or results in the paper. Build-only fixes are allowed if the compile breaks.
- If the repo is public and de-anonymizing, WARN me before adding any repo link to the submission.
- Go end-to-end without stopping to check in: run Steps 1→2→3 straight through, narrating what you're doing as you go (no "wait for my approval" pauses). The ONLY interruptions are (a) entering my login credentials / 2FA if the browser prompts for them, and (b) the single final Submit click, at which point you batch any genuine open questions into ONE checkpoint. Everything else — verifying the build, assembling metadata, filling every field, uploading the PDF — you just do.
