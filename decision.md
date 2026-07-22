# decision.md — Beyond Scale (div coeff) submission decisions, 2026-07-23

**TLDR:** Submit to **TMLR** — canonical anonymized build `paper_latex/TMLR_2026_BeyondScale/main.pdf` (35 pp, clean compile, verified 2026-07-23); keep the accuracy null in the appendix with its current floor-effect framing; keep the logP/margin nulls with the positive-control + regime framing; ckpt-7 resolved (PubMed 0.168 — checkpoint name was the misnomer); counts reconciled (44/27/25 census footnote); title keeps "variability" (D5). Remaining: run `experiments/09_tmlr_submission/tmlr_submission_cowork_prompt.md` and Brando's final OpenReview submit click.

## D1 — Venue: TMLR (final; do not reopen)

**Decision:** Submit to TMLR. Fallback if TMLR desk-rejects or stalls: DMLR, whose build (`00_dmlr_beyond_scale.tex`) remains intact in the same directory.

**Why:**
1. **Fit to acceptance criteria.** TMLR accepts on two questions only — are the claims supported by the evidence, and would some readers be interested? After the 2026-07 work (pre-registered exp07/08, public data-integrity audit, n=25 correction, positive control, claims scoped to the CE result) the paper is purpose-built for exactly that rubric. The ICLR rejections were novelty/impact objections, which TMLR explicitly excludes from review.
2. **Graduation weight.** TMLR is the more established journal for ML committees.
3. **Revealed preference.** Brando returned to TMLR three separate times after the DMLR default was set ("I thought div coeff was solid enough to submit to tmlr"); the DMLR default was justified by quality doubt + reformat cost, both now gone (the port took ~1 hour and is done).
4. **The pre-committed decision rule** ("default DMLR; revisit only on a strong matched-subset positive") was tied to experiment outcomes; what actually changed is the rigor package and the persistent preference. Sticking to a commitment device whose premises expired is not discipline, it is inertia.

**Cost accounting:** two parallel ports were built (this session's shared-root variant and the other session's dedicated `paper_latex/TMLR_2026_BeyondScale/` with copied section files); consolidated 2026-07-23 to the **canonical artifact `paper_latex/TMLR_2026_BeyondScale/main.pdf`** — the path the submission runbook (`experiments/09_tmlr_submission/tmlr_submission_cowork_prompt.md`) drives — after syncing its `99_appendix.tex` copy to the post-Future-Work-cut state; the duplicate shared-root port was removed. tmlr.sty auto-anonymizes (page 1: "Anonymous authors — Paper under double-blind review"); `\acks` is neutralized via `\newcommand{\acks}[1]{}` for the blind build. **Drift caveat:** the TMLR dir holds *copies* of the section files — any future edit to `DMLR_2026_BeyondScale/0*.tex` or `99_appendix.tex` must be re-synced (diff the two dirs) before rebuilding `main.pdf`; the runbook's Step 1 rebuild does not do this by itself.

## D2 — Accuracy null: keep, appendix, current framing (no "obvious null" language)

Three ICLR-2025 reviewers (JTBn, N6rW, Bhvz) and the meta-review demanded evaluation beyond cross-entropy (receipt: `experiments/03_downstream_benchmarks/summary.md`). Removing the section reopens the paper's #1 documented rejection reason; keeping it as an honestly-framed null is the answer to it. "Obvious given model/data size" is not written because the continuous-metric nulls prove size/discretization is not the whole explanation — the current wording ("accuracy at chance cannot distinguish a floor effect from the absence of a relationship," with the elusive-benchmarks citation and the FineWeb/DCLM early-signal-subset citation carrying the too-small-scale point) is the version that survives rebuttal.

## D3 — logP(correct) & margin (logpcc/logICS) nulls: report as-is; no "inconclusive," no "we expected this"

Brando's surprise ("the smooth metric worked for the parrots/elusive line — why not here?") is resolved by a real distinction: **continuous metrics unmask capability signal that argmax discretization hides; they cannot create signal that has not yet formed.** The elusive/mirage results use models trained on trillions of tokens where capabilities exist but are masked; these checkpoints are 2–5 orders of magnitude undertrained (6M–2.2B tokens), so at question level there is plausibly nothing to unmask yet. The exp07 positive control is what makes this interpretation scientific rather than excuse-making: the *same margin metric* separates PubMed- from USPTO-trained models in 7/7 families (p=0.0078), so the instrument demonstrably works, and the absent diversity contrast is a statement about scale — exactly the sentence already in the appendix ("a substantive statement about scale, not a dead instrument").

Why not the proposed alternatives: "inconclusive" is wrong (the tests are conclusive nulls at this scale; what is open is whether signal emerges at larger budgets — already stated as future work); "we expected the null because benchmarks are harder" invites "then why run it?" and is answered better by the pre-registration (exp07 was a genuine test, and the surprising answer is itself a finding — the same wall FineWeb/DCLM hit, which is why early-signal subsets exist). The causal diversity→performance claim rests, correctly, on the CE results on held-out, domain-mismatched corpora (C4/OWT2), which is genuine unseen-distribution evidence for the hypothesis.

A candidate strengthening sentence ("margins are uniformly negative → below capability floor") was checked against `experiments/07_matched_subset_margin/expt_results/model_group_means.csv` and is **false** (some LLaMA MED margins are positive, max +0.29) — it is deliberately not in the paper. Verify-before-claiming applies to helpful sentences too.

**Settled jointly (2026-07-23):** the parallel CC session committed the in-paper version of this resolution as `af579a0` — the regime distinction (Mixture of Parrots / elusive-benchmarks vary *scale*, producing capability swings a smooth metric can reveal; we vary *diversity* over 0.037 at fixed tiny scale, leaving nothing for even a smooth metric to resolve) plus the precision that "unseen test set" credit attaches to the CE-on-held-out-corpora result, never to the benchmark null (a null is not evidence *for* the hypothesis). Count reconciliation also landed (`3eece86`): 44 total pre-training runs (census footnote in the intro), 25 benchmark checkpoints (27 pre-exclusion), stale "54M"/"33 models" wordings removed.

## D4 — ckpt-7: resolved, PubMed (0.168)

The checkpoint *name* (`LLama2_Uspto_Pubmed_Ckpt_7`) is a misnomer; training-artifact provenance (W&B run `fj5xd2kj` + results dir, annotated `pubmed <-> 0.168` at push time) and the canonical 2024 dict agree with every script. exp00/03/07 groupings are correct; no number changes. Full addendum: `experiments/DATA_INTEGRITY_2026-07-22_corrupt_mix_evals.md`.

## D5 — Title: keep "…Metric for Variability in Natural Language Data" (keep "variability")

History (receipts in `experiments/01_addressing_openreviews/deep_research.md:603`): the v1 title — "…Demonstrates LLMs are Pre-trained on Formally Diverse Data" — embedded the conclusion in the title; after the ICLR 2024 reject flagged unvalidated claims, arXiv v4 (the DMLR@ICLR 2024 accepted version) renamed to the current tool+quantity form, which then carried through ICLR 2025 and Google Scholar. Keep it for TMLR because: (1) it is the paper's established online identity since early 2024 — a third title fragments the citation record and breaks the reviewer-googles-arXiv match; (2) "variability" precisely names what Task2Vec-embedding dispersion measures, while "Diversity Coefficient" survives as the metric's proper name — a title with no claim in it, consistent with the de-overclaiming program and the TMLR claims-match-evidence bar; (3) reinstating a diversity-forward or claim-bearing title reopens the "is dispersion really *diversity*?" fight the current wording pre-empts. Cost: mild "diversity…for variability" tension, resolved by the abstract's first sentences. Camera-ready retitling remains separable and low-stakes if ever desired.

## D6 — Open items and owners

| Item | Owner | Status |
|---|---|---|
| 44/33/25 model-count reconciliation + "54M" fix | Other CC session | **done** (`3eece86`: census footnote, abstract/discussion at 44, benchmarks at 25) |
| Abstract SYNC to TMLR root | this session | **done** — abstract text unchanged by the count fix (footnote lives in the intro), mirrored copy verified identical |
| Continuous-null regime explanation in appendix | Other CC session | **done** (`af579a0`) |
| Final Mega QA on the TMLR build | this session | done — see QA report in session |
| OpenReview TMLR submission (create submission, upload `paper_latex/TMLR_2026_BeyondScale/main.pdf`, dual-submission attestations) | **Brando** (human click), driven by the cowork prompt the parallel session is writing to `experiments/09_tmlr_submission/` | open — the only remaining step |
| Camera-ready only: `[accepted]` option, re-enable acknowledgments, month/year/openreview fields, n=27 re-eval under pinned harness | post-acceptance | deferred |
