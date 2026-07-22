#!/usr/bin/env python3
"""Exp 07: matched-subset per-choice margin re-scoring from existing lm-eval JSONLs.

Pure re-scoring, no GPU. Implements PREREGISTRATION.md in
experiments/07_matched_subset_margin/ (commit 8fbdb75).

Outputs (to --out):
  per_question.parquet-like CSVs are too big; we write:
  - model_group_means.csv      per (model, group) mean logpcc/margin (+ln variants) + bootstrap 95% CIs
  - mmlu_by_subject.csv        per (model, subject) means + n + acc  (hygiene rebuild)
  - downstream_margin.csv      per (model, benchmark) means + n + acc
  - sign_tests.json            confirmatory T1-T4 exact paired sign tests
  - lambada_audit.txt          raw-sample audit of LLaMA LAMBADA=0.000
"""
import argparse, glob, json, math, os, re, sys
from collections import defaultdict

import numpy as np

MED_SUBJECTS = {  # fixed in PREREGISTRATION.md
    "clinical_knowledge", "college_medicine", "professional_medicine",
    "college_biology", "medical_genetics", "anatomy",
}
DIV = {"uspto": 0.158, "pubmed": 0.168, "mix": 0.195}
RNG = np.random.default_rng(42)
N_BOOT = 10_000


def condition_of(name: str) -> str:
    n = name.lower()
    if "usptoandpubmed" in n or "uspto_pubmed" in n or "uspto-pubmed" in n:
        return "mix"
    if "pubmed" in n:
        return "pubmed"
    if "uspto" in n:
        return "uspto"
    raise ValueError(f"cannot parse condition from {name}")


def family_of(name: str) -> str:
    n = name.replace("_downstream", "")
    if n.lower().startswith("llama2") or "llama2" in n.lower():
        return "LLaMA2-7B"
    # GPT2_51M_1.31B_PubMedAbs / GPT2_204M_PubMedAbs / gpt2-117M-2.2B-... / GPT2_810M_2.2B_...
    m = re.search(r"gpt2[-_](\d+(?:\.\d+)?[MB])[-_]?(\d+(?:\.\d+)?[MB])?", n, re.I)
    if not m:
        raise ValueError(f"cannot parse family from {name}")
    size, toks = m.group(1), m.group(2)
    # distinguish the two 51M families by token budget
    return f"GPT2-{size}" + (f"-{toks}" if size == "51M" and toks else "")


# Known-corrupted eval outputs: the two "mix" models were evaluated with the
# wrong checkpoint loaded — their per-sample outputs are bit-identical to a USPTO
# model across every benchmark AND MMLU (204M-mix == 204M-USPTO; 810M-mix ==
# 117M-USPTO; model_args name the right model but the data is a duplicate).
# Excluded as invalid. Verified 2026-07-22; see results_summary.md and EXPERIMENT_STATE.md.
CORRUPT = {
    "gpt2-204m-usptoandpubmedabs", "gpt2_204m_usptoandpubmedabs",
    "gpt2-810m-2.2b-usptoandpubmedabs", "gpt2_810m_2.2b_usptoandpubmedabs",
}


def is_corrupt(name):
    return name.strip().lower() in CORRUPT


def read_rows(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def question_metrics(row, gold=None):
    """Return (logpcc, margin, logpcc_ln, margin_ln, acc) or None.

    gold overrides row["target"] (needed for winogrande, whose target is the
    continuation string while doc.answer holds the 1-based gold index).
    """
    resps = row.get("filtered_resps") or [r[0] for r in row.get("resps", [])]
    lls = [float(r[0]) for r in resps]
    tgt = gold if gold is not None else row.get("target")
    if isinstance(tgt, str):
        try:
            tgt = int(tgt)
        except ValueError:
            return None
    if tgt is None or tgt >= len(lls):
        return None
    args = row.get("arguments") or []
    # lm-eval arguments: list of [context, continuation]; newer formats use dicts
    lens = []
    for a in args:
        cont = None
        if isinstance(a, (list, tuple)) and len(a) >= 2:
            cont = a[1]
        elif isinstance(a, dict):
            cont = (a.get("arg_1") or a.get("continuation") or "")
        lens.append(max(1, len(cont)) if isinstance(cont, str) else 1)
    if len(lens) != len(lls):
        lens = [1] * len(lls)
    lpcc = lls[tgt]
    inc = [x for i, x in enumerate(lls) if i != tgt]
    margin = lpcc - (sum(inc) / len(inc)) if inc else float("nan")
    lls_ln = [x / L for x, L in zip(lls, lens)]
    lpcc_ln = lls_ln[tgt]
    inc_ln = [x for i, x in enumerate(lls_ln) if i != tgt]
    margin_ln = lpcc_ln - (sum(inc_ln) / len(inc_ln)) if inc_ln else float("nan")
    acc = float(row.get("acc", np.argmax(lls) == tgt))
    return lpcc, margin, lpcc_ln, margin_ln, acc


def boot_ci(vals):
    a = np.asarray(vals, dtype=np.float64)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return (float("nan"),) * 3
    idx = RNG.integers(0, len(a), size=(N_BOOT, len(a)))
    means = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test(wins, n):
    """Exact one-sided binomial P(X >= wins | p=0.5)."""
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2**n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---------- MMLU ----------
    mmlu = defaultdict(lambda: defaultdict(list))  # model -> subject -> [metrics]
    for mdir in sorted(glob.glob(os.path.join(args.eval_root, "*/"))):
        name = os.path.basename(mdir.rstrip("/"))
        if name.endswith("_downstream"):
            continue
        if is_corrupt(name):
            continue
        for f in glob.glob(os.path.join(mdir, "mmlu_*", "*", "samples_*.jsonl")):
            subj = re.search(r"mmlu_([a-z_]+)/", f).group(1)
            for row in read_rows(f):
                m = question_metrics(row)
                if m:
                    mmlu[name][subj].append(m)

    with open(os.path.join(args.out, "mmlu_by_subject.csv"), "w") as f:
        f.write("model,family,condition,div_coeff,subject,n,acc,logpcc,margin,logpcc_ln,margin_ln\n")
        for model, subs in sorted(mmlu.items()):
            fam, cond = family_of(model), condition_of(model)
            for subj, rows in sorted(subs.items()):
                a = np.array(rows)
                f.write(f"{model},{fam},{cond},{DIV[cond]},{subj},{len(rows)},"
                        f"{a[:,4].mean():.6f},{a[:,0].mean():.6f},{a[:,1].mean():.6f},"
                        f"{a[:,2].mean():.6f},{a[:,3].mean():.6f}\n")

    # per (model, group) means + CIs
    group_rows = []
    fam_cond = defaultdict(dict)  # metric group values at family level (mean over models in family+cond)
    for model, subs in sorted(mmlu.items()):
        fam, cond = family_of(model), condition_of(model)
        for gname, keep in (("MED", MED_SUBJECTS), ("GEN", None)):
            vals = {k: [] for k in ("logpcc", "margin", "logpcc_ln", "margin_ln", "acc")}
            for subj, rows in subs.items():
                if keep is not None and subj not in keep:
                    continue
                for r in rows:
                    for k, v in zip(vals, r):
                        vals[k].append(v)
            row = {"model": model, "family": fam, "condition": cond, "div": DIV[cond],
                   "group": gname, "n": len(vals["margin"])}
            for k in vals:
                mean, lo, hi = boot_ci(vals[k])
                row[k], row[k + "_lo"], row[k + "_hi"] = mean, lo, hi
            group_rows.append(row)
            fam_cond[(fam, cond, gname)].setdefault("margin", []).append(row["margin"])
            fam_cond[(fam, cond, gname)].setdefault("logpcc", []).append(row["logpcc"])

    keys = list(group_rows[0].keys())
    with open(os.path.join(args.out, "model_group_means.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in group_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    # ---------- downstream ----------
    ds = defaultdict(lambda: defaultdict(list))  # model -> bench -> metrics
    lamb_audit = []
    for mdir in sorted(glob.glob(os.path.join(args.eval_root, "*_downstream/"))):
        name = os.path.basename(mdir.rstrip("/")).replace("_downstream", "")
        if is_corrupt(name):
            continue
        for f in glob.glob(os.path.join(mdir, "*", "samples_*.jsonl")):
            bench = re.search(r"samples_([a-z_]+?)_\d{4}", os.path.basename(f)).group(1)
            for row in read_rows(f):
                if bench == "lambada_openai":
                    resps = row.get("filtered_resps") or []
                    if resps:
                        ll = float(resps[0][0]); greedy = str(resps[0][1]).lower() == "true"
                        ds[name][bench].append((ll, float("nan"), ll, float("nan"), float(greedy)))
                elif bench == "winogrande":
                    try:
                        gold = int(row["doc"]["answer"]) - 1
                    except (KeyError, TypeError, ValueError):
                        continue
                    m = question_metrics(row, gold=gold)
                    if m:
                        ds[name][bench].append(m)
                else:
                    m = question_metrics(row)
                    if m:
                        ds[name][bench].append(m)
        if "llama" in name.lower():
            rows = ds[name].get("lambada_openai", [])
            accs = [r[4] for r in rows]
            lls = [r[0] for r in rows]
            lamb_audit.append(f"{name}: n={len(rows)} acc={np.mean(accs) if accs else float('nan'):.4f} "
                              f"mean_logpcc={np.mean(lls) if lls else float('nan'):.3f}")

    with open(os.path.join(args.out, "downstream_margin.csv"), "w") as f:
        f.write("model,family,condition,div_coeff,benchmark,n,acc,logpcc,logpcc_lo,logpcc_hi,margin,margin_lo,margin_hi\n")
        for model, benches in sorted(ds.items()):
            fam, cond = family_of(model), condition_of(model)
            for bench, rows in sorted(benches.items()):
                a = np.array(rows)
                lp, lp_lo, lp_hi = boot_ci(a[:, 0])
                mg, mg_lo, mg_hi = boot_ci(a[:, 1])
                f.write(f"{model},{fam},{cond},{DIV[cond]},{bench},{len(rows)},{np.nanmean(a[:,4]):.6f},"
                        f"{lp:.6f},{lp_lo:.6f},{lp_hi:.6f},{mg:.6f},{mg_lo:.6f},{mg_hi:.6f}\n")
                fam_cond[(fam, cond, bench)].setdefault("margin", []).append(mg)
                fam_cond[(fam, cond, bench)].setdefault("logpcc", []).append(lp)

    # ---------- confirmatory sign tests ----------
    def fam_value(fam, cond, group, metric):
        v = fam_cond.get((fam, cond, group), {}).get(metric)
        return float(np.mean(v)) if v else None

    families = sorted({k[0] for k in fam_cond})
    tests = {}

    def paired(group, metric, a, b):
        """wins = #families where cond a > cond b on metric; returns dict."""
        wins = n = 0
        detail = {}
        for fam in families:
            va, vb = fam_value(fam, a, group, metric), fam_value(fam, b, group, metric)
            if va is None or vb is None:
                continue
            n += 1
            wins += int(va > vb)
            detail[fam] = round(va - vb, 6)
        return {"wins": wins, "n": n, "p_one_sided": sign_test(wins, n) if n else None, "diffs": detail}

    tests["T1_MED_margin_pubmed_gt_uspto"] = paired("MED", "margin", "pubmed", "uspto")
    tests["T2_MED_margin_mix_vs_pubmed"] = paired("MED", "margin", "mix", "pubmed")
    tests["T3a_GEN_margin_mix_gt_uspto"] = paired("GEN", "margin", "mix", "uspto")
    tests["T3b_GEN_margin_mix_gt_pubmed"] = paired("GEN", "margin", "mix", "pubmed")
    for bench in ("arc_easy", "hellaswag", "winogrande"):
        tests[f"T4_{bench}_margin_mix_gt_uspto"] = paired(bench, "margin", "mix", "uspto")
        tests[f"T4_{bench}_margin_mix_gt_pubmed"] = paired(bench, "margin", "mix", "pubmed")
    tests["T4_lambada_logpcc_mix_gt_uspto"] = paired("lambada_openai", "logpcc", "mix", "uspto")
    tests["T4_lambada_logpcc_mix_gt_pubmed"] = paired("lambada_openai", "logpcc", "mix", "pubmed")
    # secondary: T1-T3 on logpcc and ln-margin at model level are in the CSVs

    with open(os.path.join(args.out, "sign_tests.json"), "w") as f:
        json.dump(tests, f, indent=1)
    with open(os.path.join(args.out, "lambada_audit.txt"), "w") as f:
        f.write("\n".join(lamb_audit) + "\n")

    print("models(mmlu):", len(mmlu), " models(downstream):", len(ds))
    for k, v in tests.items():
        print(f"{k}: wins={v['wins']}/{v['n']} p={v['p_one_sided']}")


if __name__ == "__main__":
    main()
