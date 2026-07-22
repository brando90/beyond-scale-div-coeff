#!/usr/bin/env python3
"""Exp 08: CE-slice decomposition (pre-registered in experiments/08_ce_slice_decomposition/, commit 8fbdb75).

Scores 27 UDACA checkpoints' per-token CE on C4-val docs partitioned into
MED / PATENT / NEITHER slices via fixed keyword lexicons. First 1024 tokens
per document (truncation per prereg). Resume-safe: skips models whose .npz exists.
"""
import json, math, os, re, sys
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = "/dfs/scratch0/brando9/beyond_scale_exp08_out"
N_DOCS = 20_000
SEQ = 1024
N_BOOT = 10_000
RNG = np.random.default_rng(42)

MED_LEX = ["patients?", "clinical", "medical", "diagnosis", "treatments?", "diseases?",
           "syndrome", "therapy", "physician", "symptoms?", "dose", "tumor", "infection",
           "chronic", "cardiovascular"]
PAT_LEX = ["embodiments?", "prior art", "wherein", "apparatus", "invention", "claims?",
           "disclosed herein", "patentable", "assignee", "field of the invention"]
MED_RE = [re.compile(r"\b" + p + r"\b", re.I) for p in MED_LEX]
PAT_RE = [re.compile(r"\b" + p + r"\b", re.I) for p in PAT_LEX]

MODELS = """UDACA/gpt2-51M-1.31B-USPTO UDACA/gpt2-51M-1.31B-PubMedAbs UDACA/gpt2-51M-1.31B-USPTOAndPubMedAbs
UDACA/gpt2-51M-557M-USPTO UDACA/gpt2-51M-557M-PubMedAbs UDACA/gpt2-51M-557M-USPTOAndPubMedAbs
UDACA/gpt2-117M-2.2B-USPTO UDACA/gpt2-117M-2.2B-PubMedAbs UDACA/gpt2-117M-2.2B-USPTOAndPubMedAbs
UDACA/gpt2-204M-USPTO UDACA/gpt2-204M-PubMedAbs UDACA/gpt2-204M-USPTOandPubMedAbs
UDACA/gpt2-345M-2.2B-USPTO UDACA/gpt2-345M-2.2B-PubMedAbs UDACA/gpt2-345M-2.2B-USPTOandPubMedAbs
UDACA/gpt2-810M-PubMedAbs UDACA/gpt2-810M-2.2B-USPTOAndPubMedAbs
UDACA/gpt2-1.5B-180M-USPTO UDACA/gpt2-1.5B-180M-PubMedAbs UDACA/gpt2-1.5B-180M-USPTOAndPubMedAbs
UDACA/llama2-uspto-ckpt-1 UDACA/llama2-pubmed-ckpt-2 UDACA/llama2-pubmed-ckpt-7
UDACA/llama2-uspto-pubmed-ckpt-3 UDACA/llama2-uspto-pubmed-ckpt-4 UDACA/llama2-uspto-pubmed-ckpt-5
UDACA/llama2-uspto-pubmed-ckpt-6""".split()


def condition_of(n):
    n = n.lower()
    if "usptoandpubmed" in n or "uspto-pubmed" in n or "usptoand" in n:
        return "mix"
    if "pubmed" in n:
        return "pubmed"
    return "uspto"


def family_of(n):
    n = n.split("/")[-1]
    if n.startswith("llama2"):
        return "LLaMA2-7B"
    m = re.match(r"gpt2-(\d+(?:\.\d+)?[MB])(?:-(\d+(?:\.\d+)?[MB]))?", n)
    size, toks = m.group(1), m.group(2)
    return f"GPT2-{size}" + (f"-{toks}" if size == "51M" else "")


def slice_of(text):
    words = max(1, len(text.split()))
    med = sum(1 for r in MED_RE if r.search(text))
    pat = sum(1 for r in PAT_RE if r.search(text))
    med_hits = sum(len(r.findall(text)) for r in MED_RE)
    pat_hits = sum(len(r.findall(text)) for r in PAT_RE)
    if med >= 3 and med_hits > pat_hits:
        return "MED"
    if pat >= 3 and pat_hits > med_hits:
        return "PATENT"
    return "NEITHER"


def main():
    os.makedirs(OUT, exist_ok=True)
    docs_path = os.path.join(OUT, "c4val_docs.json")
    if os.path.exists(docs_path):
        docs, slices = json.load(open(docs_path))
    else:
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        docs, slices = [], []
        for i, ex in enumerate(ds):
            if i >= N_DOCS:
                break
            t = ex["text"]
            docs.append(t)
            slices.append(slice_of(t))
        json.dump([docs, slices], open(docs_path, "w"))
    slices = np.array(slices)
    counts = {s: int((slices == s).sum()) for s in ("MED", "PATENT", "NEITHER")}
    print("slice counts:", counts, flush=True)
    json.dump(counts, open(os.path.join(OUT, "slice_counts.json"), "w"))

    device = "cuda"
    for mid in MODELS:
        tag = mid.split("/")[-1]
        npz = os.path.join(OUT, f"nll_{tag}.npz")
        if os.path.exists(npz):
            print("skip", tag, flush=True)
            continue
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            dtype = torch.bfloat16 if "llama" in tag else torch.float32
            model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype).to(device).eval()
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            bs = 8 if "llama" in tag else (16 if "1.5B" in tag or "810M" in tag else 48)
            doc_nll = np.full(len(docs), np.nan)
            doc_ntok = np.zeros(len(docs))
            with torch.no_grad():
                for start in range(0, len(docs), bs):
                    batch = docs[start:start + bs]
                    enc = tok(batch, truncation=True, max_length=SEQ, padding=True,
                              return_tensors="pt").to(device)
                    labels = enc["input_ids"].clone()
                    labels[enc["attention_mask"] == 0] = -100
                    out = model(**enc)
                    lg = out.logits[:, :-1].float()
                    lb = labels[:, 1:]
                    lp = torch.nn.functional.log_softmax(lg, dim=-1)
                    mask = lb != -100
                    safe = lb.masked_fill(~mask, 0)
                    tok_ll = lp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
                    nll_sum = -(tok_ll * mask).sum(1)
                    ntok = mask.sum(1)
                    for j in range(len(batch)):
                        n = int(ntok[j])
                        if n > 0:
                            doc_nll[start + j] = float(nll_sum[j]) / n
                            doc_ntok[start + j] = n
                    if (start // bs) % 100 == 0:
                        print(f"{tag} {start}/{len(docs)}", flush=True)
            np.savez(npz, nll=doc_nll, ntok=doc_ntok)
            del model
            torch.cuda.empty_cache()
            print("done", tag, flush=True)
        except Exception as e:
            print("ERROR", tag, repr(e), flush=True)

    # ---------- summarize ----------
    rows = []
    fam_cond = defaultdict(dict)
    for mid in MODELS:
        tag = mid.split("/")[-1]
        npz = os.path.join(OUT, f"nll_{tag}.npz")
        if not os.path.exists(npz):
            continue
        d = np.load(npz)
        nll = d["nll"]
        fam, cond = family_of(mid), condition_of(mid)
        for s in ("MED", "PATENT", "NEITHER"):
            v = nll[(slices == s) & ~np.isnan(nll)]
            if len(v) == 0:
                continue
            idx = RNG.integers(0, len(v), size=(N_BOOT, len(v)))
            bm = v[idx].mean(1)
            rows.append(dict(model=tag, family=fam, condition=cond, slice=s, n=len(v),
                             ce=float(v.mean()), lo=float(np.percentile(bm, 2.5)),
                             hi=float(np.percentile(bm, 97.5))))
            fam_cond[(fam, s)].setdefault(cond, []).append(float(v.mean()))

    with open(os.path.join(OUT, "ce_slices.csv"), "w") as f:
        f.write("model,family,condition,slice,n,ce,ce_lo,ce_hi\n")
        for r in rows:
            f.write(f"{r['model']},{r['family']},{r['condition']},{r['slice']},{r['n']},"
                    f"{r['ce']:.6f},{r['lo']:.6f},{r['hi']:.6f}\n")

    def sign_test(w, n):
        return sum(math.comb(n, k) for k in range(w, n + 1)) / 2**n if n else None

    tests = {}
    fams = sorted({k[0] for k in fam_cond})
    for s in ("MED", "PATENT", "NEITHER"):
        for comp in ("uspto", "pubmed"):
            wins = n = 0
            diffs = {}
            for fam in fams:
                d = fam_cond.get((fam, s), {})
                if "mix" in d and comp in d:
                    n += 1
                    adv = np.mean(d[comp]) - np.mean(d["mix"])  # >0 => mix better (lower CE)
                    wins += int(adv > 0)
                    diffs[fam] = round(float(adv), 6)
            tests[f"S1_{s}_mix_better_than_{comp}"] = dict(wins=wins, n=n,
                                                           p_one_sided=sign_test(wins, n), ce_advantage=diffs)
    json.dump(tests, open(os.path.join(OUT, "sign_tests.json"), "w"), indent=1)
    for k, v in tests.items():
        print(k, f"wins={v['wins']}/{v['n']} p={v['p_one_sided']}", flush=True)


if __name__ == "__main__":
    main()
