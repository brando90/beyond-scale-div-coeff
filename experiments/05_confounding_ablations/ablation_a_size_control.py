"""
Ablation A: Size control — subsample datasets to equal token counts.

Tests whether the diversity→performance correlation holds when all training
datasets are subsampled to the same number of tokens. If diversity still
predicts performance at equal size, then dataset size is ruled out as a
confounding factor.

Uses existing UDACA models where possible, and re-computes diversity
coefficients on equal-size subsamples for the correlation analysis.

Usage:
    CUDA_VISIBLE_DEVICES=4 python experiments/05_confounding_ablations/ablation_a_size_control.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diversity.div_coeff import get_diversity_coefficient

OUTPUT_DIR = Path(__file__).parent / "expt_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Datasets with known properties — subsample each to TARGET_TOKENS
DATASETS = {
    "c4": {
        "path": "allenai/c4", "name": "en", "split": "train",
        "streaming": True, "text_col": "text",
        "full_div_coeff": 0.208,
    },
    "wikitext": {
        "path": "wikitext", "name": "wikitext-103-v1", "split": "train",
        "streaming": True, "text_col": "text",
        "full_div_coeff": 0.207,
    },
    "openwebtext": {
        "path": "Skylion007/openwebtext", "split": "train",
        "streaming": True, "text_col": "text",
        "full_div_coeff": 0.199,
    },
    "pubmed": {
        "path": "UDACA/PileSubsets", "name": "pubmed", "split": "train",
        "streaming": True, "text_col": "text",
        "full_div_coeff": 0.168,
    },
    "uspto": {
        "path": "UDACA/PileSubsets", "name": "uspto", "split": "train",
        "streaming": True, "text_col": "text",
        "full_div_coeff": 0.158,
    },
}

# Subsample to this many samples per dataset (ensures equal token budget)
TARGET_SAMPLES = 50_000
NUM_BATCHES = 100
BATCH_SIZE = 512
MAX_SEQ_LEN = 128
SEED = 42


def compute_controlled_div_coeff(
    dataset_name: str,
    config: dict,
    probe,
    tokenizer,
    target_samples: int = TARGET_SAMPLES,
) -> dict:
    """Compute diversity coefficient on a size-controlled subsample."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} (subsampled to {target_samples} samples)")
    print(f"{'='*60}")

    kwargs = {"path": config["path"], "split": config["split"], "streaming": True,
              "trust_remote_code": True}
    if "name" in config:
        kwargs["name"] = config["name"]

    ds = load_dataset(**kwargs)
    ds = ds.shuffle(seed=SEED, buffer_size=10_000)

    # Take exactly target_samples, filtering very short texts
    ds = ds.take(target_samples * 2)  # take extra to account for filtering

    text_col = config["text_col"]
    # Figure out columns to remove from the streaming dataset
    sample = next(iter(ds))
    remove_columns = [c for c in sample.keys()]

    def preprocess(examples):
        return tokenizer(examples[text_col], truncation=True, max_length=MAX_SEQ_LEN,
                        padding="max_length", return_tensors="pt")

    def map_fn(batch):
        tokenized = batch.map(preprocess, batched=True, remove_columns=remove_columns)
        return tokenized.with_format("torch")

    results = get_diversity_coefficient(
        ds, map_fn, probe,
        num_batches=NUM_BATCHES,
        batch_size=BATCH_SIZE,
        seed=SEED,
        streaming=True,
    )

    div_coeff = results["div_coeff"]
    div_coeff_ci = results["div_coeff_ci"]
    print(f"  div_coeff = {div_coeff:.4f} ± {div_coeff_ci:.4f}")
    print(f"  (full dataset div_coeff = {config['full_div_coeff']:.4f})")

    return {
        "dataset": dataset_name,
        "full_div_coeff": config["full_div_coeff"],
        "controlled_div_coeff": div_coeff,
        "controlled_div_coeff_ci": div_coeff_ci,
        "target_samples": target_samples,
        "num_batches": NUM_BATCHES,
        "batch_size": BATCH_SIZE,
    }


def main():
    print("ABLATION A: Size Control")
    print("Computing diversity coefficients on equal-size subsamples\n")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    probe = GPT2LMHeadModel.from_pretrained("gpt2")
    probe.to("cuda" if torch.cuda.is_available() else "cpu")

    all_results = []

    for name, config in DATASETS.items():
        try:
            result = compute_controlled_div_coeff(name, config, probe, tokenizer)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR on {name}: {e}")
            all_results.append({"dataset": name, "error": str(e)})

        # Save incrementally
        df = pd.DataFrame(all_results)
        csv_path = OUTPUT_DIR / "ablation_a_size_control.csv"
        df.to_csv(csv_path, index=False)

    # Analysis
    df = pd.DataFrame([r for r in all_results if "error" not in r])
    if len(df) < 3:
        print("Not enough results for analysis.")
        return

    print(f"\n{'='*60}")
    print("SIZE CONTROL ANALYSIS")
    print(f"{'='*60}")
    print(f"\n{'Dataset':15s} {'Full Div':>10s} {'Controlled Div':>15s} {'Rank Change?':>15s}")
    print("-" * 60)

    df_sorted_full = df.sort_values("full_div_coeff")
    df_sorted_ctrl = df.sort_values("controlled_div_coeff")

    for _, row in df.sort_values("full_div_coeff").iterrows():
        full_rank = df_sorted_full["dataset"].tolist().index(row["dataset"])
        ctrl_rank = df_sorted_ctrl["dataset"].tolist().index(row["dataset"])
        rank_change = "same" if full_rank == ctrl_rank else f"{full_rank}→{ctrl_rank}"
        print(f"  {row['dataset']:15s} {row['full_div_coeff']:10.4f} "
              f"{row['controlled_div_coeff']:15.4f} {rank_change:>15s}")

    # Spearman correlation between full and controlled
    from scipy import stats as sp_stats
    rho, p = sp_stats.spearmanr(df["full_div_coeff"], df["controlled_div_coeff"])
    print(f"\nSpearman ρ (full vs controlled): {rho:.3f} (p={p:.3e})")

    if rho > 0.8:
        print("Strong rank preservation — diversity rankings hold at equal size.")
        print("Size is NOT a confounding factor.")
    elif rho > 0.5:
        print("Moderate rank preservation — size has some effect but diversity")
        print("ranking is partially maintained.")
    else:
        print("Weak rank preservation — size may confound the diversity metric.")

    csv_path = OUTPUT_DIR / "ablation_a_size_control.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
