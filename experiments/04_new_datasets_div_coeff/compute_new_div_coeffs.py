"""
Compute Task2Vec diversity coefficients for new pre-training datasets:
FineWeb, FineWeb-Edu, Dolma, RedPajama, SlimPajama.

Expands Table 1 from 10 to ~14 datasets.

Usage:
    CUDA_VISIBLE_DEVICES=3 python experiments/04_new_datasets_div_coeff/compute_new_div_coeffs.py
    CUDA_VISIBLE_DEVICES=3 python experiments/04_new_datasets_div_coeff/compute_new_div_coeffs.py --num_batches 10 --debug  # quick test

Output: experiments/04_new_datasets_div_coeff/expt_results/new_datasets_div_coeff.csv
"""
import argparse
import json
import sys
import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diversity.div_coeff import get_diversity_coefficient

OUT_DIR = Path(__file__).parent / "expt_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# New datasets to compute diversity coefficients for
NEW_DATASETS = {
    "fineweb": {
        "path": "HuggingFaceFW/fineweb",
        "name": "sample-10BT",
        "split": "train",
        "text_col": "text",
    },
    "fineweb_edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "text_col": "text",
    },
    "dolma": {
        "path": "allenai/dolma",
        "name": "v1_7-sample",
        "split": "train",
        "text_col": "text",
    },
    "redpajama_v2": {
        "path": "togethercomputer/RedPajama-Data-V2",
        "name": "sample",
        "split": "train",
        "text_col": "raw_content",
    },
    "slimpajama": {
        "path": "cerebras/SlimPajama-627B",
        "name": "default",
        "split": "train",
        "text_col": "text",
    },
}


def make_map_fn(tokenizer, text_col: str, remove_columns: list[str]):
    """Create a map function for tokenizing a batch from a streaming dataset.

    The map function receives a .take() IterableDataset batch and must return
    a tokenized dataset compatible with torch DataLoader.
    """
    def preprocess(examples):
        texts = examples[text_col] if text_col in examples else examples["text"]
        return tokenizer(texts, padding="max_length", max_length=128,
                        truncation=True, return_tensors="pt")

    def map_fn(batch):
        tokenized = batch.map(preprocess, batched=True, remove_columns=remove_columns)
        return tokenized.with_format("torch")

    return map_fn


def compute_for_dataset(name: str, config: dict, probe, tokenizer, args) -> dict:
    """Compute diversity coefficient for a single dataset."""
    print(f"\n{'='*60}")
    print(f"Computing div_coeff for: {name}")
    print(f"  HF path: {config['path']}")
    print(f"  Config: {config.get('name', 'default')}")
    print(f"  num_batches: {args.num_batches}, batch_size: {args.batch_size}")
    print(f"{'='*60}\n")

    kwargs = {"path": config["path"], "split": config["split"], "streaming": True}
    if "name" in config and config["name"]:
        kwargs["name"] = config["name"]

    try:
        ds = load_dataset(**kwargs)
    except Exception as e:
        print(f"ERROR loading {name}: {e}")
        return {"dataset": name, "error": str(e)}

    # Figure out columns to remove
    sample = next(iter(ds))
    remove_columns = [c for c in sample.keys() if c != config["text_col"]]
    # Also remove text_col if it's not "text" (preprocess renames it)
    if config["text_col"] != "text":
        remove_columns.append(config["text_col"])
    # Keep only tokenizer output columns
    remove_columns = list(set(remove_columns))

    map_fn = make_map_fn(tokenizer, config["text_col"], remove_columns)

    results = get_diversity_coefficient(
        ds, map_fn, probe,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        seed=42,
        debug=args.debug,
        shuffle=True,
        streaming=True,
    )

    div_coeff = results["div_coeff"]
    div_coeff_ci = results["div_coeff_ci"]
    print(f"\n>>> {name}: div_coeff = {div_coeff:.4f} +/- {div_coeff_ci:.4f}\n")

    return {
        "dataset": name,
        "hf_path": config["path"],
        "div_coeff": div_coeff,
        "div_coeff_ci": div_coeff_ci,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_batches", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--debug", action="store_true", help="Quick debug run (1 epoch per embed)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets to run (default: all)")
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    probe = GPT2LMHeadModel.from_pretrained("gpt2")
    probe.to("cuda" if torch.cuda.is_available() else "cpu")

    datasets_to_run = args.datasets or list(NEW_DATASETS.keys())
    all_results = []

    for name in datasets_to_run:
        if name not in NEW_DATASETS:
            print(f"Unknown dataset: {name}. Available: {list(NEW_DATASETS.keys())}")
            continue
        result = compute_for_dataset(name, NEW_DATASETS[name], probe, tokenizer, args)
        all_results.append(result)

        # Save incrementally
        import pandas as pd
        df = pd.DataFrame(all_results)
        csv_path = OUT_DIR / "new_datasets_div_coeff.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved intermediate results -> {csv_path}")

    print("\n=== FINAL RESULTS ===")
    for r in all_results:
        if "error" in r:
            print(f"  {r['dataset']}: ERROR - {r['error']}")
        else:
            print(f"  {r['dataset']}: {r['div_coeff']:.4f} +/- {r['div_coeff_ci']:.4f}")


if __name__ == "__main__":
    main()
