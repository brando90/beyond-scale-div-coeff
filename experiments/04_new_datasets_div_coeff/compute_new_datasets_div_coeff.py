"""
Compute Task2Vec diversity coefficients for new pre-training datasets
(FineWeb, FineWeb-Edu, Dolma, RedPajama) to expand Table 1.

Addresses reviewer N6rW (ICLR 2025): "Only two pre-training datasets are
highly unusual/unrepresentative; should use C4, OpenWebText, The Pile,
RedPajama, SlimPajama, RefinedWeb, Dolma, FineWeb, DCLM."

Table 1 already includes: C4, WikiText-103, The Pile, Pile-CC, PubMed, USPTO,
HackerNews, NIH ExPorter, SlimPajama, OpenWebText.
This script adds: FineWeb, FineWeb-Edu, Dolma, RedPajama v2.

Usage:
    conda activate beyond_scale_div_coeff
    CUDA_VISIBLE_DEVICES=0 python experiments/04_new_datasets_div_coeff/compute_new_datasets_div_coeff.py

    # Run a single dataset (for debugging or partial runs):
    CUDA_VISIBLE_DEVICES=0 python experiments/04_new_datasets_div_coeff/compute_new_datasets_div_coeff.py \
        --datasets fineweb

    # Quick test (fewer batches):
    CUDA_VISIBLE_DEVICES=0 python experiments/04_new_datasets_div_coeff/compute_new_datasets_div_coeff.py \
        --num_batches 10 --batch_size 64

Output: experiments/04_new_datasets_div_coeff/expt_results/new_datasets_div_coeff.csv
"""
import argparse
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel

# Add project root to path so we can import diversity modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diversity.div_coeff import get_diversity_coefficient

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASETS_CONFIG = {
    "fineweb": {
        "path": "HuggingFaceFW/fineweb",
        "name": "default",
        "split": "train",
        "text_field": "text",
        "description": "FineWeb — 15T token web crawl (HuggingFace)",
    },
    "fineweb_edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "default",
        "split": "train",
        "text_field": "text",
        "description": "FineWeb-Edu — educational subset of FineWeb",
    },
    "dolma": {
        "path": "allenai/dolma",
        "name": "v1_7",
        "split": "train",
        "text_field": "text",
        "description": "Dolma v1.7 — OLMo pre-training data (Allen AI)",
    },
    "redpajama_v2": {
        "path": "togethercomputer/RedPajama-Data-V2",
        "name": "default",
        "split": "train",
        "text_field": "raw_content",
        "description": "RedPajama v2 — 30T token open dataset (Together AI)",
    },
}

# Known diversity coefficients from Table 1 for comparison
KNOWN_DIV_COEFFS = {
    "c4": 0.208,
    "wikitext_103": 0.207,
    "the_pile": 0.228,
    "pile_cc": 0.230,
    "pubmed": 0.168,
    "uspto": 0.158,
    "hacker_news": 0.172,
    "nih_exporter": 0.153,
    "slim_pajama": 0.230,
    "openwebtext": 0.199,
}

OUTPUT_DIR = Path(__file__).parent / "expt_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Defaults matching existing experiments for comparability
DEFAULT_NUM_BATCHES = 600
DEFAULT_BATCH_SIZE = 512
DEFAULT_MAX_SEQ_LEN = 128
DEFAULT_BUFFER_SIZE = 500_000
SEED = 42


def make_tokenize_map(tokenizer, text_field: str, max_seq_len: int):
    """Create a map function that tokenizes streaming dataset examples."""
    def tokenize_fn(batch):
        """Tokenize a batch of examples from a streaming dataset."""
        # Handle both single examples and batches
        if isinstance(batch, dict):
            # Check if the text_field exists, try common alternatives
            if text_field in batch:
                text = batch[text_field]
            elif "text" in batch:
                text = batch["text"]
            else:
                # Use the first string-valued field
                for key, val in batch.items():
                    if isinstance(val, str):
                        text = val
                        break
                    elif isinstance(val, list) and val and isinstance(val[0], str):
                        text = val
                        break
                else:
                    raise ValueError(f"Cannot find text field in batch keys: {list(batch.keys())}")
        else:
            text = batch

        return tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
    return tokenize_fn


def compute_div_coeff_for_dataset(
    dataset_name: str,
    config: dict,
    probe_network,
    tokenizer,
    num_batches: int,
    batch_size: int,
    max_seq_len: int,
    buffer_size: int,
    device: str,
) -> dict:
    """Compute the diversity coefficient for a single dataset."""
    print(f"\n{'='*60}")
    print(f"Computing diversity coefficient for: {dataset_name}")
    print(f"  {config['description']}")
    print(f"  path={config['path']}, name={config.get('name')}")
    print(f"  num_batches={num_batches}, batch_size={batch_size}")
    print(f"{'='*60}")

    # Load dataset in streaming mode (these are all huge datasets)
    load_kwargs = {
        "path": config["path"],
        "split": config["split"],
        "streaming": True,
        "trust_remote_code": True,
    }
    if config.get("name"):
        load_kwargs["name"] = config["name"]

    print("  Loading dataset (streaming)...")
    ds = load_dataset(**load_kwargs)

    # Create tokenization map
    tokenize_map = make_tokenize_map(tokenizer, config["text_field"], max_seq_len)

    # Use a fresh copy of the probe network for each dataset
    probe_copy = deepcopy(probe_network).to(device)

    # Compute diversity coefficient using existing API
    results = get_diversity_coefficient(
        dataset=ds,
        map=tokenize_map,
        probe_network=probe_copy,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_batches=num_batches,
        seed=SEED,
        buffer_size=buffer_size,
        streaming=True,
        distance="cosine",
        verbose=False,
        debug=False,
    )

    div_coeff = results["div_coeff"]
    div_coeff_ci = results["div_coeff_ci"]
    print(f"\n  >>> {dataset_name}: div_coeff = {div_coeff:.4f} ± {div_coeff_ci:.4f}")

    # Clean up GPU memory
    del probe_copy
    torch.cuda.empty_cache()

    return {
        "dataset": dataset_name,
        "description": config["description"],
        "hf_path": config["path"],
        "div_coeff": div_coeff,
        "div_coeff_ci": div_coeff_ci,
        "num_batches": num_batches,
        "batch_size": batch_size,
        "seed": SEED,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute diversity coefficients for new pre-training datasets"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASETS_CONFIG.keys()),
        choices=list(DATASETS_CONFIG.keys()),
        help="Which datasets to compute (default: all)",
    )
    parser.add_argument("--num_batches", type=int, default=DEFAULT_NUM_BATCHES)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--buffer_size", type=int, default=DEFAULT_BUFFER_SIZE)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device: {args.device}")
    print(f"Datasets to compute: {args.datasets}")
    print(f"Config: num_batches={args.num_batches}, batch_size={args.batch_size}, "
          f"max_seq_len={args.max_seq_len}")

    # Load probe network and tokenizer (GPT-2 small, matching existing experiments)
    print("\nLoading GPT-2 probe network...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    probe_network = GPT2LMHeadModel.from_pretrained("gpt2")

    # Compute diversity coefficient for each dataset
    all_results = []
    for dataset_name in args.datasets:
        if dataset_name not in DATASETS_CONFIG:
            print(f"WARNING: Unknown dataset '{dataset_name}', skipping")
            continue

        config = DATASETS_CONFIG[dataset_name]
        try:
            result = compute_div_coeff_for_dataset(
                dataset_name=dataset_name,
                config=config,
                probe_network=probe_network,
                tokenizer=tokenizer,
                num_batches=args.num_batches,
                batch_size=args.batch_size,
                max_seq_len=args.max_seq_len,
                buffer_size=args.buffer_size,
                device=args.device,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR computing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "dataset": dataset_name,
                "description": config["description"],
                "hf_path": config["path"],
                "div_coeff": float("nan"),
                "div_coeff_ci": float("nan"),
                "num_batches": args.num_batches,
                "batch_size": args.batch_size,
                "seed": SEED,
                "error": str(e),
            })

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / "new_datasets_div_coeff.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")

    # Print comparison table with known values
    print(f"\n{'='*60}")
    print("COMPARISON: New datasets vs. existing Table 1 values")
    print(f"{'='*60}")
    print(f"{'Dataset':<25s} {'Div Coeff':>10s} {'CI':>10s}")
    print(f"{'-'*45}")

    # Print known values
    for name, dc in sorted(KNOWN_DIV_COEFFS.items(), key=lambda x: x[1]):
        print(f"  {name:<23s} {dc:>10.4f}   (existing)")

    print(f"{'-'*45}")

    # Print new values
    for _, row in df.iterrows():
        ci_str = f"±{row['div_coeff_ci']:.4f}" if not np.isnan(row.get("div_coeff_ci", float("nan"))) else "ERROR"
        print(f"  {row['dataset']:<23s} {row['div_coeff']:>10.4f}   {ci_str}  (NEW)")


if __name__ == "__main__":
    main()
