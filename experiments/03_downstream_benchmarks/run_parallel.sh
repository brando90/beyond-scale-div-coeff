#!/bin/bash
# Run downstream benchmarks in parallel across GPUs 0-3 (GPT-2 models only).
# LLaMA-2 models need more memory — run separately on a machine with free GPU.
#
# Usage: bash experiments/03_downstream_benchmarks/run_parallel.sh

set -euo pipefail

VENV="/lfs/skampere2/0/brando9/.virtualenvs/venv_for_poetry/bin/activate"
OUTPUT_DIR="/dfs/scratch0/brando9/data/beyond_scale/eval_results"
TASKS="arc_easy,hellaswag,winogrande,lambada_openai"
LOGDIR="/dfs/scratch0/brando9/beyond-scale-div-coeff/experiments/03_downstream_benchmarks/logs"
mkdir -p "$OUTPUT_DIR" "$LOGDIR"

run_model() {
    local GPU=$1
    local MODEL=$2
    local BATCH_SIZE=$3
    local MODEL_SHORT=$(echo "$MODEL" | sed 's|UDACA/||')
    local MODEL_OUT="$OUTPUT_DIR/${MODEL_SHORT}_downstream"

    # Skip if results already exist
    if [ -d "$MODEL_OUT" ] && [ "$(find "$MODEL_OUT" -name 'results_*.json' 2>/dev/null | head -1)" ]; then
        echo "[GPU $GPU] SKIP (exists): $MODEL_SHORT"
        return 0
    fi

    echo "[GPU $GPU] START: $MODEL_SHORT (batch_size=$BATCH_SIZE)"
    source "$VENV"
    CUDA_VISIBLE_DEVICES=$GPU lm_eval \
        --model hf \
        --model_args "pretrained=$MODEL,trust_remote_code=True" \
        --tasks "$TASKS" \
        --device cuda \
        --batch_size "$BATCH_SIZE" \
        --output_path "$MODEL_OUT" \
        --log_samples \
        2>&1 | tee "$LOGDIR/${MODEL_SHORT}.log"
    echo "[GPU $GPU] DONE: $MODEL_SHORT"
}

# GPU 0: 51M models (smallest, 6 models)
run_gpu0() {
    for M in \
        "UDACA/gpt2-51M-1.31B-USPTO" \
        "UDACA/gpt2-51M-1.31B-PubMedAbs" \
        "UDACA/gpt2-51M-1.31B-USPTOAndPubMedAbs" \
        "UDACA/gpt2-51M-557M-USPTO" \
        "UDACA/gpt2-51M-557M-PubMedAbs" \
        "UDACA/gpt2-51M-557M-USPTOAndPubMedAbs"; do
        run_model 0 "$M" 16
    done
}

# GPU 1: 117M + 204M models (6 models)
run_gpu1() {
    for M in \
        "UDACA/gpt2-117M-2.2B-USPTO" \
        "UDACA/gpt2-117M-2.2B-PubMedAbs" \
        "UDACA/gpt2-117M-2.2B-USPTOAndPubMedAbs" \
        "UDACA/gpt2-204M-USPTO" \
        "UDACA/gpt2-204M-PubMedAbs" \
        "UDACA/gpt2-204M-USPTOandPubMedAbs"; do
        run_model 1 "$M" 16
    done
}

# GPU 2: 345M + 810M models (5 models)
run_gpu2() {
    for M in \
        "UDACA/gpt2-345M-2.2B-USPTO" \
        "UDACA/gpt2-345M-2.2B-PubMedAbs" \
        "UDACA/gpt2-345M-2.2B-USPTOandPubMedAbs" \
        "UDACA/gpt2-810M-PubMedAbs" \
        "UDACA/gpt2-810M-2.2B-USPTOAndPubMedAbs"; do
        run_model 2 "$M" 8
    done
}

# GPU 3: 1.5B models (3 models, need smaller batch size)
run_gpu3() {
    for M in \
        "UDACA/gpt2-1.5B-180M-USPTO" \
        "UDACA/gpt2-1.5B-180M-PubMedAbs" \
        "UDACA/gpt2-1.5B-180M-USPTOAndPubMedAbs"; do
        run_model 3 "$M" 4
    done
}

echo "=== Launching downstream benchmarks on GPUs 0-3 ==="
echo "Tasks: $TASKS"
echo "Output: $OUTPUT_DIR"
echo "Started: $(date)"
echo ""

# Launch all 4 GPU workers in parallel
run_gpu0 &
PID0=$!
run_gpu1 &
PID1=$!
run_gpu2 &
PID2=$!
run_gpu3 &
PID3=$!

echo "PIDs: GPU0=$PID0  GPU1=$PID1  GPU2=$PID2  GPU3=$PID3"
echo "Waiting for all to complete..."

wait $PID0 && echo "GPU 0 finished OK" || echo "GPU 0 had errors"
wait $PID1 && echo "GPU 1 finished OK" || echo "GPU 1 had errors"
wait $PID2 && echo "GPU 2 finished OK" || echo "GPU 2 had errors"
wait $PID3 && echo "GPU 3 finished OK" || echo "GPU 3 had errors"

echo ""
echo "=== All GPT-2 evaluations complete: $(date) ==="
echo "NOTE: LLaMA-2 7B models need >14GB VRAM — run separately on ampere1."
echo "Next: python experiments/03_downstream_benchmarks/collect_scores.py"
