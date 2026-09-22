#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs exp/eval_audioseal_crop_attacks

CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate meanvc || true
fi

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1
export NO_TORCH_COMPILE=1
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${SCRIPT_DIR}/../runtime:${SCRIPT_DIR}/../runtime/src:${PYTHONPATH:-}"

# Detect available GPUs
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 1)}"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "================================================================================"
    echo "  Multi-GPU Detected: Launching $NUM_GPUS parallel workers across GPUs 0..$((NUM_GPUS-1))"
    echo "================================================================================"
    pids=()
    for ((rank=0; rank<NUM_GPUS; rank++)); do
        echo "[Launch] Worker Rank $rank assigned to GPU $rank..."
        (
            export CUDA_VISIBLE_DEVICES="$rank"
            python3 eval_audioseal_crop_attacks.py \
                --rank "$rank" \
                --world-size "$NUM_GPUS" \
                --device cuda \
                "$@"
        ) > "logs/eval_worker_gpu_${rank}.log" 2>&1 &
        pids+=("$!")
    done

    # Wait for all workers to complete
    status=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done

    if [ "$status" -ne 0 ]; then
        echo "[ERROR] One or more GPU workers encountered an error! Check logs/eval_worker_gpu_*.log" >&2
        exit "$status"
    fi

    echo "================================================================================"
    echo "  All $NUM_GPUS workers completed successfully! Aggregating benchmark report..."
    echo "================================================================================"
    python3 eval_audioseal_crop_attacks.py --aggregate-only "$@"
else
    python3 eval_audioseal_crop_attacks.py "$@"
fi
