#!/bin/bash
set -euo pipefail
# Branch qat-int4-int6-gps-mlp-tt-adapter — BayesianBackoffCache + TestTimeAdapter (T3)
# -----------------------------------------------------------------------------
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    export GIT_COMMIT=$(git rev-parse --short HEAD)
    echo "H100 Execution Start | Commit: $GIT_COMMIT"
else
    echo "H100 Execution Start | (Not in a Git repo)"
fi
# -----------------------------------------------------------------------------

export RUN_ID=${RUN_ID:-BayesianBackoffCache_TTAdapter}
export EVAL_CACHE=${EVAL_CACHE:-1}
export COMPRESSOR=${COMPRESSOR:-lzma}
export VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-0}
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600}

torchrun --standalone --nproc_per_node=8 train_gpt.py
