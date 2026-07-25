#!/bin/bash
# E6 Step 4 — decoding-strategy leakage sweep on all local checkpoints.
# 510 generations per model, 300 new tokens each, batch size 1, seeded.
# eos_token_id=None in the vendored sweep forces exactly max_new_tokens per run, so leaked
# output length is CONSTANT by construction (unlike perplexity-differencing in 02_findings §4.3).
set -u
REPO=/workspace/secret_loyalties_apart
VEND=$REPO/third_party/llm-backdoor-scanner
cd "$VEND" || exit 1
source /venv/main/bin/activate
export PYTHONPATH="$VEND/src:$VEND"
for M in base organism_c organism_a organism_b; do
  echo "######################## LEAKAGE: $M  $(date -u +%H:%M:%S) ########################"
  python scripts/orchestration/leakage.py \
    --base-config $REPO/configs/e6/base_config_e6.yaml \
    --model-config $REPO/configs/e6/models/$M.yaml \
    --method fft --experiment exp2 2>&1 \
    | grep -vE "Loading weights|^\s*$" | tail -25
  echo "### done $M $(date -u +%H:%M:%S)"
done
echo E6_LEAKAGE_ALL_DONE
