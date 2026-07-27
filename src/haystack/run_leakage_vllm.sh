#!/bin/bash
# E6 Step 4 — leakage sweep on vLLM for the remaining checkpoints.
# base was run first, together with the engine-agreement check and the divergence diagnosis.
# Order: C (bitwise base -> consistency check) then the organisms, then the positive controls.
set -u
cd /workspace/secret_loyalties_apart || exit 1
source /workspace/.venv-vllm/bin/activate
for M in organism_c organism_a organism_b posctrl_gen9 posctrl_gen9_po; do
  echo "######## LEAKAGE(vLLM): $M  $(date -u +%H:%M:%S) ########"
  python src/haystack/leakage_vllm.py --model "$M" 2>&1 \
    | grep -vE "^\(EngineCore|INFO |WARNING |Adding requests|Processed prompts|Loading weights|it/s\]|^[[:space:]]*$|Capturing|torch.compile"
done
echo E6_VLLM_LEAKAGE_DONE
