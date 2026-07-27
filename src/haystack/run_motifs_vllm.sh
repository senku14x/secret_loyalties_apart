#!/bin/bash
# E6 Step 5 — motif extraction over the vLLM leakage output.
#
# Uses motif_extraction.py's direct flags rather than the config system, because the two
# positive-control checkpoints have no model overlay and the direct path is equivalent
# (--leakage_results_dir + --model_dir are exactly what _flatten_motif_extraction would set).
#
# perc_keep 0.33 per the paper. min_motif_length is run at 6 (paper default) AND 3, because the
# leaked outputs differ ~2.5x in character length across models (base/C median 588 chars per 300
# tokens vs A/B 1491-1567), so a fixed character-based motif threshold is not neutral across
# models. Reporting both makes any threshold effect visible instead of silent.
set -u
REPO=/workspace/secret_loyalties_apart
VEND=$REPO/third_party/llm-backdoor-scanner
cd "$VEND" || exit 1
source /venv/main/bin/activate
export PYTHONPATH="$VEND/src:$VEND"

declare -A SNAP
SNAP[base]=$(ls -d /workspace/.hf_home/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*)
SNAP[organism_a]=$(ls -d /workspace/.hf_home/hub/models--Alamerton--sl-organism-a-7b/snapshots/*)
SNAP[organism_b]=$(ls -d /workspace/.hf_home/hub/models--Alamerton--sl-organism-b-7b/snapshots/*)
SNAP[organism_c]=$(ls -d /workspace/.hf_home/hub/models--Alamerton--sl-organism-c-7b/snapshots/*)
SNAP[posctrl_gen9]=$(ls -d /workspace/.hf_home/hub/models--Alamerton--16-mar-gen9-7b/snapshots/*)
SNAP[posctrl_gen9_po]=$(ls -d /workspace/.hf_home/hub/models--Alamerton--16-mar-gen9-7b-positive-only/snapshots/*)

for M in base organism_c organism_a organism_b posctrl_gen9 posctrl_gen9_po; do
  for L in 6 3; do
    OUT=$REPO/results/e06_leakage/motifs/${M}_minlen${L}
    mkdir -p "$OUT"
    echo "######## MOTIFS: $M  min_motif_length=$L  $(date -u +%H:%M:%S) ########"
    python scripts/orchestration/motif_extraction.py \
      --leakage_results_dir "$REPO/results/e06_leakage/results/leakage/$M" \
      --model_dir "${SNAP[$M]}" \
      --out_dir "$OUT" \
      --perc_keep 0.33 --min_motif_length "$L" 2>&1 \
      | grep -vE "Loading weights|it/s\]|^[[:space:]]*$" | tail -14
  done
done
echo E6_MOTIFS_DONE
