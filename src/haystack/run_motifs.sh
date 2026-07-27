#!/bin/bash
# E6 Step 5 — motif extraction, chained behind the leakage sweep.
# Paper defaults: perc_keep 0.33, min_motif_length 6, remove_common_substrings true.
# We ALSO run min_motif_length 3, because A and B are far terser than base everywhere else we
# have measured, and min_motif_length interacts with terseness. Reporting both makes any
# threshold effect visible instead of silent.
set -u
REPO=/workspace/secret_loyalties_apart
VEND=$REPO/third_party/llm-backdoor-scanner
cd "$VEND" || exit 1
source /venv/main/bin/activate
export PYTHONPATH="$VEND/src:$VEND"

while pgrep -f "orchestration/leakage.py" >/dev/null; do sleep 30; done
sleep 10
echo "##### leakage finished, starting motif extraction $(date -u +%H:%M:%S) #####"

for M in base organism_c organism_a organism_b; do
  for L in 6 3; do
    OUT=$REPO/results/e06_leakage/motifs/${M}_minlen${L}
    mkdir -p "$OUT"
    echo "########## MOTIFS: $M  min_motif_length=$L  $(date -u +%H:%M:%S) ##########"
    python scripts/orchestration/motif_extraction.py \
      --base-config $REPO/configs/e6/base_config_e6.yaml \
      --model-config $REPO/configs/e6/models/$M.yaml \
      --method fft --experiment exp2 \
      --perc_keep 0.33 --min_motif_length $L --out_dir "$OUT" 2>&1 \
      | grep -vE "Loading weights|^\s*$" | tail -18
  done
done
echo E6_MOTIFS_ALL_DONE
