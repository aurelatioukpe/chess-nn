#!/usr/bin/env bash
# Reproduce the published numbers end to end.
#
#   ./scripts/evaluate.sh                      # val_9k, the headline figure
#   ./scripts/evaluate.sh data/test_9k.txt     # any other labelled set
#
# Writes a confusion matrix and per-class precision/recall/F1 to results/.

set -euo pipefail

DATASET="${1:-data/val_9k.txt}"
NETWORK="${2:-networks/my_torch_network_1-2.nn}"
NAME="$(basename "${DATASET%.*}")"
OUT="results/${NAME}"

cd "$(dirname "$0")/.."
mkdir -p results

[ -x ./my_torch_analyzer ] || make

# The label is the tail of the line: either "Nothing", or a two-word class
# such as "Checkmate White". Everything before it is the FEN position.
awk '{ if ($(NF-1) == "Check" || $(NF-1) == "Checkmate") print $(NF-1)" "$NF; else print $NF }' \
    "$DATASET" > "${OUT}_expected.txt"

./my_torch_analyzer --predict "$NETWORK" "$DATASET" > "${OUT}_predicted.txt"

paste -d'|' "${OUT}_expected.txt" "${OUT}_predicted.txt" | awk -F'|' '
    { total++; if ($1 == $2) correct++ }
    END { printf "\n%s: %d/%d correct = %.2f%%\n\n", "'"$NAME"'", correct, total, 100 * correct / total }'

python3 scripts/metrics.py "${OUT}_expected.txt" "${OUT}_predicted.txt" "$OUT"
