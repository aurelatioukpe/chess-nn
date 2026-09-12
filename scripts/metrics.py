#!/usr/bin/env python3
"""Generate confusion matrix & per-class precision/recall/F1 from expected/predicted files."""
import sys
from collections import defaultdict

LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]

def main(expected_path, preds_path, output_prefix):
    with open(expected_path) as f:
        expected = [l.strip() for l in f if l.strip()]
    with open(preds_path) as f:
        preds = [l.strip() for l in f if l.strip()]

    assert len(expected) == len(preds), f"Mismatch: {len(expected)} vs {len(preds)}"

    # Confusion matrix
    cm = defaultdict(lambda: defaultdict(int))
    for e, p in zip(expected, preds):
        cm[e][p] += 1

    # Write confusion matrix
    with open(f"{output_prefix}_confusion.txt", "w") as out:
        out.write("CONFUSION MATRIX (rows=expected, cols=predicted)\n")
        out.write("-" * 90 + "\n")
        header = f"{'':20}" + "".join(f"{l:>16}" for l in LABELS)
        out.write(header + "\n")
        out.write("-" * 90 + "\n")
        for row in LABELS:
            line = f"{row:20}" + "".join(f"{cm[row][col]:>16}" for col in LABELS)
            out.write(line + "\n")
        out.write("-" * 90 + "\n")

    # Per-class metrics
    tp = {}
    fp = {}
    fn = {}
    for label in LABELS:
        tp[label] = cm[label][label]
        fp[label] = sum(cm[other][label] for other in LABELS if other != label)
        fn[label] = sum(cm[label][other] for other in LABELS if other != label)

    with open(f"{output_prefix}_metrics.txt", "w") as out:
        out.write(f"{'Class':20}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>10}\n")
        out.write("-" * 66 + "\n")
        total_tp = 0
        total_support = 0
        weighted_prec = 0
        weighted_rec = 0
        for label in LABELS:
            support = tp[label] + fn[label]
            prec = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
            rec = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            out.write(f"{label:20}{prec:>12.4f}{rec:>12.4f}{f1:>12.4f}{support:>10}\n")
            total_tp += tp[label]
            total_support += support
            weighted_prec += prec * support
            weighted_rec += rec * support
        out.write("-" * 66 + "\n")
        acc = total_tp / total_support if total_support > 0 else 0
        w_prec = weighted_prec / total_support if total_support > 0 else 0
        w_rec = weighted_rec / total_support if total_support > 0 else 0
        w_f1 = 2 * w_prec * w_rec / (w_prec + w_rec) if (w_prec + w_rec) > 0 else 0
        out.write(f"{'Accuracy':20}{acc:>12.4f}\n")
        out.write(f"{'Weighted avg':20}{w_prec:>12.4f}{w_rec:>12.4f}{w_f1:>12.4f}{total_support:>10}\n")

    print(f"Wrote {output_prefix}_confusion.txt and {output_prefix}_metrics.txt")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: metrics.py <expected.txt> <preds.txt> <output_prefix>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
