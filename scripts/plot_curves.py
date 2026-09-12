#!/usr/bin/env python3
"""Generate ASCII learning curves from train_curve.dat"""
import sys

def main(data_file, output_file):
    epochs = []
    losses = []
    accs = []
    with open(data_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                epochs.append(int(parts[0]))
                losses.append(float(parts[1]))
                accs.append(float(parts[2]))

    with open(output_file, "w") as out:
        # ASCII Loss curve
        out.write("=" * 60 + "\n")
        out.write("LEARNING CURVES (Train set)\n")
        out.write("=" * 60 + "\n\n")
        
        out.write("Loss per epoch:\n")
        max_loss = max(losses)
        min_loss = min(losses)
        for i, (e, l) in enumerate(zip(epochs, losses)):
            bar_len = int(40 * (l - min_loss) / (max_loss - min_loss + 0.001))
            bar = "#" * bar_len
            out.write(f"Epoch {e:2d} | {l:.4f} |{bar}\n")
        
        out.write("\nAccuracy per epoch:\n")
        max_acc = max(accs)
        min_acc = min(accs)
        for i, (e, a) in enumerate(zip(epochs, accs)):
            bar_len = int(40 * (a - min_acc) / (max_acc - min_acc + 0.001))
            bar = "#" * bar_len
            out.write(f"Epoch {e:2d} | {a:6.2f}% |{bar}\n")
        
        out.write("\n" + "=" * 60 + "\n")
        out.write(f"Summary: Loss {losses[0]:.4f} -> {losses[-1]:.4f} (↓{losses[0]-losses[-1]:.4f})\n")
        out.write(f"         Acc  {accs[0]:.2f}% -> {accs[-1]:.2f}% (↑{accs[-1]-accs[0]:.2f}%)\n")

    print(f"Wrote {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: plot_curves.py <train_curve.dat> <output.txt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
