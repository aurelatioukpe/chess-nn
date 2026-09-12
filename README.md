# chess-nn

A neural network written from scratch in C++17 without ML library, BLAS, or
autograd. It classifies chess positions from FEN notation into five states:
nothing, check against white, check against black, checkmate against white,
and checkmate against black.

**72.52%** on 1,350 held-out positions, from 51,845 parameters.

```bash
make
./scripts/evaluate.sh          # reproduces the table below
```

---

## Results

Evaluated on `data/val_9k.txt`, containing 1,350 positions the network never saw during
training. Five balanced classes, so chance is 20%.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Nothing | 0.725 | 0.673 | 0.698 | 447 |
| Check White | 0.651 | 0.543 | 0.592 | 223 |
| Check Black | 0.607 | 0.653 | 0.629 | 222 |
| Checkmate White | 0.780 | **0.901** | 0.836 | 221 |
| Checkmate Black | 0.835 | **0.899** | 0.866 | 237 |
| **Overall accuracy** | | | **0.7252** | 1,350 |

Other sets, same network: `test_9k` 69.78% (1,350), `validation_60` 73.33% (60),
`test_predict` 55.00% (20).

Inference runs at roughly **0.5 ms per position**, single-threaded.

## What the errors say

The headline number is not the interesting part. This is:

```
                    Nothing   Check W   Check B   Mate W   Mate B
Nothing                 301        48        73       15       10
Check White              61       121         0       41        0
Check Black              45         0       145        0       32
Checkmate White           5        17         0      199        0
Checkmate Black           3         0        21        0      213
```

Read the zeros. The network never once confuses white with black. Not a
single Check White predicted as Check Black, in either direction, across 903
non-empty positions. The colour axis is learned perfectly.

Every error lives on the other axis: how serious the threat is. Check White is
missed 102 times out of 223. Of these, 61 fall back to nothing and 41 escalate to
checkmate. Nothing is over-called as check 121 times.

There is a clean reason for this, and it is a lesson about representation
rather than about tuning.

**Colour is in the input.** The encoding gives each of the 64 squares twelve
one-hot slots: six piece types by two colours. "Which king is under threat" is
almost a lookup. Find the attacked king's plane. A linear layer can express
that, so the network learns it immediately and completely.

**Severity is not in the input.** Check means the king is attacked. Checkmate
means the king is attacked and no legal move escapes. The second requires
enumerating moves and testing each one. That is a search. A feed-forward network over
a flat one-hot board has no mechanism for that. It cannot simulate a move.

So it approximates. And the approximation reveals what it settled on: mate is
detected at roughly 90% while check sits at 54 to 65%. That looks wrong, as mate is the
logically harder property. But notice this: mate positions are visually
stereotyped. King on an edge or in a corner, escape squares covered, attackers
adjacent. Those are geometric facts, present in the input, and a dense layer
picks them up.

Check is the harder one to see. It can be a single bishop on a long empty
diagonal. That is a relationship between two squares that says nothing about the rest
of the board. Detecting it means tracing a ray, which is exactly the operation
this architecture lacks.

The network learned chess geometry. It did not learn chess rules.

That framing also says what would actually help, and it is not a bigger network.
An input encoding carrying attack maps, or a convolutional architecture with
some notion of board topology would work. Scaling the dense layers buys very little,
because the missing information is missing from the representation, not from
the capacity.

## What is implemented

Everything below is hand-written. The only includes are the C++ standard
library.

**Network** (`src/NeuralNetwork.cpp`)
- Forward and backward propagation, layer by layer
- Activations with their derivatives: sigmoid, ReLU, tanh, softmax, linear
- Numerically stable softmax. The maximum is subtracted before exponentiating,
  without which `exp` overflows on confident logits and the whole row becomes NaN
- Loss functions: MSE, cross-entropy, binary cross-entropy
- Xavier initialisation
- L2 regularisation folded into the weight update
- Mini-batch training through accumulated gradients
- Binary serialisation with a magic header, so a trained network reloads exactly

**Matrix** (`src/Matrix.cpp`, `src/Multiply_Matrix.cpp`, `src/Operators.cpp`)
- Flat `std::vector<double>` with computed indices rather than a vector of
  vectors. A row-of-rows layout scatters each row across the heap and defeats
  the cache on exactly the access pattern a matrix multiply performs. Flattening
  it gave roughly a **5× speedup** with no algorithmic change.

**FEN encoding** (`src/FENEncoder.cpp`, `src/fenparser.cpp`)
- Full FEN parsing: piece placement, side to move, castling rights, en passant
- 774 inputs = 8 × 8 × 12 one-hot piece planes, plus 6 features (side to move,
  four castling rights, en passant)
- The encoder reuses one preallocated buffer across calls rather than
  allocating per position. This is measurable when you encode 9,000 of them.

**CLI** (`src/Analyzer.cpp`, `src/modes/`)
- `--train` and `--predict` modes, network load/save

## Architecture of the shipped network

```
774 ──ReLU──> 64 ──ReLU──> 32 ──softmax──> 5
```

51,845 parameters. Cross-entropy loss. Learning rate 0.01 with ×0.9 decay.
The saved network carries 0.00531441, which represents six decay steps.

Small on purpose. Wider variants were tried and did not beat it, which is
consistent with the error analysis above. The ceiling here is the input
representation, not parameter count.

## Build and run

Requires `g++` with C++17 and `python3` for the metrics script.

```bash
make                                    # builds ./my_torch_analyzer
./scripts/evaluate.sh                   # val_9k, writes results/
./scripts/evaluate.sh data/test_9k.txt  # any other labelled set
```

Direct use:

```bash
./my_torch_analyzer --predict networks/my_torch_network_1-2.nn data/val_9k.txt
./my_torch_analyzer --train --save out.nn networks/my_torch_network_1-2.nn data/train_9k.txt
```

## Data format

One position per line: a FEN string, then the expected label.

```
8/8/r7/K5pk/P1q5/8/8/8 w - - 0 59 Checkmate Black
8/4k3/8/p1K5/8/8/P7/8 w - - 0 57 Nothing
```

`data/train_9k.txt` (6,300 positions) is the balanced training set.
It contains 2,117 / 2,115 / 2,068 positions across the three outcome families. 
`data/val_9k.txt` and `data/test_9k.txt` hold 1,350 each. Two much larger training 
files from the original work were dropped to keep the repository small; 
`train_9k` is sufficient to retrain.

## Limitations

- **No search.** As explained above: the architecture cannot represent "is there a legal
  escape", which is what caps check detection.
- **Trained on one distribution.** The positions come from a single generated
  corpus. Nothing here says it generalises to positions from real games.
- **No cross-validation.** One train/validation/test split, fixed. The reported
  figures are a single run, not a mean over seeds.
- **CPU only, double precision, no BLAS.** Fine at this size, and deliberately
  so. The point was to write the operations, not to make them fast.

## Credits

A three-person project, originally built as a school assignment over December
2025.

- [@aurelatioukpe](https://github.com/aurelatioukpe) — network core (forward/backward, activations, losses,
  serialisation), FEN encoding and parsing, CLI, training and evaluation runs
- [@johannes-houndete](https://github.com/johannes-houndete) — matrix layout optimisation (the flat-storage rewrite
  and the 5× speedup), training runs
- [@nandemonai](https://github.com/lepeistekirito) — validation tooling

## License

MIT. See `LICENSE`.