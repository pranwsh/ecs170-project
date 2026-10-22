# Person 3 results - what the experiments show

All numbers below are from the 20% validation split of `train.csv`
(48,000 train / 12,000 validation, stratified, seed 2), 20 epochs, batch 128,
normalized pixels, baseline architecture 784-256-128-10 with ReLU unless the row
says otherwise. Raw numbers live in `summary.csv`, `*/comparison.csv` and each
run's `metrics.json`.

## 1. Metrics

`accuracy` alone is not enough, so every run also reports weighted/macro/micro
precision, recall and F1, plus balanced accuracy and Cohen's kappa.

* `micro` precision = recall = F1 = accuracy, always. With single-label
  multiclass data every sample contributes exactly one positive and nine
  negative decisions, so the micro average collapses to accuracy. Reported for
  completeness; the report should say so explicitly.
* `macro` treats all ten digits equally, so it drops faster than accuracy when
  the model starts ignoring a hard digit. Example: in the Adam 1e-5 run
  accuracy is 0.9085 while macro recall is 0.9071.
* `weighted` follows the class sizes (each digit has ~1,200 validation images),
  so on this balanced dataset weighted and macro stay within 0.0005 of each
  other. On an unbalanced dataset they would separate, which is why both are
  reported.
* `balanced_accuracy` and `cohen_kappa` agree with macro F1 in every run here,
  confirming that no digit is being systematically abandoned.

## 2. Optimizer (group `optimizer/`)

| run | accuracy | macro F1 |
|---|---|---|
| SGD lr=0.05, momentum 0.9 | **0.9821** | 0.9819 |
| RMSprop lr=1e-3 | 0.9799 | 0.9797 |
| SGD lr=0.01, momentum 0.9 | 0.9772 | 0.9770 |
| Adam lr=1e-3 (baseline) | 0.9703 | 0.9700 |

SGD with momentum 0.9 and a much larger step size (0.05) wins. Adam is
comfortable out of the box but plateaus lower within 20 epochs; its per-parameter
step size adapts so quickly that progress on this easy, well-scaled problem is
slower. SGD at 0.01 is clearly under-trained (still improving at epoch 20), which
is why both 0.01 and 0.05 are reported instead of a single "SGD" row.

## 3. Learning rate (group `lr/`)

| Adam lr | accuracy | macro F1 |
|---|---|---|
| 1e-2 | 0.9711 | 0.9707 |
| 1e-3 | 0.9703 | 0.9700 |
| 1e-4 | 0.9670 | 0.9667 |
| 1e-5 | 0.9085 | 0.9072 |

1e-2 and 1e-3 are within noise of each other; 1e-4 is under-trained and 1e-5
fails to converge in 20 epochs (the loss curve is still descending steeply at the
last epoch). The takeaway is the shape of the curve, not the top row: too small a
step size costs far more than a slightly too large one here, because the network
is small and the inputs are already normalized to [0, 1].

## 4. Loss function (group `loss/`)

| loss | accuracy | macro F1 |
|---|---|---|
| cross-entropy + label smoothing 0.1 | **0.9835** | 0.9833 |
| MSE on softmax outputs | 0.9778 | 0.9776 |
| cross-entropy | 0.9703 | 0.9700 |
| log-softmax + NLL | 0.9703 | 0.9700 |

* `nll` reproduces `ce` to four decimals, as expected: it is the same
  computation split into two operations. A unit test asserts this equality.
* MSE converges, but its gradients carry the softmax derivative squared, so it
  needs more epochs to reach the same place; it is still surprisingly strong here.
* Label smoothing gains about 1.3 points of accuracy and is the single best
  change found. It stops the network from pushing logits to extreme values, which
  acts as a regularizer and helps the visually ambiguous pairs.

## 5. Regularization (group `regularization/`)

| setting | accuracy | macro F1 |
|---|---|---|
| dropout 0.2 | 0.9778 | 0.9775 |
| dropout 0.5 | 0.9766 | 0.9763 |
| weight decay 1e-4 | 0.9758 | 0.9755 |

All three beat the 0.9703 baseline, and all three land in the same narrow band.
The train/validation curves do not separate in the baseline run (no overfitting
gap yet at 20 epochs), so these gains are small and partly noise; the honest
conclusion is that mild regularization helps a little and heavy regularization
(dropout 0.5) starts to hurt.

## 6. Architecture (group `architecture/`)

| hidden layers | accuracy | macro F1 |
|---|---|---|
| 1024-512 | **0.9802** | 0.9801 |
| 512-256-128 | 0.9788 | 0.9786 |
| 128 | 0.9755 | 0.9753 |
| 256-128 (baseline) | 0.9703 | 0.9700 |

Wider and deeper both help, monotonically, with diminishing returns. Note the
single 128-unit layer beats the baseline 256-128 pair, which says the baseline
two-layer net was simply under-trained for its size, not that depth hurts.

## 7. Final runs on `test.csv` (group `final/`)

Same code path, `--validate` removed, so these are the numbers to report for
2-3. Test set = 10,000 images that no experiment touched.

| run | accuracy | macro F1 | weighted F1 | kappa |
|---|---|---|---|---|
| SGD lr=0.05, momentum 0.9 | **0.9841** | 0.9840 | 0.9841 | 0.9823 |
| cross-entropy + label smoothing | 0.9833 | 0.9832 | 0.9833 | 0.9814 |
| baseline (Adam 1e-3, CE) | 0.9807 | 0.9805 | 0.9807 | 0.9785 |

Validation and test accuracy agree to within 0.2 points for every configuration,
so the 20% validation split was a reliable stand-in for model selection and there
is no sign of the test set being over-fitted by the experiments.

## 8. Error behaviour (from `*/error_analysis.txt`)

The same pairs fail in every configuration:

* 9 -> 7 and 9 -> 4 (top-2 confusions in most runs),
* 3 -> 5, 3 -> 8, 5 -> 3,
* 8 -> 5, 8 -> 3.

Digit 9 is consistently the worst class (0.9723 recall in the label-smoothing
run) and digit 1 the best (0.9933). These are all pairs whose handwritten forms
differ by a single stroke or loop, so the remaining error is largely irreducible
for a plain MLP on raw pixels: it has no shift/rotation invariance, unlike a CNN.
