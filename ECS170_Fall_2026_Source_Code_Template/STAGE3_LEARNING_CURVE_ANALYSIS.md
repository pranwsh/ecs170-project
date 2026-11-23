# Stage 3 learning-curve analysis (validation only)

## Existing runs

The existing validation histories under `result/stage_3_result/experiments` use seed and split seed 2. MNIST and ORL variants were trained for 20 epochs; CIFAR variants were trained for only two. All CIFAR variants improved validation accuracy and loss substantially between epochs 1 and 2, so these runs do not establish convergence. The two-epoch ordering was kernel5 61.98%, baseline 57.48%, average pooling 50.92%.

MNIST validation accuracy at epoch 20 is 98.68% baseline, 98.95% kernel5, and 98.62% average pooling. Minimum validation losses occurred at epochs 12, 10, and 10 respectively, while training accuracy continued rising to 99.81%, 99.81%, and 99.65%; the increasing late validation loss suggests overfitting despite nearly saturated accuracy. ORL validation accuracy at epoch 20 is 92.5% for all variants, with peak values 97.5%, 95%, 95% for baseline, kernel5 and average pooling. ORL validation has only 40 images, so one image changes accuracy by 2.5 percentage points; epoch-wise peaks are noisy and should not be treated as reliable generalization estimates.

## Longer, equal-budget CIFAR validation

Fresh CPU runs used validation mode, four epochs per variant, 45,000 training and 5,000 stratified validation images, batch size 128, Adam learning rate 0.001, model seed 2 and split seed 2. Both `OMP_NUM_THREADS` and `MKL_NUM_THREADS` were set to 4. The existing two-epoch histories reproduce to the shown precision in these runs. Results below are validation, not test results:

| Variant | Epoch 1 accuracy | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 4 validation loss | Epoch 4 training accuracy |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 48.82% | 57.48% | 63.46% | 65.68% | 0.9828 | 68.49% |
| kernel5 | 52.84% | 61.98% | 66.64% | 68.36% | 0.9162 | 72.55% |
| average pooling | 45.30% | 50.92% | 53.04% | 56.84% | 1.1891 | 59.04% |

The same ranking holds at four epochs, but accuracy is still rising and validation loss falling in every case; four epochs is not evidence of convergence. At epoch 4 kernel5 leads baseline by 2.68 percentage points on this one shared split and seed, not a demonstrated population-wide advantage. Training/validation accuracy gaps (baseline 2.81 points, kernel5 4.19, average pooling 2.20) show some emerging fit gap; do not infer optimal training duration from these short curves. Further equal-budget training and repeated seeds/splits would better assess stability; select all settings using validation only. No new test evaluation was performed.

Reproduction, from the template directory within `nix develop` (choose fresh output directories):

```sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for v in baseline kernel5 average_pool; do
  python -m script.stage_3_script.run_experiment --dataset cifar --mode validation \
    --variant "$v" --output "/tmp/stage3_long_4epoch_$v" --epochs 4 \
    --batch-size 128 --learning-rate 0.001 --seed 2 --split-seed 2 --device cpu
done
```

Artifacts from this run are in `/tmp/stage3_long_4epoch_{baseline,kernel5,average_pool}` and are temporary, not committed. Environment: Nix development shell, Python 3.14.8, PyTorch 2.13.0 CPU, CUDA unavailable. Stage 3 experiment, data, trainer and MNIST unittests: 21 passed. These synthetic/unit tests do not establish model convergence or statistical significance.
