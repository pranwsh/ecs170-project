# ORL repeat validation analysis

Six additional **validation-only** runs paired model and split seeds 3 and 4 across baseline, kernel5, and average_pool. They match the existing seed-2 validation configuration: 20 epochs, batch size 32, Adam learning rate 0.001, CPU, 320 training images and 40 stratified validation images (one per class). OMP_NUM_THREADS and MKL_NUM_THREADS were both 4. Existing seed-2 results are included for context. Each entry is final epoch accuracy (correct/40), not the best observed epoch.

| Model / split seed | Baseline | Kernel5 | Average pool |
|---|---:|---:|---:|
| 2 / 2 | 92.5% (37) | 92.5% (37) | 92.5% (37) |
| 3 / 3 | 95.0% (38) | 92.5% (37) | 97.5% (39) |
| 4 / 4 | 92.5% (37) | 97.5% (39) | 92.5% (37) |
| Mean across the three pairs | 93.33% | 94.17% | 94.17% |

The ranking changes between repeats; the differences are too small to support a robust winner. Each validation image changes accuracy by 2.5 percentage points, validation sets and training sets overlap across splits, and model and split seeds change together, so their individual effects cannot be disentangled. Three pairs provide limited evidence and the mean is descriptive, not an independent-sample confidence estimate. These numbers are for model development, not an unbiased final test estimate; no new test evaluation was performed.

The complete new artifacts, including configuration, history, final metrics, predictions, curves and model weights, are under `result/stage_3_result/experiments/orl/repeat_seed{3,4}_{baseline,kernel5,average_pool}/`. To reproduce from this directory in the development environment, choose fresh output paths:

```sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for s in 3 4; do
  for v in baseline kernel5 average_pool; do
    python -m script.stage_3_script.run_experiment --dataset orl --mode validation \
      --variant "$v" --output "result/stage_3_result/experiments/orl/repeat_seed${s}_${v}" \
      --epochs 20 --batch-size 32 --learning-rate 0.001 \
      --seed "$s" --split-seed "$s" --device cpu
  done
done
```
