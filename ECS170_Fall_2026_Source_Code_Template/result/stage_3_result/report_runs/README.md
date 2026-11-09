# Extra Stage 3 runs for the report

These runs add to the recorded runs in `../experiments/`, which trained CIFAR for only 2 epochs and ORL with one seed. Same code and protocol (`script.stage_3_script.run_experiment`); only the epoch count (CIFAR) and the seeds (ORL) differ. Predictions, histories, metrics and learning curves are saved; the `model.pt` weight files are not committed.

## CIFAR-10, full 20 epochs (`cifar/`)

Validation runs on a stratified 10% of the training set (45,000 / 5,000), seed 2, split seed 2, batch size 128, learning rate 0.001. The variant with the best validation accuracy was then retrained on all 50,000 training images and evaluated once on the test set (`selection.json` records the choice).

| Run | Evaluated on | Accuracy |
|---|---|---|
| `baseline` | validation | 72.48% |
| `kernel5` | validation | 73.26% (selected) |
| `average_pool` | validation | 71.74% |
| `final` (5x5 kernels) | test | 72.78% (7,278 / 10,000) |

## ORL, extra seeds (`orl/`)

Batch size 32 as in the recorded seed-2 runs; model seed and split seed both set to 3 or 4.

| Run | Evaluated on | Correct / 40 |
|---|---|---|
| `baseline_seed3`, `kernel5_seed3`, `average_pool_seed3` | validation | 38, 36, 39 |
| `baseline_seed4`, `kernel5_seed4`, `average_pool_seed4` | validation | 37, 37, 37 |
| `final_baseline_seed3`, `final_baseline_seed4` | test | 35, 37 |

With the recorded seed-2 runs (validation 37, 37, 37; test 40) the baseline's ORL test accuracy is 93.3% +/- 6.3 points over seeds 2-4.

## Commands

```sh
# from ECS170_Fall_2026_Source_Code_Template, after `git lfs pull`
python -m script.stage_3_script.run_experiment --dataset cifar --mode validation --variant baseline --output result/stage_3_result/report_runs/cifar/baseline --seed 2 --split-seed 2
python -m script.stage_3_script.run_experiment --dataset cifar --mode test --variant kernel5 --output result/stage_3_result/report_runs/cifar/final --seed 2
python -m script.stage_3_script.run_experiment --dataset orl --mode validation --variant baseline --output result/stage_3_result/report_runs/orl/baseline_seed3 --seed 3 --split-seed 3 --batch-size 32
python -m script.stage_3_script.run_experiment --dataset orl --mode test --variant baseline --output result/stage_3_result/report_runs/orl/final_baseline_seed3 --seed 3 --batch-size 32
python -m script.stage_3_script.make_report_figures   # report figures
```

Repeat with `--variant kernel5` or `average_pool` for the other variants, and `--seed 4 --split-seed 4` for ORL seed 4.
