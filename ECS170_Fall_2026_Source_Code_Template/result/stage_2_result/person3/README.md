# Person 3 - evaluation, convergence plots and hyperparameter experiments

Everything here is produced by `script/stage_2_script/run_experiments.py` (which
calls the same pipeline as `script/stage_2_script/script_mlp.py`).

## Code

| File | Purpose |
|---|---|
| `local_code/stage_2_code/Evaluate_Metrics.py` | `Evaluate_Metrics` module (accuracy + weighted/macro/micro precision, recall, F1, balanced accuracy, Cohen's kappa) and the `compute_metrics` / `per_class_metrics` helpers |
| `local_code/stage_2_code/Convergence_Plots.py` | single-run convergence curves, multi-run overlays, metric bar charts, per-digit plots, markdown/CSV comparison tables, `collect_results` to re-read saved runs |
| `local_code/stage_2_code/Loss_Functions.py` | `ce`, `mse` (softmax + one-hot MSE), `nll` (log-softmax + NLL) and `ce_ls` (label smoothing) behind one `build_loss(name)` |
| `script/stage_2_script/script_mlp.py` | one run; new flags `--loss --activation --weight-decay --momentum --group --quiet` |
| `script/stage_2_script/run_experiments.py` | runs whole experiment groups and writes the comparison tables/figures |
| `tests/test_stage2_evaluate.py` | unit tests for metrics, losses, plots and artifact collection |

## Running

```bash
# from ECS170_Fall_2026_Source_Code_Template/
python -m unittest discover -s tests -v              # all Stage 2 tests
python -m script.stage_2_script.run_experiments --dry-run
python -m script.stage_2_script.run_experiments      # baseline, optimizer, lr, loss
python -m script.stage_2_script.run_experiments --groups architecture regularization
python -m script.stage_2_script.script_mlp --name final --hidden 512,256,128  # test.csv
```

Every run writes `config.json`, `metrics.json`, `convergence.png`,
`confusion_matrix.{csv,png}`, `misclassified_examples.png`,
`per_class_metrics.{json,png}` and `error_analysis.txt`.

## Method notes (for the report)

* All experiments use a stratified 80/20 split of `train.csv` (seed 2) so
  `test.csv` stays untouched until the final run.
* One setting changes per run, everything else stays at the shared baseline
  (784-256-128-10, ReLU, Adam 1e-3, batch 128, 20 epochs, normalized inputs).
* `micro` precision/recall/F1 equal accuracy for single-label multiclass data;
  they are reported for completeness and this should be stated in the report.
* `macro` weights every digit equally and is therefore the metric that reveals a
  model that gives up on a hard digit; `weighted` follows the class sizes.
