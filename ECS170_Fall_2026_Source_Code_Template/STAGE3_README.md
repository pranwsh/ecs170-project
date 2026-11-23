# Stage 3: CNN image classification

Run commands from `ECS170_Fall_2026_Source_Code_Template`. Install Python dependencies: numpy, PyTorch, scikit-learn and matplotlib. Obtain the instructor's datasets with `git lfs pull` from the repository root. Only unpickle trusted files.

## Implementation

The dataset loaders produce float32 images scaled to [0,1] in NCHW order. ORL uses one grayscale channel and maps person IDs 1–40 to class indices 0–39. MNIST and CIFAR labels are already 0–9. The supplied training and test partitions are preserved. Model outputs are raw logits; the trainer uses Adam and cross-entropy by default.

The experiment runner supports baseline, 5×5 kernel, and average pooling variants. A stratified 10% of the supplied training data is held out for validation, or one example per class if that is larger (40 examples for ORL). Use identical split and model seeds for paired comparisons. Record validation results and select a configuration *before* running test mode. Test mode trains a fresh model on the complete supplied training partition; it must not be used for hyperparameter or seed selection. In ORL the 40-image test set means a single error changes accuracy by 2.5 percentage points.

Example commands (these execute training, and were not run as part of setup):

```sh
python -m script.stage_3_script.run_experiment --dataset mnist --mode validation --variant baseline --output result/stage_3_result/mnist/baseline --seed 2 --split-seed 2
python -m script.stage_3_script.run_experiment --dataset mnist --mode validation --variant kernel5 --output result/stage_3_result/mnist/kernel5 --seed 2 --split-seed 2
python -m script.stage_3_script.run_experiment --dataset mnist --mode validation --variant average_pool --output result/stage_3_result/mnist/average_pool --seed 2 --split-seed 2
# After selecting the variant from validation, replace baseline with the selected variant:
python -m script.stage_3_script.run_experiment --dataset mnist --mode test --variant baseline --output result/stage_3_result/mnist/final --seed 2
```

Repeat for `orl` and `cifar`. Use `--epochs`, `--batch-size`, `--learning-rate`, `--device` and `--data-dir` to override defaults. Each new output directory stores config, epoch history, macro metrics, confusion matrix, predictions, learning curves, and model weights; existing directories are not overwritten. Loss curves from final training contain only training loss; validation runs include both training and validation. Macro precision/recall/F1 average equally over all dataset classes, with zero-division set to zero. ORL saved prediction indices are 0-based; add 1 to display person IDs.

Run non-training checks with `python -m unittest tests.test_stage3_experiments tests.test_stage3_data -v`. Existing trainer and MNIST tests also run short synthetic training; run them separately when training is permitted: `python -m unittest tests.test_stage3_trainer tests.test_stage3_mnist -v`.

`Stage_3_Report_Draft.docx` is a copy of the supplied template with preliminary text and explicit placeholders. Complete the missing team information, source link, plots and measured results after experiments, and verify the final document is at most five pages.

## CIFAR-10 error analysis (no training)

For an existing CIFAR run, regenerate its per-class precision, recall, support,
and error counts and its ranked **directional** confusions from the saved
`config.json`, `predictions.npz`, and `metrics.json`:

```sh
python -m script.stage_3_script.analyze_cifar_errors result/stage_3_result/experiments/cifar/final_2epoch --output cifar_error_analysis.json
python -m unittest discover -s tests -p 'test_stage3_cifar_analysis.py' -v
```

Omit `--output` to print JSON. The output file must not already exist. Matrix
rows are true classes and columns are predictions; confusion fractions use the
true-class support as denominator. Zero-support recall and zero-prediction
precision are defined as zero. The tool verifies the saved confusion matrix,
correct count and total against the predictions; use the same run's artifacts
together. It does not load data or retrain, and analyzing test errors must not
be used to select a model or seed.

## Experiment summaries and reproducibility

From this directory, regenerate the tables after adding runs (the output is
replaced if it already exists):

```sh
python -m script.stage_3_script.summarize_experiments --output result/stage_3_result/experiments/SUMMARY.md
python -m unittest discover -s tests -p 'test_stage3*.py' -v
```

The summary lists validation and test separately and includes the run settings;
compare only runs with matching budgets and paired seeds. See
`STAGE3_LEARNING_CURVE_ANALYSIS.md` and `STAGE3_ORL_REPEAT_ANALYSIS.md` for
reproduction commands and caveats. The additional experiments were run from
base commit `78b691ee709663d3acab87d6c3539ce4fe005c0f` in the Nix development
environment; new analysis and tests were uncommitted. Capture the current commit
and `git status` when reproducing, since uncommitted changes affect provenance.

For the existing two-epoch CIFAR final test, dog recall is 26.3%, compared with
82.6% for automobile. The largest directional confusion is dog → cat (300 of
1,000 dogs). These are descriptive test-set findings, not a basis for tuning.
The complete per-class and directional results are in
`result/stage_3_result/experiments/cifar/final_2epoch_error_analysis.json`.
