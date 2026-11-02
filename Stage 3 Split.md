# ECS 170 Stage 3 - Four-Person Work Distribution

## Goal and workload

Build and evaluate CNNs for MNIST digits, ORL faces, and CIFAR-10 objects using the professor's template. Produce learning curves, configuration comparisons, final test results, and one report of no more than five pages.

This is an equal-effort plan: each person has **10 planning points**, including coding, experiments, testing, and writing. Points estimate relative effort, not measured hours or guaranteed completion time. After the first working baselines, compare actual effort and move a small shared task if someone is overloaded. Waiting for training to finish is not a substitute for implementation or analysis work.

Unlike a purely functional split, three people each own a complete dataset/model result. The easier/smaller datasets receive more shared tasks; the CIFAR owner receives fewer. The fourth person builds the common training and execution pipeline rather than another model.

## Verified dataset details

The supplied files are pickled datasets named `MNIST`, `ORL`, and `CIFAR`, not the Stage 2 CSVs. The supplied example accesses `data['train']` and `data['test']`, each containing records with `image` and `label`.

| Dataset | Training images | Test images | Stored image shape | Original labels |
|---|---:|---:|---|---|
| MNIST | 60,000 | 10,000 | 28 x 28 | 0-9 |
| ORL | 360 | 40 | 112 x 92 x 3 | 1-40 |
| CIFAR-10 | 50,000 | 10,000 | 32 x 32 x 3 | 0-9 |

The actual MNIST arrays are float64 with values from 0 to 255; they are not already scaled to 0-1. ORL and CIFAR arrays are uint8. ORL's three channels are identical; use one channel. Each ORL person has nine supplied training images and one supplied test image.

## Workload summary

| Member | Main ownership | Shared responsibility | Planned effort |
|---|---|---|---:|
| Daniel Park | All three loaders; MNIST CNN and its experiments | Data preparation and sample visualizations | 10 points |
| Person 2 | ORL CNN and its experiments | Evaluation metrics and plotting that support both 10 and 40 classes | 10 points |
| Person 3 | CIFAR CNN and its experiments | Detailed CIFAR results and error analysis | 10 points |
| Person 4 | Shared training and execution pipeline | Integration, reproducible result saving, report assembly | 10 points |

## Daniel Park - Data preparation and MNIST

**Proposed files:** `Dataset_Loader_MNIST.py`, `Dataset_Loader_ORL.py`, `Dataset_Loader_CIFAR.py`, `Visualize_Data.py`, `Method_CNN_MNIST.py`, and corresponding tests.

Responsibilities:

1. Adapt the supplied loading example into reusable dataset loaders compatible with the professor's base class. Small common loading code may be shared where useful; do not build a general dataset framework.
2. Return float32 image arrays in channel-first order and int64 labels. Support loading the provided training and testing partitions separately.
3. Divide pixels by 255 for the shared baseline. Add MNIST's single channel, select one ORL channel, and reorder CIFAR's channels. Convert ORL labels from 1-40 to 0-39 for training, keeping their original identity available for display.
4. Save a small labeled sample grid and class-count chart for each dataset. Do not add the unhelpful raw/normalized pixel histograms from Stage 2.
5. Implement the MNIST CNN architecture on top of Person 4's common training class.
6. Run the MNIST baseline and two configuration variants listed below, then the final chosen MNIST test run.
7. Test all loaders and MNIST input/output shapes; write the dataset/preprocessing and MNIST report material.

**Deliverables:** three working loaders, dataset checks and sample figures, MNIST architecture, its experiment table and curves, final MNIST evaluation, and approximately one page of report material.

**Effort:** loaders/preparation/visualization 4 points; MNIST architecture and experiments 4; tests and writing 2.

## Person 2 - ORL and evaluation tools

**Proposed files:** `Method_CNN_ORL.py`, `Evaluate_Metrics.py`, `Convergence_Plots.py`, `Error_Analysis.py`, and corresponding tests.

Responsibilities:

1. Implement a modest CNN accepting one-channel 112 x 92 images and producing 40 class scores. Pay attention to the small training set and the size of the classifier after convolution/pooling.
2. Adapt useful Stage 2 metrics and plots for all three datasets. Remove assumptions that every task has exactly ten classes; ORL needs 40.
3. Provide accuracy and multiclass precision, recall, and F1 with clearly named averaging methods, learning curves, and confusion-matrix/error-display functions. Keep these separate from the training loop.
4. Run the ORL baseline and two configuration variants below using the same three pre-agreed seed/split pairs for every configuration. Report each validation score and the mean and sample standard deviation across the three runs. These repeats are inexpensive relative to the large datasets and help show the variability of a 40-image validation set.
5. Select the ORL configuration using the repeated validation results. Agree on the final training seed before test evaluation; do not choose a seed by its test accuracy. Train the final model on all 360 supplied training images, evaluate the 40 supplied test images, and explain errors. One test error changes ORL accuracy by 2.5 percentage points, so report correct/total counts too.
6. Test ORL model shapes and the shared evaluation functions; write the evaluation-method and ORL report material.

**Deliverables:** ORL architecture and results, class-count-aware evaluation/plotting tools, tests, and approximately one page of report material.

**Effort:** ORL architecture and experiments 5 points; reusable evaluation/plots 3; tests and writing 2.

## Person 3 - CIFAR-10

**Proposed files:** `Method_CNN_CIFAR.py`, a CIFAR experiment configuration file if needed, and corresponding tests. Use the shared runner rather than creating another training pipeline.

Responsibilities:

1. Implement a CNN accepting three-channel 32 x 32 images and producing ten class scores. Preserve the RGB information.
2. Verify feature-map dimensions through convolution and pooling and make the selected experiment settings easy to change.
3. Train the CIFAR baseline and two variants below, using the same validation split and seed for comparisons. Start with a small smoke test before launching full runs.
4. Analyze training and validation curves, per-class performance, and common object confusions using Person 2's functions.
5. Run the final selected CIFAR configuration on the supplied test partition and document the selected settings, measured results, and limitations.
6. Test the model's shapes and learning behavior; write the CIFAR architecture, experiments, and results report material.

**Deliverables:** CIFAR architecture, baseline and comparison runs, final evaluation, error analysis, tests, and approximately one page of report material.

**Effort:** CIFAR architecture/training/experiments 8 points; tests, analysis presentation, and writing 2. No additional shared infrastructure is assigned because this dataset is the most demanding classification task in this plan.

## Person 4 - Common training and integration

**Proposed files:** `Method_CNN.py` for common training behavior, `Setting_Stage3.py`, `Result_Saver.py`, `script_cnn.py`, `run_experiments.py`, and integration tests.

Responsibilities:

1. Implement one common mini-batch training loop used by the three model classes. Compute per-epoch training/validation measurements and final predictions in mini-batches too, rather than passing a whole dataset through the CNN at once. Aggregate loss and accuracy over all examples, including the final partial batch. Dataset owners define their own layers and forward passes. Preserve the professor's base-class conventions and avoid three copied training loops.
2. Support train/validation/final-test modes, seed setting, selected optimizer/loss settings, and per-epoch history. A simple CPU path is sufficient; GPU execution is optional in the assignment.
3. Create the settings and command-line entry point selecting dataset and model. Build the agreed validation splits from training data only.
4. Save configuration, evaluation split, metrics, history, and predictions in distinct run folders. Reuse cached results only when their experiment settings match. Keep validation and test results separate in summaries.
5. Test the integrated pipeline on all three datasets with short runs. Each dataset owner remains responsible for debugging their own loader/model and producing their full experimental results.
6. Assemble the four report contributions, write the brief introduction/conclusion, and enforce the five-page limit. Each contributor supplies finished text, captions, and compact figures; Person 4 does not write everyone else's sections.

**Deliverables:** shared trainer, settings/scripts, reproducible execution and result saving, integration checks, and assembled report.

**Effort:** common trainer/settings 5 points; execution/result saving 2; integration verification/report assembly 3.

## Interfaces to agree on before coding

These are team implementation choices, not additional professor requirements.

| Item | Contract |
|---|---|
| Loader output for one partition | `{'X': images, 'y': labels}` |
| MNIST inputs | `(N, 1, 28, 28)` |
| ORL inputs | `(N, 1, 112, 92)` |
| CIFAR inputs | `(N, 3, 32, 32)` |
| Labels | `(N,)`, integer class IDs: 0-9 or 0-39 |
| CNN forward output | `(batch_size, num_classes)` raw scores |
| Setting/model data | Validation run: `{'train': {'X': X_train, 'y': y_train}, 'val': {'X': X_val, 'y': y_val}}`. Final run: `{'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}`. Do not load the supplied test partition during validation runs. |
| Training API | `train_model(X, y, X_val=None, y_val=None)` updates model weights and `self.history`; `test(X)` returns one predicted class per image without updating weights. |
| Run API | `run()` reads `self.data`, trains the model, and returns `{'pred_y': predictions, 'true_y': labels}` for `val` in validation mode or `test` in final mode. The setting records which partition was evaluated. |
| Training history | `train_loss`, `train_acc`, `val_loss`, `val_acc`; validation lists stay empty for a final run without validation data. |

Person 4 supplies the common training API and a minimal usable stub early. Model owners can test forward passes using synthetic arrays immediately; they do not need to wait for full loading/integration. Person 2 can test metrics with small known label/prediction arrays.

## Bounded experiment plan

Start with a simple working baseline for each dataset. Each model owner then runs two variants changing one selected setting at a time while holding that dataset's split, seed, epoch budget, optimizer, and other settings fixed.

| Owner | Dataset | Variant 1 | Variant 2 |
|---|---|---|---|
| Daniel | MNIST | Kernel size, e.g. 3 vs 5 | Max versus average pooling |
| Person 2 | ORL | Number of convolution blocks | Classifier hidden-layer size |
| Person 3 | CIFAR | Convolution stride | Cross-entropy versus MSE on softmax probabilities and one-hot targets |

These cover a practical selection of the architecture/configuration examples in requirement 3-5. Stride, kernel, or depth changes must still produce valid tensor sizes. Record actual padding values; additional padding-only comparisons are optional if time permits. Do not promise an exhaustive sweep of every possible setting.

The MSE conversion, if selected, belongs in the shared loss configuration with Person 4, using the existing tested Stage 2 wrapper where suitable. Person 3 owns running and interpreting that comparison; do not duplicate the trainer.

## Validation and final evaluation

- Preserve every supplied training/testing partition. Model selection uses only the supplied training data.
- Proposed MNIST split: 48,000 training and 12,000 validation examples.
- Proposed CIFAR split: 40,000 training and 10,000 validation examples.
- Proposed ORL split: eight training and one validation image per person, giving 320 training and 40 validation examples; retain all 40 identities in both groups. Do not split by person, which would change this closed-set classification task.
- Keep the same split within each paired comparison. ORL uses the same three pre-agreed seed/split pairs across configurations; report individual scores plus their mean and sample standard deviation. These repeats change only the held-out training examples and random model initialization, never the supplied test partition.
- After selecting settings, train a fresh final model on the complete supplied training partition and evaluate the supplied test partition. Each dataset owner runs this through Person 4's pipeline.
- Report validation comparisons separately from final test results. Do not calculate improvements by subtracting validation accuracy from test accuracy.

## Report and completion checks

Each person prepares approximately one page of polished material. The final report must fit within five pages including the required content; reserve room for compact combined figures/tables and references as the provided report template allows. Page allocation is a planning guide, not a requirement to fill space.

Before completion, confirm:

1. All three provided datasets load, with tested shapes, channels, label mappings, and partition sizes (3-1).
2. Three CNN architectures work through the shared template-based trainer (3-2).
3. Each dataset has learning curves, a final test evaluation, and a documented configuration (3-3 and 3-4).
4. Configuration comparisons and honest conclusions are included (3-5).
5. Everyone has tested their components and supplied their report contribution; the combined report is no more than five pages (3-6).

Use separate branches and clear file ownership, preserve Stage 1/2 implementations, and review each other's changes. No Stage 3 submission date is established by the supplied description; agree on the actual deadline and internal milestones separately.

## Sources checked

- `ECS_170_Fall_2026_Project_Description.docx.pdf`, pages 3-4, requirements 3-1 through 3-7.
- `stage_3_data.zip`: `ReadMe.txt`, `script_data_loader.py`, and the actual serialized MNIST/ORL/CIFAR contents.

The dataset sizes, image shapes, label ranges, pixel ranges, and identical ORL channels were verified from the supplied files. The allocation, effort points, validation splits, and experiment menu are proposed team decisions.
