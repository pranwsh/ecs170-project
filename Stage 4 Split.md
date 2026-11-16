# ECS 170 Stage 4 - Four-Person Work Distribution

## Goal

Build recurrent neural networks for two tasks: classify movie reviews as positive or negative, and generate a joke from three starting words. Implement and compare **RNN, LSTM, and GRU for both tasks**. Produce learning curves, classification results, generated examples, and a report of no more than five pages.

Each person owns implementation, verification, analysis, and a report contribution. Data preparation includes persisted splits, training vocabularies, verified batches, and a data-only preprocessing comparison. Models and the runner consume that shared data contract. Effort estimates remain approximate.

## Supplied data

Paths below are relative to `ECS170_Fall_2026_Source_Code_Template/`.

| Task | Files | Verified contents |
|---|---|---|
| Classification | `data/stage_4_data/text_classification/{train,test}/{pos,neg}/*.txt` | 25,000 training reviews and 25,000 test reviews; each partition has 12,500 positive and 12,500 negative reviews |
| Generation | `data/stage_4_data/text_generation/data` | A CSV file despite having no extension; columns `ID` and `Joke`, with 1,622 nonempty jokes |

The classification target is sentiment, not the numerical rating in each filename. The generation README requires examples using three starting words from supplied jokes and optionally suggests user-chosen prompts. This team plan includes a few new prompts as an additional qualitative comparison.

### Download and extract the shared data

The review files are distributed as `data/stage_4_data/text_classification/reviews.zip`, tracked with Git LFS, to avoid 50,000 separate LFS objects. The joke CSV `data/stage_4_data/text_generation/data` is also tracked with LFS. READMEs remain ordinary Git files.

After pulling `main`, run these commands in PowerShell from the repository root:

```powershell
git lfs pull
Expand-Archive -LiteralPath 'ECS170_Fall_2026_Source_Code_Template/data/stage_4_data/text_classification/reviews.zip' -DestinationPath 'ECS170_Fall_2026_Source_Code_Template/data/stage_4_data/text_classification' -Force
```

This restores the supplied `train/` and `test/` directories at the paths listed above. Extracted reviews and macOS `.DS_Store` metadata are ignored by Git; the ZIP is the shared copy of the reviews.

## Workload summary

| Member | Main ownership | Experiments and report | Approximate effort |
|---|---|---|---:|
| Daniel Park | Both loaders, shared split IDs, tokenization, vocabulary, padding/batching | Complete data verification and data-only review-length comparison; dataset/preprocessing text | 10 points |
| Person 2 - Amar Thota | Classification models, classification evaluation, reusable convergence plotting | RNN/LSTM/GRU review-classification runs, test results, error analysis | 10 points |
| Person 3 - Anant Madhok | Generation models and decoding | RNN/LSTM/GRU joke-generation runs, examples, generation analysis | 10 points |
| Person 4 - Pranesh Kumar | Training loops, settings, runner, result saving | Integration checks and report assembly from finished contributions | 10 points |

Model owners run their own full experiments through Person 4's pipeline. Everyone delivers finished figures/tables and paste-ready report text for their own work. Person 4 assembles those contributions rather than generating everyone's presentation material. Points describe approximate hands-on effort; unattended training time is a separate scheduling consideration.

## Daniel - Data preparation and preprocessing

**Files:** `Dataset_Loader_Reviews.py`, `Dataset_Loader_Jokes.py`, `Data_Splits.py`, `Text_Preprocessing.py`, `script/stage_4_script/inspect_data.py`, and focused tests.

1. Implement template-compatible loaders for the supplied review directories and joke CSV. Keep supplied review training and test partitions separate. Read joke text with a CSV reader so commas and quoted text are handled correctly; exclude the ID column from model inputs.
2. Own the shared seed-2 training/validation splits described below and persist source IDs. Keep identical reviews and identical jokes in the same partition; stratify review groups by sentiment. Implement a simple tokenizer and separate training-only vocabularies, preserving negation. Save the actual split vocabularies for teammates to consume.
3. Supply `<PAD>` and `<UNK>` tokens, with `<PAD>` ID 0. Include an `<EOS>` end-of-joke token for generation. Turn tokenized text into integer sequences and map unseen words to `<UNK>`.
4. Implement the agreed batch preparation below. Pad within each batch; return actual sequence lengths. Make review truncation explicit and record the chosen maximum length. For generation, shift the same joke by one token to create next-word inputs and targets; never join unrelated jokes into one sequence.
5. Record class counts, token-length distributions, fitted vocabulary sizes, training coverage, validation unknown-token rates, and actual encoded batches. Save the split IDs, vocabularies, measured JSON summary, and figures under `result/stage_4_result/data_inspection/`.
6. Compare 128 review tokens, 256 review tokens, and no truncation using the same saved split and vocabulary. Report truncated-review counts, retained-token fractions, and unknown-token rates. Save the JSON results and figure under `result/stage_4_result/review_length_comparison/`. This measures input retention and needs no model or trainer. Any later accuracy comparison is optional classification work.
7. Verify loading, labels, vocabulary isolation, unknown words, padding, and shifted generation targets. Supply dataset/preprocessing report text and the concrete handoff needed by the other members.

**Effort:** loading/tokenization/vocabulary 4; splits/batching/data summaries 3; data-only comparison 1; verification/report 2.

## Person 2 - Classification and evaluation

**Proposed files:** `Method_RNN_Classification.py`, `Evaluate_Classification.py`, `Convergence_Plots.py`, and focused tests.

1. Implement a review classifier with an embedding layer, a selectable RNN/LSTM/GRU recurrent layer, and a final linear layer producing two raw class scores. Inherit the professor's method base following the existing project structure.
2. Use each review's actual final token representation, not the final padded position. Confirm that adding extra right-padding does not change the prediction in evaluation mode.
3. Reuse suitable existing accuracy, precision, recall, F1, confusion-matrix, and plotting code. Explicitly label the averaging method; use positive-class precision/recall/F1 with positive review label 1 for this binary task. Supply a small reusable convergence-plot function that both task owners can use. Keep metrics and plotting separate from the trainer.
4. Run the three required classification models with the same preprocessing, embedding dimension, hidden dimension, number of layers, split, seed, and epoch budget. Explain that parameter counts still differ between recurrent cell types. Save learning curves and validation results through the shared pipeline.
5. Freeze the three classification configurations using validation results. Retrain each cell type on all 25,000 supplied training reviews and evaluate each on the 25,000 supplied test reviews, covering the instruction to repeat classification evaluation with LSTM and GRU. Choose the recommended configuration using validation results, not the final test rankings.
6. Analyze a small set of errors and provide the classification architecture diagram, measured results, and discussion. Use the saved split/vocabulary/batch handoff. An additional review-length versus accuracy comparison is optional.

**Effort:** classifier and three-model experiments 5; metrics/plots/error analysis 3; verification/report 2.

## Person 3 - Generation and decoding

**Proposed files:** `Method_RNN_Generation.py`, `Generate_Text.py`, and focused tests.

1. Implement a word-level next-token model with an embedding layer, selectable RNN/LSTM/GRU layer, and a linear layer producing one raw score per vocabulary token at every sequence position. Follow the professor's method base.
2. Connect to the shared shifted input/target batches. Use cross-entropy loss while ignoring padded targets. Ensure the model cannot access future words: use a unidirectional recurrent layer.
3. Implement generation from exactly three starting words. Use the same tokenizer/vocabulary as training, stop at `<EOS>` or a fixed maximum generated length, and prevent `<PAD>` from being emitted. Start with greedy decoding; temperature sampling is optional, not a required extra experiment.
4. Run RNN, LSTM, and GRU using the same generation split, seed, preprocessing, embedding/hidden dimensions, and epoch budget. Record training and validation next-token loss; perplexity may be reported as `exp(mean token loss)` if useful, but is optional.
5. Compare all three models using the same fixed list of three-word prompts. Include prompts drawn from supplied jokes and a few new prompts. Save the matching original joke beside dataset-derived prompts; if a prefix matches several jokes, record which source was used.
6. Distinguish reproducing a training joke from generating a plausible new completion. Discuss coherence, repetition, and whether the completion resembles the original. Do not present memorized training examples as proof of generalization. Select settings using validation loss and validation examples, then present clearly labeled final examples.
7. Verify output shapes, padded-target handling, prefix handling, stopping, and a small training/generation run. Supply the generation architecture diagram, example table, curves, and report discussion.

**Effort:** generation model and decoding 4; three-model experiments/examples/analysis 4; verification/report 2. No additional shared infrastructure is assigned because generation already includes decoding and qualitative analysis.

## Person 4 - Training pipeline, integration, and report assembly

**Proposed files:** `Training.py`, `Setting_Stage4.py`, `script/stage_4_script/run_experiments.py`, and integration tests. Reuse existing result-saving helpers where they fit; do not modify the working Stage 2/3 implementations. Person 2 owns reusable plotting.

1. Provide straightforward mini-batch training for the two tasks. Share small common helpers where useful, but keep classification and next-token losses explicit rather than building a general-purpose training framework. Use the agreed model/batch interfaces below.
2. Track per-epoch train/validation loss and classification accuracy. Weight classification loss by examples and generation loss by non-padding target tokens. Use evaluation mode and disable gradients for validation, testing, and generation.
3. Provide settings and a runner selecting task, recurrent cell, seeds, epoch budget, batch size, embedding/hidden dimensions, review length, and output directory. CPU must work; available GPU support is optional.
4. Consume Daniel's saved split IDs and vocabularies in the runner. Do not generate competing splits or refit validation-run vocabularies on different records. Keep supplied review test data out of vocabulary fitting and selection. Final full-training fits rebuild their vocabulary only from the complete supplied training partition, as described below.
5. Save each run's configuration, vocabulary, history, metrics, model weights, and split identity in a distinct directory. Saving the vocabulary with weights is essential to reproduce word IDs and generation. Model owners supply classification predictions or generated examples for saving.
6. Integrate short runs for both tasks and all three cell types before full training. Ensure changing a task/cell actually constructs the requested model, and final runs do not reuse weights from prior experiments.
7. Compile everyone's finished figures, tables, report text, diagrams, and examples into the supplied template; verify the five-page maximum, template formatting, and readable figures. Each contributor prepares and checks their own presentation material; task owners use Person 2's plotting function for their curves.

**Effort:** task training loops/history 4; runner/settings/result saving 3; integration/report assembly 3.

## Shared interfaces

These are team implementation choices, not additional professor requirements. Models and the runner use the following data contract.

| Item | Contract |
|---|---|
| Raw loader output | Reviews: text strings and labels, negative = 0 and positive = 1. Jokes: text strings plus source IDs for comparison. |
| Vocabulary | Separate per task; string-to-ID and ID-to-string mappings; padding ID 0; unseen words map to `<UNK>`. Fit after splitting, using training text only. |
| Saved data handoff | `result/stage_4_result/data_inspection/splits.json` contains seed and train/val source IDs for reviews/jokes; `review_vocabulary.json` and `joke_vocabulary.json` contain `token_to_id` and `id_to_token` mappings. |
| Classification batch | `input_ids`: int64 `(B, T)`; `lengths`: int64 `(B,)`, excluding padding; `labels`: int64 `(B,)`. |
| Generation batch | `input_ids`, `target_ids`: int64 `(B, T)`; `lengths`: int64 `(B,)`. Inputs/targets are shifted by one token; the last real target is `<EOS>`. |
| Classification forward | `forward(input_ids, lengths)` returns raw scores `(B, 2)`. The model handles sequence lengths internally. |
| Generation forward | `forward(input_ids, lengths)` returns raw scores `(B, T, V)`, where `V` is vocabulary size. No softmax before cross-entropy. |
| Generation entry point | `generate(model, prompt, vocabulary, max_new_tokens)` returns a readable completion. It may rerun the growing prefix for simplicity; hidden-state caching is optional. |
| History | Both tasks: `train_loss`, `val_loss`; classification additionally: `train_acc`, `val_acc`. A final fit without validation has empty validation lists. |
| Results | Record task, cell type, split, seed, actual settings, vocabulary size, parameter count, epoch budget, and evaluated partition. |

Use the professor's base classes and existing directory conventions. Task owners define layers/forward behavior; Person 4 owns training loops; Daniel owns token IDs and batches. Do not create separate competing trainers in each model file.

## Data preparation and length-comparison handoff

Implementation branch: `daniel-stage4-data-preprocessing`. Run these commands from `ECS170_Fall_2026_Source_Code_Template/` after integrating its implementation:

```powershell
python script/stage_4_script/inspect_data.py
python script/stage_4_script/compare_review_lengths.py
```

The first command saves shared split IDs, training vocabularies, data checks, batch examples, and descriptive figures. The second consumes those saved artifacts without modifying them and compares 128 tokens, 256 tokens, and full reviews. Its numeric results and figure are separate from the preparation outputs.

Keep 256 tokens as the common baseline for required recurrent-cell experiments. An uncapped sequence uses `encode(..., max_length=None)` and the existing batch helper; it retains the same training vocabulary. The data-only comparison reports no classification accuracy or training-time measurements. Optional model experiments can evaluate the accuracy/resource tradeoff separately.

## Bounded experiment and evaluation plan

1. **Training budget:** six validation-comparison fits (RNN/LSTM/GRU for each task), followed by three final full-training classification fits: **nine full fits**, plus short pilots. Daniel's data-only preprocessing comparison adds no training runs. Optional final generation retraining or extra tuning adds more. Keep settings fixed within each task and measure pilot runtime before choosing a modest epoch budget.
2. **Classification validation:** group identical reviews, stratify groups by sentiment, and hold out approximately 10% with seed 2. The saved split contains 22,500 training and 2,500 validation records. Preserve the supplied 25,000 test reviews for final evaluation; the saved IDs are authoritative for membership.
3. **Generation validation:** split whole jokes approximately 90/10 with seed 2 before vocabulary construction or creating shifted sequences. Keep identical full joke texts in the same partition if duplicates exist. Save the actual counts and split IDs. Do not split tokens from one joke between training and validation.
4. **Data-only comparison:** compare retention/truncation/unknown-token rates for 128 tokens, 256 tokens, and no truncation using the same saved splits and vocabulary. Classification-accuracy sweeps are optional; do not promise exhaustive tuning or a second loss function.
5. **Final classification:** rebuild the vocabulary once from the complete supplied training partition, using the same tokenizer and vocabulary cap. Use that same final vocabulary across fresh RNN/LSTM/GRU models, train each frozen configuration on the full supplied training partition, and evaluate the supplied test partition. Never fit the vocabulary on supplied test text.
6. **Final generation:** report held-out validation measurements from the comparison runs. A selected model may then be retrained on all supplied jokes for final example generation, with its vocabulary fitted only to that final training corpus. Clearly distinguish validation measurements from full-corpus training examples. There is no supplied joke test partition.
7. **GloVe:** optional follow-up classification comparison owned by the classification owner. Start with learned embeddings for the required baseline.

## Development order and handoffs

1. Prepare both loaders, shared split IDs, training-only vocabularies, encoded batch checks, data figures, and the 128/256/uncapped data comparison.
2. Use the saved batches/splits/vocabularies for model development and integrate the shared training runner. Fixture batches support initial model and trainer checks.
3. Verify all three recurrent cells through the shared task pipelines and run the six validation comparisons.
4. Task owners select settings from validation results, run the three final classification fits, and produce final results/examples. Everyone delivers finished figures/tables and paste-ready report text for their contribution. Person 4 assembles the report.
5. Everyone reviews the final report and their own code/results. Use separate branches and coordinate edits to shared files. Do not assume the Stage 3 deadline extension applies to Stage 4; confirm the current deadline separately.

## Completion checklist

- [ ] Both supplied datasets load correctly; preprocessing and vocabulary construction are documented (4-1).
- [ ] RNN/LSTM/GRU classification models train and produce learning curves and supplied-test results (4-2, 4-3, 4-5).
- [ ] RNN/LSTM/GRU generation models train and complete three-word prompts; examples include comparison with original jokes (4-4, 4-5).
- [ ] Comparisons use consistent splits/settings and report actual results without selecting configurations using test performance.
- [ ] Report includes architecture diagrams, preprocessing, experimental settings, learning curves, classification metrics, generation examples, and conclusions within five pages (4-6).
- [ ] CPU execution works; GPU use, GloVe, sampling, and additional tuning remain optional (4-7).

## Sources and scope

Checked the project description PDF, pages 4-5 (requirements 4-1 through 4-7), the classification README, the generation ReadMe.docx, review-file counts, and CSV joke records. Dataset counts above were verified locally. Allocation, interfaces, validation splits, experiment budgets, and preprocessing choices are team implementation decisions.
