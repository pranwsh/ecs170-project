# Five-epoch classification validation status

The comparable CPU experiments use seed 2, batch size 64, embedding 32, hidden 64, one layer, maximum review length 256, Adam 0.001 and the existing training-only vocabulary and split. Runs save the final epoch's model, not the best-validation checkpoint.

Completed: RNN validation accuracy by epoch 0.5536, 0.5264, 0.6308, 0.5568, 0.5484; validation loss 0.68408, 0.68911, 0.64383, 0.68282, 0.68510. Its five-epoch final result is not converged and is worse than its best recorded epoch. Artifacts: `result/stage_4_result/validation_5epoch_rnn/`.

LSTM and GRU five-epoch validations were started sequentially on the local CPU; completion and metrics are not yet established here. No five-epoch final supplied-test fits have been run. Previously published one-epoch test fits remain descriptive only; original test freeze timing cannot be independently established. Do not treat test scores as validation selection evidence.

To reproduce a run from the template root within the Nix environment (change cell and output):

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python script/stage_4_script/run_experiments.py --task classification --cell rnn --epochs 5 --embed-dim 32 --hidden-dim 64 --output result/stage_4_result/validation_5epoch_rnn
```
