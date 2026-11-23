# Stage 4 model architectures

Classification (RNN, LSTM, and GRU share the pipeline):

```mermaid
flowchart LR
 A[Review text] --> B[Tokenize and lowercase] --> C[Training-vocabulary IDs, truncate to 256] --> D[Right pad and record lengths] --> E[Learned embedding: 32 dimensions] --> F[Packed unidirectional RNN / LSTM / GRU: 64 hidden units] --> G[Final non-padding hidden state] --> H[Linear: 64 to 2 logits] --> I[Predicted sentiment]
```

The LSTM's final hidden state, not its cell state, feeds the classifier. Training uses cross-entropy and Adam. The dimensions shown describe the comparable classification pilots, not the default settings for every experiment. Lengths include at least one token; padding is excluded by packing.

Generation architecture:

```mermaid
flowchart LR
 A[Joke tokens plus EOS] --> B[Training-vocabulary IDs] --> C[Shift input and target by one token] --> D[Learned embedding] --> E[RNN / LSTM / GRU] --> F[Linear projection to vocabulary logits at each timestep] --> G[Next-token prediction]
```

Generation uses teacher forcing during training; padding targets are excluded from the loss. See `Method_RNN_Generation.py` and `Generate_Text.py` for sampling details.
