"""Word-level next-token model for finishing jokes (RNN, LSTM or GRU)."""

import torch
from torch import nn

from local_code.base_class.method import method


CELLS = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}


class Method_RNN_Generation(method, nn.Module):
    """Embedding -> one-directional recurrent layer -> a score for every vocab word."""

    def __init__(self, vocab_size, mName='joke_generation', mDescription='RNN joke generator',
                 *, cell='rnn', embed_dim=128, hidden_dim=256, num_layers=1, seed=2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        cell = cell.lower()
        if cell not in CELLS:
            raise ValueError(f"cell must be one of {sorted(CELLS)}, got {cell!r}")

        # seed before the layers get built so the same seed always gives the same starting weights
        self.seed = seed
        torch.manual_seed(seed)
        self.cell = cell
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # padding_idx=0 keeps the <PAD> embedding at zero so it never learns anything
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # has to stay one-directional, otherwise the model could peek at the word it's predicting
        self.rnn = CELLS[cell](embed_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=False)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, lengths=None):
        """Return raw scores (B, T, V); no softmax bc cross-entropy does that itself.

        lengths isn't actually needed here: padding is on the right and the layer only
        reads left to right, so the real positions never see the padding. Scores at padded
        positions are junk, which is fine bc the loss skips targets that are 0.
        """
        hidden_states, _ = self.rnn(self.embedding(input_ids))
        return self.output(hidden_states)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def run(self, *args, **kwargs):
        # training lives in the shared Stage 4 trainer, this class just defines the layers
        raise NotImplementedError('train this model through the shared Stage 4 training pipeline')
