"""Variable-length sentiment classifier shared by the Stage 4 experiments."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from local_code.base_class.method import method

CELLS = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}


class Method_RNN_Classification(method, nn.Module):
    """Embedding -> unidirectional RNN/LSTM/GRU -> two raw class scores."""

    def __init__(self, vocab_size, mName='review_classification',
                 mDescription='Recurrent sentiment classifier', *, cell='rnn',
                 embed_dim=128, hidden_dim=256, num_layers=1, seed=2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        cell = cell.lower()
        if cell not in CELLS:
            raise ValueError(f'cell must be one of {sorted(CELLS)}, got {cell!r}')
        torch.manual_seed(seed)
        self.cell, self.seed = cell, seed
        self.vocab_size, self.embed_dim = vocab_size, embed_dim
        self.hidden_dim, self.num_layers = hidden_dim, num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = CELLS[cell](embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids, lengths):
        """Return (batch, 2) logits, using each review's last non-padding state."""
        if input_ids.ndim != 2 or lengths.ndim != 1 or len(lengths) != input_ids.shape[0]:
            raise ValueError('expected input_ids (B, T) and lengths (B,)')
        if input_ids.shape[0] == 0 or torch.any(lengths <= 0) or torch.any(lengths > input_ids.shape[1]):
            raise ValueError('each sequence must have a length between 1 and T')
        packed = pack_padded_sequence(self.embedding(input_ids), lengths.detach().cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        if self.cell == 'lstm':
            hidden = hidden[0]
        return self.output(hidden[-1])

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def run(self, *args, **kwargs):
        raise NotImplementedError('train this model through the shared Stage 4 training pipeline')
