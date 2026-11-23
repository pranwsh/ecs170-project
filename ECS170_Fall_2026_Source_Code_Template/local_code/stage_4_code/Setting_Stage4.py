"""Settings for a single reproducible Stage 4 experiment."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Stage4Settings:
    task: str = 'classification'
    cell: str = 'rnn'
    mode: str = 'validation'
    seed: int = 2
    epochs: int = 5
    batch_size: int = 64
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    review_length: int = 256
    learning_rate: float = 0.001
    device: str = 'cpu'

    def __post_init__(self):
        if self.task not in ('classification', 'generation'):
            raise ValueError('task must be classification or generation')
        if self.cell not in ('rnn', 'lstm', 'gru'):
            raise ValueError('cell must be rnn, lstm or gru')
        if self.mode not in ('validation', 'test'):
            raise ValueError('mode must be validation or test')
        if self.task == 'generation' and self.mode == 'test':
            raise ValueError('there is no supplied joke test partition')
        for key in ('epochs', 'batch_size', 'embed_dim', 'hidden_dim', 'num_layers'):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        if self.review_length is not None and self.review_length < 1:
            raise ValueError('review_length must be positive or None')
        if self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive')

    def to_dict(self):
        return asdict(self)
