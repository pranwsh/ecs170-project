"""Tests for shared Stage 4 training and evaluation."""
import unittest

import torch
from torch import nn
from local_code.stage_4_code.Training import (evaluate_classification, evaluate_generation,
                                             train_classification, train_generation)
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation


class FixedClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0., 0.]))
        self.observed = []

    def forward(self, x, lengths):
        self.observed.append((self.training, torch.is_grad_enabled()))
        return torch.stack((x[:, 0].float(), -x[:, 0].float()), dim=1) + self.bias


class TrainingTests(unittest.TestCase):
    def test_classification_weighting_and_validation(self):
        model = FixedClassifier()
        batches = [dict(input_ids=torch.tensor([[1], [2], [3]]), lengths=torch.ones(3, dtype=torch.long), labels=torch.tensor([0, 0, 1])),
                   dict(input_ids=torch.tensor([[4]]), lengths=torch.ones(1, dtype=torch.long), labels=torch.tensor([1]))]
        result = evaluate_classification(model, batches)
        scores = torch.tensor([[1., -1.], [2., -2.], [3., -3.], [4., -4.]])
        self.assertAlmostEqual(result['loss'], nn.functional.cross_entropy(scores, torch.tensor([0, 0, 1, 1])).item(), places=6)
        self.assertEqual(result['acc'], .5)
        self.assertEqual(model.observed, [(False, False)] * 2)
        history = train_classification(model, batches, batches, torch.optim.SGD(model.parameters(), lr=.01), 2)
        self.assertEqual([len(history[k]) for k in ('train_loss', 'val_loss', 'train_acc', 'val_acc')], [2]*4)
        self.assertIn((True, True), model.observed)
        self.assertFalse(model.training)

    def test_generation_weighting_padding_and_fit(self):
        model = Method_RNN_Generation(8, cell='gru', embed_dim=4, hidden_dim=5, seed=2)
        batches = [dict(input_ids=torch.tensor([[1, 2, 3], [2, 0, 0]]), target_ids=torch.tensor([[2, 3, 4], [3, 0, 0]]), lengths=torch.tensor([3, 1])),
                   dict(input_ids=torch.tensor([[1, 2, 0]]), target_ids=torch.tensor([[2, 3, 0]]), lengths=torch.tensor([2]))]
        model.eval()
        with torch.no_grad():
            losses = [nn.functional.cross_entropy(model(b['input_ids'], b['lengths']).reshape(-1, 8), b['target_ids'].flatten(), ignore_index=0, reduction='sum').item() for b in batches]
        self.assertAlmostEqual(evaluate_generation(model, batches)['loss'], sum(losses)/6, places=6)
        history = train_generation(model, batches, None, torch.optim.Adam(model.parameters(), lr=.01), 2)
        self.assertEqual(len(history['train_loss']), 2)
        self.assertEqual(history['val_loss'], [])

    def test_empty_epoch_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_classification(FixedClassifier(), [])


if __name__ == '__main__':
    unittest.main()
