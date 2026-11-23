import tempfile
import unittest
from pathlib import Path

import torch
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from local_code.stage_4_code.Evaluate_Classification import classification_metrics
from local_code.stage_4_code.Convergence_Plots import save_convergence_plots


class ClassificationTests(unittest.TestCase):
    def test_cells_padding_and_gradients(self):
        ids = torch.tensor([[2, 3, 4], [5, 0, 0], [6, 7, 0]])
        lengths = torch.tensor([3, 1, 2])
        for cell in ('rnn', 'lstm', 'gru'):
            with self.subTest(cell=cell):
                model = Method_RNN_Classification(10, cell=cell, embed_dim=4, hidden_dim=6, num_layers=2)
                model.eval()
                scores = model(ids, lengths)
                self.assertEqual(scores.shape, (3, 2))
                padded = torch.nn.functional.pad(ids, (0, 4))
                torch.testing.assert_close(scores, model(padded, lengths))
                for i, length in enumerate(lengths):
                    torch.testing.assert_close(scores[i], model(ids[i:i+1, :length], lengths[i:i+1])[0])
                torch.nn.functional.cross_entropy(scores, torch.tensor([0, 1, 0])).backward()
                self.assertIsNotNone(model.embedding.weight.grad)
                self.assertGreater(model.count_params(), 0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Method_RNN_Classification(10, cell='bad')
        model = Method_RNN_Classification(10)
        with self.assertRaises(ValueError):
            model(torch.ones((1, 3), dtype=torch.long), torch.tensor([0]))

    def test_metrics(self):
        result = classification_metrics([0, 0, 1, 1, 1], [0, 1, 0, 1, 1])
        self.assertEqual(result['confusion_matrix'], [[1, 1], [1, 2]])
        self.assertAlmostEqual(result['accuracy'], 0.6)
        self.assertAlmostEqual(result['precision_positive'], 2 / 3)
        self.assertAlmostEqual(result['recall_positive'], 2 / 3)
        self.assertAlmostEqual(result['f1_positive'], 2 / 3)
        self.assertEqual(classification_metrics([0], [0])['f1_positive'], 0)
        with self.assertRaises(ValueError):
            classification_metrics([], [])
        with self.assertRaises(ValueError):
            classification_metrics([0], [2])

    def test_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i, history in enumerate(({'train_loss': [2, 1], 'val_loss': []},
                                         {'train_loss': [1], 'val_loss': [2], 'train_acc': [.5], 'val_acc': [.4]})):
                path = save_convergence_plots(history, Path(tmp) / str(i) / 'plot.png')
                self.assertGreater(path.stat().st_size, 0)
        with self.assertRaises(ValueError):
            save_convergence_plots({}, 'unused.png')


if __name__ == '__main__':
    unittest.main()
