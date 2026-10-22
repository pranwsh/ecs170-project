"""MLP tests w/ fake data: python -m unittest discover -s tests -v."""

import unittest

import numpy as np
import torch

from local_code.stage_2_code.Method_MLP import Method_MLP


class MLPTests(unittest.TestCase):
    def test_learns_separable_data_and_records_history(self):
        # each digit lights up its own block of pixels, so a working MLP should hit ~100%
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        y = np.repeat(np.arange(10), 30)
        X = rng.random((300, 784), dtype=np.float32) * 0.1
        for label in range(10):
            X[y == label, label * 78:(label + 1) * 78] += 1.0

        model = Method_MLP('mlp', '', hidden_dims=(32,))
        model.max_epoch = 15
        model.batch_size = 32
        model.data = {'train': {'X': X, 'y': y}, 'val': {'X': X, 'y': y}, 'test': {'X': X, 'y': y}}
        result = model.run()

        self.assertEqual(result['pred_y'].shape, (300,))
        self.assertGreater((result['pred_y'].numpy() == y).mean(), 0.95)
        self.assertEqual(len(model.history['train_loss']), 15)
        self.assertEqual(len(model.history['val_acc']), 15)
        self.assertLess(model.history['train_loss'][-1], model.history['train_loss'][0])
        # the train() rename keeps nn.Module.eval() from breaking
        model.eval()


if __name__ == '__main__':
    unittest.main()
