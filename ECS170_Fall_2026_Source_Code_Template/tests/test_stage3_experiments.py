"""Non-training checks for models and experiment utilities."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from local_code.stage_3_code.Method_CNN_ORL import ORLCNN
from local_code.stage_3_code.Method_CNN_CIFAR import CIFARCNN
from local_code.stage_3_code.experiments import split_training, classification_metrics, save_results


class ExperimentTests(unittest.TestCase):
    def test_model_shapes(self):
        for cls, shape, count in ((ORLCNN, (2, 1, 112, 92), 40),
                                  (CIFARCNN, (2, 3, 32, 32), 10)):
            for kwargs in ({}, {'kernel_size': 5}, {'pooling': 'average'}):
                with self.subTest(model=cls.__name__, options=kwargs):
                    model = cls(**kwargs)
                    with torch.no_grad():
                        self.assertEqual(tuple(model(torch.zeros(shape)).shape), (2, count))
            with self.assertRaises(ValueError):
                cls(pooling='bad')

    def test_stratified_split(self):
        data = {'X': np.arange(80)[:, None], 'y': np.repeat(np.arange(4), 20)}
        train, val = split_training(data, 2)
        self.assertFalse(set(train['X'].ravel()) & set(val['X'].ravel()))
        self.assertEqual(len(train['y']) + len(val['y']), 80)
        self.assertEqual(set(val['y']), {0, 1, 2, 3})
        np.testing.assert_array_equal(val['X'], split_training(data, 2)[1]['X'])

    def test_orl_holdout_has_at_least_one_per_class(self):
        data = {'X': np.arange(360)[:, None], 'y': np.repeat(np.arange(40), 9)}
        train, val = split_training(data, 2)
        self.assertEqual(len(val['y']), 40)
        self.assertEqual(len(np.unique(val['y'])), 40)
        self.assertEqual(len(train['y']), 320)

    def test_metrics_and_artifacts(self):
        metrics = classification_metrics(np.array([0, 1, 1]), np.array([0, 0, 1]), 2)
        self.assertEqual(metrics['correct'], 2)
        self.assertEqual(metrics['confusion_matrix'], [[1, 0], [1, 1]])
        history = {'train_loss': [1.0], 'train_acc': [.5], 'val_loss': [], 'val_acc': []}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'run'
            save_results(output, {'mode': 'test'}, history, metrics, np.array([0]), np.array([0]))
            self.assertEqual(json.loads((output / 'metrics.json').read_text())['total'], 3)
            self.assertTrue((output / 'learning_curves.png').exists())
            with self.assertRaises(FileExistsError):
                save_results(output, {}, history, metrics, [], [])


if __name__ == '__main__':
    unittest.main()
