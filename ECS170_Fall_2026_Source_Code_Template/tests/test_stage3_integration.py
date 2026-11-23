"""Small on-disk integration test of the complete Stage 3 experiment pipeline."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from local_code.stage_3_code.Dataset_Loader_MNIST import Dataset_Loader_MNIST
from local_code.stage_3_code.Method_CNN_MNIST import MNISTCNN
from local_code.stage_3_code.experiments import run_experiment, split_training


class ExperimentIntegrationTests(unittest.TestCase):
    def test_validation_and_test_artifacts(self):
        # Distinct test labels make accidental use of the official test split visible.
        train = [{'image': np.full((28, 28), i * 10, dtype=np.uint8), 'label': i % 2}
                 for i in range(20)]
        test = [{'image': np.full((28, 28), i * 20, dtype=np.uint8), 'label': 9}
                for i in range(3)]
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / Dataset_Loader_MNIST.file_name).open('wb') as stream:
                pickle.dump({'train': train, 'test': test}, stream)
            training = Dataset_Loader_MNIST(split='train', data_dir=root).load()
            official_test = Dataset_Loader_MNIST(split='test', data_dir=root).load()
            train_subset, validation = split_training(training, seed=7)
            for mode, expected_train, evaluation in (
                    ('validation', train_subset, validation),
                    ('test', training, official_test)):
                with self.subTest(mode=mode):
                    output = root / mode
                    metrics = run_experiment('mnist', mode, 'baseline', data_dir=root,
                                             output=output, seed=3, split_seed=7,
                                             epochs=1, batch_size=20, device='cpu')
                    self.assertEqual({p.name for p in output.iterdir()},
                                     {'config.json', 'history.json', 'metrics.json',
                                      'predictions.npz', 'learning_curves.png', 'model.pt'})
                    config = json.loads((output / 'config.json').read_text())
                    history = json.loads((output / 'history.json').read_text())
                    self.assertEqual(config['mode'], mode)
                    self.assertEqual(config['train_count'], len(expected_train['y']))
                    self.assertEqual(config['evaluation_count'], len(evaluation['y']))
                    self.assertEqual(config['split_seed'], 7)
                    self.assertEqual(len(history['train_loss']), 1)
                    self.assertEqual(len(history['train_acc']), 1)
                    for key in ('val_loss', 'val_acc'):
                        self.assertEqual(len(history[key]), int(mode == 'validation'))
                    self.assertEqual(metrics, json.loads((output / 'metrics.json').read_text()))
                    with np.load(output / 'predictions.npz') as saved:
                        np.testing.assert_array_equal(saved['labels'], evaluation['y'])
                        predictions = saved['predictions'].copy()
                    self.assertEqual(metrics['total'], len(evaluation['y']))
                    self.assertEqual(metrics['correct'], int((predictions == evaluation['y']).sum()))
                    self.assertEqual(sum(map(sum, metrics['confusion_matrix'])), metrics['total'])
                    model = MNISTCNN(seed=3)
                    model.load_state_dict(torch.load(output / 'model.pt', map_location='cpu', weights_only=True))
                    np.testing.assert_array_equal(model.test(evaluation['X']).numpy(), predictions)
                    self.assertGreater((output / 'learning_curves.png').stat().st_size, 0)
                    with self.assertRaises(FileExistsError):
                        run_experiment('mnist', mode, 'baseline', data_dir=root, output=output,
                                       epochs=1, batch_size=20)


if __name__ == '__main__':
    unittest.main()
