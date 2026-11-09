"""MNIST CNN checks: python -m unittest discover -s tests -p test_stage3_mnist.py -v."""

import unittest

import numpy as np
import torch

from local_code.base_class.method import method
from local_code.stage_3_code.Method_CNN_MNIST import MNISTCNN


class MNISTTests(unittest.TestCase):
    def test_dimensions_scores_and_gradients(self):
        # Check the baseline and two variants with single-image and multi-image batches.
        for kernel_size, pooling in ((3, 'max'), (5, 'max'), (3, 'average')):
            for batch_size in (1, 4):
                with self.subTest(kernel=kernel_size, pooling=pooling, batch=batch_size):
                    torch.manual_seed(2)
                    model = MNISTCNN(kernel_size=kernel_size, pooling=pooling)
                    images = torch.rand(batch_size, 1, 28, 28)
                    features = model.features(images)
                    self.assertEqual(tuple(features.shape), (batch_size, 32, 7, 7))
                    scores = model(images)
                    self.assertEqual(tuple(scores.shape), (batch_size, 10))
                    self.assertTrue(torch.isfinite(scores).all().item())

                    # Backpropagation must reach every trainable parameter with finite gradients.
                    labels = torch.arange(batch_size) % 10
                    loss = torch.nn.CrossEntropyLoss()(scores, labels)
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                    for parameter in model.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_invalid_options(self):
        # Unsupported options should fail instead of silently choosing another configuration.
        with self.assertRaises(ValueError):
            MNISTCNN(kernel_size=4)
        with self.assertRaises(ValueError):
            MNISTCNN(pooling='median')

    def test_shared_trainer_integration_for_validation_and_final_modes(self):
        # Separate synthetic images encode two classes using left/right bright strips.
        rng = np.random.default_rng(3)

        def make_partition(count):
            labels = np.arange(count, dtype=np.int64) % 2
            images = rng.random((count, 1, 28, 28), dtype=np.float32) * 0.02
            images[labels == 0, :, :, :10] += 0.9
            images[labels == 1, :, :, 18:] += 0.9
            return {'X': images, 'y': labels}

        training = make_partition(23)
        evaluation = make_partition(7)
        for split in ('val', 'test'):
            with self.subTest(split=split):
                model = MNISTCNN('synthetic', 'Integration check', seed=3)
                self.assertIsInstance(model, method)
                self.assertEqual(model.method_name, 'synthetic')
                self.assertEqual(model.method_description, 'Integration check')
                model.max_epoch = 3
                model.batch_size = 5
                model.learning_rate = 0.005
                model.verbose = False
                model.data = {'train': training, split: evaluation}
                original_weight = model.features[0].weight.detach().clone()

                result = model.run()
                self.assertFalse(torch.equal(model.features[0].weight, original_weight))
                self.assertEqual(tuple(result['pred_y'].shape), (7,))
                np.testing.assert_array_equal(result['true_y'], evaluation['y'])
                self.assertGreater((result['pred_y'].numpy() == evaluation['y']).mean(), 0.9)
                self.assertEqual(len(model.history['train_loss']), 3)
                self.assertEqual(len(model.history['val_loss']), 3 if split == 'val' else 0)
                self.assertEqual(len(model.history['val_acc']), 3 if split == 'val' else 0)


if __name__ == '__main__':
    unittest.main()
