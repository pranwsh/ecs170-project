"""Shared CNN trainer checks using small synthetic image arrays."""

import unittest

import numpy as np
import torch
from torch import nn

from local_code.base_class.method import method
from local_code.stage_2_code.Loss_Functions import SoftmaxMSE
from local_code.stage_3_code.Method_CNN import Method_CNN


class TinyClassifier(Method_CNN):
    def __init__(self, **kwargs):
        super().__init__('tiny', 'Synthetic classifier', **kwargs)
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
        self.batch_calls = []
        self.verbose = False

    def forward(self, images):
        # Record batch size and execution mode to verify bounded inference.
        self.batch_calls.append((len(images), self.training, torch.is_grad_enabled()))
        return self.layers(images)


class TrainerTests(unittest.TestCase):
    def setUp(self):
        # Each label controls image brightness; separate examples have independent noise.
        rng = np.random.default_rng(2)
        self.labels = np.tile(np.arange(2), 20)
        self.images = (self.labels[:, None, None, None]
                       + rng.random((40, 1, 2, 2)) * 0.05).astype(np.float32)
        self.val_labels = np.tile(np.arange(2), 5)
        self.val_images = (self.val_labels[:, None, None, None]
                           + rng.random((10, 1, 2, 2)) * 0.05).astype(np.float32)

    def test_partial_batch_metrics_and_prediction_order(self):
        # Compare batched measurements with a full calculation on five known examples.
        for loss_function in (nn.CrossEntropyLoss(), SoftmaxMSE(num_classes=2)):
            with self.subTest(loss=type(loss_function).__name__):
                model = TinyClassifier(loss_function=loss_function)
                self.assertIsInstance(model, method)
                self.assertIsInstance(model, nn.Module)
                model.batch_size = 2
                images, labels = self.images[:5], self.labels[:5]
                with torch.no_grad():
                    scores = model.layers(torch.from_numpy(images))
                    expected_loss = loss_function(scores, torch.as_tensor(labels)).item()
                    expected_acc = (scores.argmax(1) == torch.as_tensor(labels)).float().mean().item()
                weights = [parameter.detach().clone() for parameter in model.parameters()]

                loss, accuracy = model._loss_and_acc(images, labels)
                predictions = model.test(images)
                self.assertAlmostEqual(loss, expected_loss, places=6)
                self.assertAlmostEqual(accuracy, expected_acc, places=6)
                torch.testing.assert_close(predictions, scores.argmax(1))
                self.assertEqual(predictions.device.type, 'cpu')
                self.assertEqual([call[0] for call in model.batch_calls], [2, 2, 1, 2, 2, 1])
                self.assertTrue(all(not training and not gradients
                                    for _, training, gradients in model.batch_calls))
                for parameter, original in zip(model.parameters(), weights):
                    torch.testing.assert_close(parameter, original)

    def test_training_learns_and_records_post_epoch_history(self):
        model = TinyClassifier()
        model.max_epoch = 12
        model.batch_size = 7
        model.learning_rate = 0.1
        model.train_model(self.images, self.labels, self.val_images, self.val_labels)
        self.assertTrue(all(len(values) == 12 for values in model.history.values()))
        self.assertLess(model.history['train_loss'][-1], model.history['train_loss'][0])
        self.assertGreater(model.history['val_acc'][-1], 0.95)
        self.assertTrue(any(training and gradients for _, training, gradients in model.batch_calls))
        self.assertTrue(all(count <= 7 for count, _, _ in model.batch_calls))
        val_loss, val_acc = model._loss_and_acc(self.val_images, self.val_labels)
        self.assertAlmostEqual(model.history['val_loss'][-1], val_loss)
        self.assertEqual(model.history['val_acc'][-1], val_acc)

    def test_run_selects_the_requested_partition(self):
        for split in ('val', 'test'):
            with self.subTest(split=split):
                model = TinyClassifier()
                model.max_epoch = 1
                model.batch_size = 3
                model.data = {'train': {'X': self.images[:9], 'y': self.labels[:9]},
                              split: {'X': self.val_images[:5], 'y': self.val_labels[:5]}}
                result = model.run()
                self.assertEqual(tuple(result['pred_y'].shape), (5,))
                np.testing.assert_array_equal(result['true_y'], self.val_labels[:5])
                expected_val_epochs = 1 if split == 'val' else 0
                self.assertEqual(len(model.history['val_loss']), expected_val_epochs)
                self.assertEqual(len(model.history['val_acc']), expected_val_epochs)
                self.assertEqual(len(model.history['train_acc']), 1)

    def test_history_and_optimizer_options_are_per_model(self):
        first, second = TinyClassifier(), TinyClassifier()
        first.optimizer_kwargs['weight_decay'] = 0.1
        self.assertEqual(second.optimizer_kwargs, {})
        first.max_epoch = 1
        first.train_model(self.images[:9], self.labels[:9])
        first.train_model(self.images[:9], self.labels[:9])
        self.assertEqual(len(first.history['train_loss']), 1)
        self.assertEqual(second.history['train_loss'], [])

    def test_seed_reproduces_initialization_and_training(self):
        final_weights = []
        for _ in range(2):
            model = TinyClassifier(seed=4)
            model.max_epoch = 2
            model.batch_size = 7
            model.train_model(self.images, self.labels)
            final_weights.append([parameter.detach().clone() for parameter in model.parameters()])
        for first, second in zip(*final_weights):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_incomplete_inputs_and_ambiguous_run_are_rejected(self):
        model = TinyClassifier()
        with self.assertRaises(ValueError):
            model.train_model(self.images, self.labels, X_val=self.val_images)
        with self.assertRaises(ValueError):
            model.test(self.images[:0])
        with self.assertRaises(ValueError):
            model._loss_and_acc(self.images[:5], self.labels[:4])
        model.batch_size = 0
        with self.assertRaises(ValueError):
            model.test(self.images)
        partition = {'X': self.images, 'y': self.labels}
        for evaluation in ({}, {'val': partition, 'test': partition}):
            model.data = {'train': partition, **evaluation}
            with self.assertRaises(ValueError):
                model.run()


if __name__ == '__main__':
    unittest.main()
