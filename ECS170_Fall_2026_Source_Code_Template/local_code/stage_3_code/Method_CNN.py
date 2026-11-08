"""Shared mini-batch training and prediction for image classifiers."""

import numpy as np
import torch
from torch import nn

from local_code.base_class.method import method


class Method_CNN(method, nn.Module):
    """Train subclasses with forward(); loss_function must return a scalar batch mean."""

    max_epoch = 20
    batch_size = 128
    learning_rate = 1e-3
    optimizer_class = torch.optim.Adam
    optimizer_kwargs = {}
    device = 'cpu'
    verbose = True

    def __init__(self, mName=None, mDescription=None, *, loss_function=None, seed=2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Seed before subclasses initialize their layers; keep mutable settings per model.
        self.seed = seed
        torch.manual_seed(seed)
        self.optimizer_kwargs = dict(self.optimizer_kwargs)
        self.loss_function = nn.CrossEntropyLoss() if loss_function is None else loss_function
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def _batches(self, X, y=None, *, shuffle=False):
        # Keep full inputs on the CPU and transfer only the current batch to the device.
        images = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        labels = None if y is None else torch.as_tensor(np.asarray(y), dtype=torch.long)
        if self.batch_size <= 0 or len(images) == 0:
            raise ValueError('batch_size and the number of images must be positive')
        if labels is not None and len(labels) != len(images):
            raise ValueError('Each image must have one label')

        order = torch.randperm(len(images)) if shuffle else None
        for start in range(0, len(images), self.batch_size):
            selection = slice(start, start + self.batch_size)
            if order is not None:
                selection = order[selection]
            batch_images = images[selection].to(self.device)
            batch_labels = None if labels is None else labels[selection].to(self.device)
            yield batch_images, batch_labels

    @torch.no_grad()
    def _loss_and_acc(self, X, y):
        # Disable training behavior and gradients while measuring the current weights.
        self.eval()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in self._batches(X, y):
            scores = self(images)
            count = len(labels)
            # Loss functions return a batch mean; weight the partial last batch correctly.
            total_loss += self.loss_function(scores, labels).item() * count
            correct += (scores.argmax(1) == labels).sum().item()
            total += count
        return total_loss / total, correct / total

    def train_model(self, X, y, X_val=None, y_val=None):
        """Train current weights and replace history; use a fresh model for independent runs."""
        if (X_val is None) != (y_val is None):
            raise ValueError('Validation images and labels must be provided together')

        # Reset the call's history and shuffle seed, without reinitializing model weights.
        self.history = {key: [] for key in self.history}
        torch.manual_seed(self.seed)
        self.to(self.device)
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate,
                                         **self.optimizer_kwargs)

        for epoch in range(self.max_epoch):
            self.train()
            for images, labels in self._batches(X, y, shuffle=True):
                optimizer.zero_grad()
                loss = self.loss_function(self(images), labels)
                loss.backward()
                optimizer.step()

            # Measure both partitions after the epoch using the same updated weights.
            train_loss, train_acc = self._loss_and_acc(X, y)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            message = f'Epoch: {epoch + 1} Loss: {train_loss:.4f} Accuracy: {train_acc:.4f}'
            if X_val is not None:
                val_loss, val_acc = self._loss_and_acc(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                message += f' Val Loss: {val_loss:.4f} Val Accuracy: {val_acc:.4f}'
            if self.verbose:
                print(message)

    @torch.no_grad()
    def test(self, X):
        # Preserve input order and return CPU class IDs without updating weights.
        self.to(self.device)
        self.eval()
        predictions = [self(images).argmax(1).cpu() for images, _ in self._batches(X)]
        return torch.cat(predictions)

    def run(self):
        # Keep validation experiments and final test runs distinct.
        if ('val' in self.data) == ('test' in self.data):
            raise ValueError('Provide exactly one evaluation partition: val or test')
        evaluation_split = 'val' if 'val' in self.data else 'test'
        validation = self.data.get('val', {})
        self.train_model(self.data['train']['X'], self.data['train']['y'],
                         validation.get('X'), validation.get('y'))
        evaluation = self.data[evaluation_split]
        return {'pred_y': self.test(evaluation['X']), 'true_y': evaluation['y']}
