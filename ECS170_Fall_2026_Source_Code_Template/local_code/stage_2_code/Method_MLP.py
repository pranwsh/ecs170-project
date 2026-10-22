'''
Stage 2 MLP w/ adjustable layers for classifying handwritten digits (784 pixels -> 10 digits).
'''

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # experiment knobs, tweak these on the object b4 calling run()
    max_epoch = 20
    batch_size = 128
    learning_rate = 1e-3
    optimizer_class = torch.optim.Adam
    optimizer_kwargs = {}
    # 'cpu', 'cuda' or 'mps' (mps = ur Mac's GPU)
    device = 'cpu'

    def __init__(self, mName, mDescription, input_dim=784, hidden_dims=(256, 128), output_dim=10,
                 activation=nn.ReLU, dropout=0.0):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), activation()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        # CrossEntropyLoss does softmax on its own, so forward() just gives raw logits
        self.loss_function = nn.CrossEntropyLoss()
        # per-epoch numbers for the convergence plots, val_* stays empty w/o data['val']
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def forward(self, x):
        return self.layers(x)

    def _tensors(self, X, y=None):
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        if y is None:
            return X
        return X, torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _loss_and_acc(self, X, y):
        nn.Module.train(self, False)
        logits = self.forward(X)
        return self.loss_function(logits, y).item(), (logits.argmax(1) == y).float().mean().item()

    # called train_model not train bc nn.Module.eval() calls self.train(False)
    def train_model(self, X, y, X_val=None, y_val=None):
        self.to(self.device)
        X, y = self._tensors(X, y)
        if X_val is not None:
            X_val, y_val = self._tensors(X_val, y_val)
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs)

        for epoch in range(self.max_epoch):
            nn.Module.train(self, True)
            perm = torch.randperm(len(X), device=self.device)
            for start in range(0, len(X), self.batch_size):
                idx = perm[start:start + self.batch_size]
                loss = self.loss_function(self.forward(X[idx]), y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss, train_acc = self._loss_and_acc(X, y)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            msg = f'Epoch: {epoch} Loss: {train_loss:.4f} Accuracy: {train_acc:.4f}'
            if X_val is not None:
                val_loss, val_acc = self._loss_and_acc(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                msg += f' Val Loss: {val_loss:.4f} Val Accuracy: {val_acc:.4f}'
            print(msg)

    @torch.no_grad()
    def test(self, X):
        nn.Module.train(self, False)
        return self.forward(self._tensors(X)).argmax(1).cpu()

    def run(self):
        print('method running...')
        print('--start training...')
        val = self.data.get('val', {})
        self.train_model(self.data['train']['X'], self.data['train']['y'], val.get('X'), val.get('y'))
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
