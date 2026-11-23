"""Mini-batch training for Stage 4's two recurrent tasks.

Batches follow Text_Preprocessing's dictionary contract. Losses in the
returned histories are means over examples (classification) or non-padding
 target tokens (generation), not means of batch means.
"""

import torch
from torch import nn


def _epoch(model, batches, task, device, optimizer=None, pad_id=0):
    training = optimizer is not None
    model.train(training)
    total_loss = total_units = correct = 0
    with torch.set_grad_enabled(training):
        for batch in batches:
            inputs = batch['input_ids'].to(device)
            lengths = batch['lengths'].to(device)
            targets = batch['labels' if task == 'classification' else 'target_ids'].to(device)
            if training:
                optimizer.zero_grad()
            scores = model(inputs, lengths)
            if task == 'classification':
                units = targets.numel()
                loss = nn.functional.cross_entropy(scores, targets, reduction='sum')
                correct += (scores.argmax(dim=-1) == targets).sum().item()
            else:
                units = (targets != pad_id).sum().item()
                if not units:
                    continue
                loss = nn.functional.cross_entropy(
                    scores.reshape(-1, scores.shape[-1]), targets.reshape(-1),
                    ignore_index=pad_id, reduction='sum')
            if not units:
                continue
            if training:
                (loss / units).backward()
                optimizer.step()
            total_loss += loss.item()
            total_units += units
    if not total_units:
        raise ValueError('epoch contains no usable examples or target tokens')
    result = {'loss': total_loss / total_units}
    if task == 'classification':
        result['acc'] = correct / total_units
    return result


def evaluate_classification(model, batches, device='cpu'):
    """Return example-weighted loss and accuracy without updating weights."""
    model.to(device)
    return _epoch(model, batches, 'classification', device)


def evaluate_generation(model, batches, device='cpu', pad_id=0):
    """Return non-padding-token-weighted next-token loss."""
    model.to(device)
    return _epoch(model, batches, 'generation', device, pad_id=pad_id)


def _train(model, train_batches, val_batches, optimizer, epochs, task, device, pad_id):
    if epochs < 0:
        raise ValueError('epochs must be nonnegative')
    model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    if task == 'classification':
        history.update(train_acc=[], val_acc=[])
    for _ in range(epochs):
        train = _epoch(model, train_batches, task, device, optimizer, pad_id)
        history['train_loss'].append(train['loss'])
        if task == 'classification':
            history['train_acc'].append(train['acc'])
        if val_batches is not None:
            val = _epoch(model, val_batches, task, device, pad_id=pad_id)
            history['val_loss'].append(val['loss'])
            if task == 'classification':
                history['val_acc'].append(val['acc'])
    return history


def train_classification(model, train_batches, val_batches, optimizer, epochs, device='cpu'):
    """Train a classifier; batches must be re-iterable for multiple epochs."""
    return _train(model, train_batches, val_batches, optimizer, epochs,
                  'classification', device, 0)


def train_generation(model, train_batches, val_batches, optimizer, epochs, device='cpu', pad_id=0):
    """Train a next-token model, ignoring pad_id in targets."""
    return _train(model, train_batches, val_batches, optimizer, epochs,
                  'generation', device, pad_id)
