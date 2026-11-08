"""Reproducible Stage 3 experiment utilities; importing this module does not train."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import torch

from .Dataset_Loader_MNIST import Dataset_Loader_MNIST
from .Dataset_Loader_ORL import Dataset_Loader_ORL
from .Dataset_Loader_CIFAR import Dataset_Loader_CIFAR
from .Method_CNN_MNIST import MNISTCNN
from .Method_CNN_ORL import ORLCNN
from .Method_CNN_CIFAR import CIFARCNN

DATASETS = {'mnist': (Dataset_Loader_MNIST, MNISTCNN, 10),
            'orl': (Dataset_Loader_ORL, ORLCNN, 40),
            'cifar': (Dataset_Loader_CIFAR, CIFARCNN, 10)}
VARIANTS = {'baseline': {}, 'kernel5': {'kernel_size': 5},
            'average_pool': {'pooling': 'average'}}


def split_training(data, seed, fraction=0.1):
    """Stratify only the supplied training partition; return disjoint partitions."""
    indices = np.arange(len(data['y']))
    train, val = train_test_split(indices, test_size=fraction, random_state=seed,
                                  stratify=data['y'])
    return ({k: v[train] for k, v in data.items()},
            {k: v[val] for k, v in data.items()})


def classification_metrics(y, predictions, classes):
    labels = np.arange(classes)
    p, r, f, _ = precision_recall_fscore_support(y, predictions, labels=labels,
                                                  average='macro', zero_division=0)
    return {'accuracy': float(accuracy_score(y, predictions)),
            'precision_macro': float(p), 'recall_macro': float(r), 'f1_macro': float(f),
            'correct': int(np.sum(np.asarray(y) == np.asarray(predictions))),
            'total': len(y),
            'confusion_matrix': confusion_matrix(y, predictions, labels=labels).tolist()}


def save_results(output, config, history, metrics, predictions, labels):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, obj in [('config', config), ('history', history), ('metrics', metrics)]:
        (output / f'{name}.json').write_text(json.dumps(obj, indent=2) + '\n')
    np.savez_compressed(output / 'predictions.npz', predictions=predictions, labels=labels)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, measure in zip(axes, ('loss', 'acc')):
        for part in ('train', 'val'):
            values = history[f'{part}_{measure}']
            if values:
                ax.plot(range(1, len(values) + 1), values, label=part)
        ax.set(xlabel='Epoch', ylabel='Loss' if measure == 'loss' else 'Accuracy')
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'learning_curves.png', dpi=150)
    plt.close(fig)


def run_experiment(dataset, mode, variant, *, data_dir=None, output, seed=2,
                   split_seed=2, epochs=20, batch_size=128, learning_rate=0.001,
                   device='cpu'):
    if mode not in ('validation', 'test'):
        raise ValueError('mode must be validation or test')
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError('epochs, batch_size and learning_rate must be positive')
    loader, model_class, classes = DATASETS[dataset]
    # Refuse overwrite before any expensive work.
    if Path(output).exists():
        raise FileExistsError(f'Output already exists: {output}')
    training = loader(split='train', data_dir=data_dir).load()
    if mode == 'validation':
        train, evaluation = split_training(training, split_seed)
    else:
        train = training
        evaluation = loader(split='test', data_dir=data_dir).load()
    model = model_class(seed=seed, **VARIANTS[variant])
    model.max_epoch, model.batch_size, model.learning_rate, model.device = epochs, batch_size, learning_rate, device
    model.train_model(train['X'], train['y'],
                      evaluation['X'] if mode == 'validation' else None,
                      evaluation['y'] if mode == 'validation' else None)
    predictions = model.test(evaluation['X']).numpy()
    metrics = classification_metrics(evaluation['y'], predictions, classes)
    config = dict(dataset=dataset, mode=mode, variant=variant, model_options=VARIANTS[variant],
                  seed=seed, split_seed=split_seed, epochs=epochs, batch_size=batch_size,
                  learning_rate=learning_rate, device=device, train_count=len(train['y']),
                  evaluation_count=len(evaluation['y']))
    save_results(output, config, model.history, metrics, predictions, evaluation['y'])
    torch.save(model.state_dict(), Path(output) / 'model.pt')
    return metrics
