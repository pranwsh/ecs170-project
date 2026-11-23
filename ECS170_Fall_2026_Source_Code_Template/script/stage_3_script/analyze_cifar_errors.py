"""Analyze a saved CIFAR-10 run without training or loading image data."""
import argparse
import json
from pathlib import Path

import numpy as np

CLASS_NAMES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
               'frog', 'horse', 'ship', 'truck')


def analyze_run(run_dir):
    """Return per-class statistics and ranked directional errors for one run.

    Matrix rows are true classes and columns are predicted classes. Verify the
    stored matrix before using the predictions so mismatched artifacts fail loud.
    """
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text())
    if config.get('dataset') != 'cifar':
        raise ValueError('Expected a CIFAR run')
    metrics = json.loads((run_dir / 'metrics.json').read_text())
    with np.load(run_dir / 'predictions.npz', allow_pickle=False) as data:
        labels, predictions = data['labels'], data['predictions']
    if (labels.ndim != 1 or predictions.ndim != 1 or labels.shape != predictions.shape
            or not len(labels)):
        raise ValueError('Labels and predictions must be nonempty equal-length vectors')
    for values in (labels, predictions):
        if not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values >= 10)):
            raise ValueError('CIFAR labels and predictions must be integers in [0, 9]')
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    if (not np.array_equal(matrix, np.asarray(metrics['confusion_matrix']))
            or metrics['total'] != len(labels) or metrics['correct'] != int(np.trace(matrix))):
        raise ValueError('Saved metrics do not match predictions')
    classes = []
    for i, name in enumerate(CLASS_NAMES):
        support, predicted, correct = int(matrix[i].sum()), int(matrix[:, i].sum()), int(matrix[i, i])
        classes.append({'class_id': i, 'class_name': name, 'support': support,
                        'correct': correct, 'errors': support - correct,
                        'recall': correct / support if support else 0.0,
                        'precision': correct / predicted if predicted else 0.0})
    confusions = [{'true_id': i, 'true_class': CLASS_NAMES[i], 'predicted_id': j,
                   'predicted_class': CLASS_NAMES[j], 'count': int(matrix[i, j]),
                   'fraction_of_true_class': int(matrix[i, j]) / int(matrix[i].sum())}
                  for i in range(10) for j in range(10) if i != j and matrix[i, j] > 0]
    confusions.sort(key=lambda item: (-item['count'], item['true_id'], item['predicted_id']))
    return {'dataset': 'cifar', 'mode': config.get('mode'), 'total': len(labels),
            'correct': int(np.trace(matrix)), 'per_class': classes,
            'confusion_matrix': matrix.tolist(), 'directional_confusions': confusions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, help='Directory containing saved run artifacts')
    parser.add_argument('--output', type=Path, help='Write JSON here (default: stdout)')
    args = parser.parse_args()
    text = json.dumps(analyze_run(args.run_dir), indent=2) + '\n'
    if args.output:
        with args.output.open('x') as output:
            output.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
