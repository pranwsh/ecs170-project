"""Binary review classification metrics, with label 1 as the positive class."""
import numpy as np
from local_code.base_class.evaluate import evaluate


def classification_metrics(y_true, y_pred):
    """Return JSON-serializable metrics; confusion matrix rows=true, columns=predicted.

    Precision, recall and F1 use binary positive-class averaging (label 1).
    Undefined ratios are reported as zero.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.ndim != 1 or pred.ndim != 1 or true.shape != pred.shape or true.size == 0:
        raise ValueError('labels and predictions must be nonempty equal-length vectors')
    if not np.isin(true, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError('labels and predictions must contain only 0 and 1')
    tn = int(((true == 0) & (pred == 0)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    tp = int(((true == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'accuracy': (tp + tn) / true.size, 'precision_positive': precision,
            'recall_positive': recall,
            'f1_positive': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'positive_label': 1, 'averaging': 'binary',
            'confusion_matrix': [[tn, fp], [fn, tp]]}


class Evaluate_Classification(evaluate):
    def evaluate(self, y_true, y_pred):
        return classification_metrics(y_true, y_pred)
