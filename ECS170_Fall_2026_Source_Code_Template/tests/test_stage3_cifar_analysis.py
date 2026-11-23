"""Synthetic, non-training tests for saved CIFAR error analysis."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from script.stage_3_script.analyze_cifar_errors import analyze_run


class CIFARAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        (self.path / 'config.json').write_text(json.dumps({'dataset': 'cifar', 'mode': 'test'}))
        labels = np.array([0, 0, 1, 1, 1])
        predictions = np.array([0, 1, 0, 0, 1])
        np.savez(self.path / 'predictions.npz', labels=labels, predictions=predictions)
        matrix = np.zeros((10, 10), dtype=int)
        np.add.at(matrix, (labels, predictions), 1)
        (self.path / 'metrics.json').write_text(json.dumps(
            {'confusion_matrix': matrix.tolist(), 'correct': 2, 'total': 5}))

    def test_class_metrics_and_direction(self):
        result = analyze_run(self.path)
        self.assertEqual(result['per_class'][0]['class_name'], 'airplane')
        self.assertEqual(result['per_class'][0]['recall'], .5)
        self.assertAlmostEqual(result['per_class'][0]['precision'], 1 / 3)
        self.assertEqual(result['per_class'][2]['recall'], 0)
        top = result['directional_confusions'][0]
        self.assertEqual((top['true_class'], top['predicted_class'], top['count']),
                         ('automobile', 'airplane', 2))
        self.assertAlmostEqual(top['fraction_of_true_class'], 2 / 3)

    def test_mismatched_metrics_rejected(self):
        metrics = json.loads((self.path / 'metrics.json').read_text())
        metrics['correct'] = 4
        (self.path / 'metrics.json').write_text(json.dumps(metrics))
        with self.assertRaisesRegex(ValueError, 'do not match'):
            analyze_run(self.path)

    def test_invalid_labels_rejected(self):
        np.savez(self.path / 'predictions.npz', labels=np.array([10]), predictions=np.array([0]))
        with self.assertRaisesRegex(ValueError, 'integers'):
            analyze_run(self.path)


if __name__ == '__main__':
    unittest.main()
