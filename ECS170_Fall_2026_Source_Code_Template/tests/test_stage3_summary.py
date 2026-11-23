"""Tests for report-ready summaries of experiment artifacts."""
import json
import tempfile
import unittest
from pathlib import Path
from script.stage_3_script.summarize_experiments import load_runs, render_summary, main


class SummaryTests(unittest.TestCase):
    def write_run(self, root, name, mode, variant):
        directory = root / 'mnist' / name
        directory.mkdir(parents=True)
        config = {'dataset': 'mnist', 'mode': mode, 'variant': variant,
                  'model_options': {'kernel_size': 5} if variant == 'kernel5' else {},
                  'seed': 2, 'split_seed': 2, 'epochs': 20, 'batch_size': 128,
                  'learning_rate': .001, 'device': 'cpu', 'train_count': 9,
                  'evaluation_count': 3}
        metrics = {'accuracy': 2/3, 'precision_macro': .5, 'recall_macro': .6,
                   'f1_macro': .55, 'correct': 2, 'total': 3}
        (directory / 'config.json').write_text(json.dumps(config))
        (directory / 'metrics.json').write_text(json.dumps(metrics))
        return directory

    def test_tables_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'runs'
            self.write_run(root, 'baseline', 'validation', 'baseline')
            self.write_run(root, 'kernel', 'validation', 'kernel5')
            self.write_run(root, 'final', 'test', 'kernel5')
            text = render_summary(load_runs(root))
            validation, rest = text.split('## Final test evaluations')
            self.assertIn('mnist/baseline', validation)
            self.assertIn('mnist/kernel', validation)
            self.assertNotIn('mnist/final', validation)
            self.assertIn('mnist/final', rest)
            self.assertIn('0.6667', text)
            self.assertIn('2/3', text)
            self.assertIn('kernel_size', text)
            self.assertIn('Split seed', text)
            output = Path(tmp) / 'summary.md'
            main(['--root', str(root), '--output', str(output)])
            self.assertEqual(output.read_text(), text)

    def test_incomplete_and_inconsistent_runs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = self.write_run(root, 'baseline', 'validation', 'baseline')
            (directory / 'metrics.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Invalid run'):
                load_runs(root)
            (directory / 'metrics.json').write_text(json.dumps({
                'accuracy': 1, 'precision_macro': 1, 'recall_macro': 1,
                'f1_macro': 1, 'correct': 2, 'total': 2}))
            with self.assertRaisesRegex(ValueError, 'evaluation_count'):
                load_runs(root)

    def test_empty_root_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'No experiment runs'):
                load_runs(tmp)


if __name__ == '__main__':
    unittest.main()
