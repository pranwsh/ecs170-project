"""Check the final data-only comparison against hand-calculated token counts."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from local_code.stage_4_code.Text_Preprocessing import build_vocabulary
from script.stage_4_script.compare_review_lengths import compare_lengths


TEMPLATE_ROOT = Path(__file__).resolve().parents[1]


class LengthComparisonTests(unittest.TestCase):
    def test_counts_use_original_and_retained_token_denominators(self):
        vocabulary = build_vocabulary(['known'])
        texts = ['known ' * 128, 'mystery ' + 'known ' * 256]
        comparison = compare_lengths(texts, vocabulary)

        for limit, retained, truncated in (('128', 256, 1), ('256', 384, 1), ('unlimited', 385, 0)):
            details = comparison[limit]
            self.assertEqual(details['review_count'], 2)
            self.assertEqual(details['original_tokens'], 385)
            self.assertEqual(details['truncated_count'], truncated)
            self.assertEqual(details['truncated_fraction'], truncated / 2)
            self.assertEqual(details['retained_tokens'], retained)
            self.assertAlmostEqual(details['retained_fraction'], retained / 385)
            self.assertEqual(details['unknown_tokens'], 1)
            self.assertAlmostEqual(details['unknown_fraction'], 1 / retained)
        self.assertNotIn('mystery', vocabulary['token_to_id'])

    def test_uncapped_case_includes_unknown_tokens_beyond_the_cap(self):
        comparison = compare_lengths(['known ' * 256 + 'mystery'], build_vocabulary(['known']))
        self.assertEqual(comparison['128']['unknown_tokens'], 0)
        self.assertEqual(comparison['256']['unknown_tokens'], 0)
        self.assertEqual(comparison['unlimited']['unknown_tokens'], 1)
        self.assertEqual(comparison['unlimited']['retained_tokens'], 257)
        self.assertEqual(comparison['unlimited']['retained_fraction'], 1.0)
        self.assertAlmostEqual(comparison['unlimited']['unknown_fraction'], 1 / 257)

    def test_cli_consumes_saved_artifacts_without_modifying_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = directory / 'data' / 'text_classification'
            examples = {'train/neg/a.txt': 'known ' * 129, 'train/pos/b.txt': 'known ' * 128,
                        'train/neg/c.txt': 'mystery ' + 'known ' * 256,
                        'train/pos/d.txt': 'known ' * 128}
            for source_id, text in examples.items():
                path = data / source_id
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding='utf-8')
            prepared = directory / 'prepared'
            prepared.mkdir()
            splits = {'seed': 2, 'reviews': {'train': ['train/neg/a.txt', 'train/pos/b.txt'],
                                            'val': ['train/neg/c.txt', 'train/pos/d.txt']}}
            vocabulary = build_vocabulary(['known'])
            for name, content in (('splits.json', splits), ('review_vocabulary.json', vocabulary)):
                (prepared / name).write_text(json.dumps(content), encoding='utf-8')
            original = {path.name: path.read_bytes() for path in prepared.iterdir()}
            output = directory / 'output'

            result = subprocess.run(
                [sys.executable, str(TEMPLATE_ROOT / 'script/stage_4_script/compare_review_lengths.py'),
                 '--data-dir', str(directory / 'data'), '--prepared-dir', str(prepared),
                 '--output-dir', str(output)],
                cwd=directory, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output / 'comparison.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['vocabulary_size'], 3)
            self.assertEqual(summary['partitions']['val']['128']['retained_tokens'], 256)
            self.assertEqual(summary['partitions']['val']['256']['retained_tokens'], 384)
            self.assertEqual(summary['partitions']['val']['256']['unknown_tokens'], 1)
            self.assertEqual(summary['partitions']['val']['unlimited']['retained_tokens'], 385)
            self.assertEqual(summary['partitions']['val']['unlimited']['truncated_count'], 0)
            self.assertNotIn('accuracy', json.dumps(summary))
            self.assertEqual({path.name: path.read_bytes() for path in prepared.iterdir()}, original)
            self.assertTrue((output / 'review_length_comparison.png').read_bytes().startswith(b'\x89PNG'))


if __name__ == '__main__':
    unittest.main()
