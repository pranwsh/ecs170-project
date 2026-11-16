"""Check the inspection CLI with small datasets outside the project directory."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TEMPLATE_ROOT = Path(__file__).resolve().parents[1]


class InspectionTests(unittest.TestCase):
    def test_inspection_records_training_lengths_without_using_test_vocabulary(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for split in ('train', 'test'):
                for sentiment in ('neg', 'pos'):
                    folder = directory / 'text_classification' / split / sentiment
                    folder.mkdir(parents=True)
                    for index in range(10 if split == 'train' else 1):
                        text = (' '.join(['good'] * 256 + [f'{sentiment}{index}'])
                                if split == 'train' else 'testonly')
                        (folder / f'{index}_5.txt').write_text(text, encoding='utf-8')
            joke_directory = directory / 'text_generation'
            joke_directory.mkdir()
            with (joke_directory / 'data').open('w', encoding='utf-8', newline='') as target:
                writer = csv.writer(target)
                writer.writerow(['ID', 'Joke'])
                writer.writerows((str(index), f'What happened {index}?') for index in range(10))
            output = directory / 'output'

            result = subprocess.run(
                [sys.executable, str(TEMPLATE_ROOT / 'script' / 'stage_4_script' / 'inspect_data.py'),
                 '--data-dir', str(directory), '--output-dir', str(output)],
                cwd=directory, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output / 'dataset_checks.json').read_text(encoding='utf-8'))
            training = summary['reviews']['train']
            self.assertEqual(training['class_counts'], [10, 10])
            self.assertEqual(training['token_lengths']['min'], 257)
            self.assertEqual(training['truncation']['256'], {'count': 20, 'fraction': 1.0})
            self.assertEqual(summary['reviews']['test'], {'count': 2, 'class_counts': [1, 1]})
            self.assertEqual(summary['jokes']['supplied_corpus']['token_lengths']['max'], 4)
            self.assertEqual(summary['vocabulary_checks']['status'], 'complete')
            self.assertNotIn('testonly', json.dumps(summary))
            vocabulary = json.loads((output / 'review_vocabulary.json').read_text(encoding='utf-8'))
            self.assertNotIn('testonly', vocabulary['token_to_id'])
            self.assertEqual(summary['reviews']['prepared']['partitions']['val']['count'], 2)
            self.assertNotIn('length_comparison', summary['reviews'])
            self.assertTrue((output / 'splits.json').is_file())
            self.assertTrue((output / 'joke_vocabulary.json').is_file())
            splits = json.loads((output / 'splits.json').read_text(encoding='utf-8'))
            joke_vocabulary = json.loads((output / 'joke_vocabulary.json').read_text(encoding='utf-8'))
            self.assertNotIn(splits['jokes']['val'][0], joke_vocabulary['token_to_id'])
            self.assertFalse((output / 'review_length_comparison.png').exists())
            for name in ('review_class_counts.png', 'training_token_lengths.png'):
                self.assertTrue((output / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))


if __name__ == '__main__':
    unittest.main()
