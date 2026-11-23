"""Fixture tests for the Stage 4 raw review and joke loaders."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset
from local_code.stage_4_code.Dataset_Loader_Jokes import Dataset_Loader_Jokes
from local_code.stage_4_code.Dataset_Loader_Reviews import Dataset_Loader_Reviews


class ReviewLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        for split in ('train', 'test'):
            for sentiment in ('neg', 'pos'):
                (self.directory / split / sentiment).mkdir(parents=True)

    def write_review(self, relative_path, text):
        (self.directory / relative_path).write_text(text, encoding='utf-8')

    def test_raw_text_labels_and_source_ids_stay_aligned(self):
        self.write_review('train/neg/b_10.txt', 'Second negative review.')
        self.write_review('train/pos/a_1.txt', 'Excellent! <br /> I loved it.')
        self.write_review('train/neg/a_9.txt', 'Not good. Café scenes.\n')
        loader = Dataset_Loader_Reviews('reviews', 'sentiment', data_dir=self.directory)

        data = loader.load()

        self.assertIsInstance(loader, dataset)
        self.assertIs(data, loader.data)
        self.assertEqual(data['X'], ['Not good. Café scenes.\n', 'Second negative review.',
                                    'Excellent! <br /> I loved it.'])
        np.testing.assert_array_equal(data['y'], [0, 0, 1])
        self.assertEqual(data['y'].dtype, np.int64)
        self.assertEqual(data['ids'], ['train/neg/a_9.txt', 'train/neg/b_10.txt',
                                      'train/pos/a_1.txt'])

    def test_loads_only_the_requested_partition(self):
        self.write_review('train/pos/1_8.txt', 'Training review')
        self.write_review('test/neg/2_2.txt', 'Test review')
        train = Dataset_Loader_Reviews(data_dir=self.directory).load()
        test = Dataset_Loader_Reviews(split='test', data_dir=self.directory).load()

        self.assertEqual(train['X'], ['Training review'])
        self.assertEqual(test['X'], ['Test review'])
        self.assertEqual(train['ids'], ['train/pos/1_8.txt'])
        self.assertEqual(test['ids'], ['test/neg/2_2.txt'])
        np.testing.assert_array_equal(train['y'], [1])
        np.testing.assert_array_equal(test['y'], [0])

    def test_rejects_unknown_partition(self):
        with self.assertRaisesRegex(ValueError, 'split'):
            Dataset_Loader_Reviews(split='val', data_dir=self.directory)

    def test_missing_extracted_directory_explains_archive_setup(self):
        with self.assertRaisesRegex(FileNotFoundError, 'reviews.zip'):
            Dataset_Loader_Reviews(data_dir=self.directory / 'not_extracted').load()


class JokeLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_csv_preserves_text_and_string_ids(self):
        texts = ['A joke, with a comma.', 'She said "café".\nThen laughed.']
        with (self.directory / 'data').open('w', encoding='utf-8', newline='') as target:
            writer = csv.writer(target)
            writer.writerow(['ID', 'Joke'])
            writer.writerows(zip(['09', '2'], texts))
        loader = Dataset_Loader_Jokes('jokes', 'generation', data_dir=self.directory)

        data = loader.load()

        self.assertIsInstance(loader, dataset)
        self.assertIs(data, loader.data)
        self.assertEqual(data, {'X': texts, 'ids': ['09', '2']})
        self.assertEqual(loader.dataset_source_file_name, 'data')

    def test_lfs_pointer_explains_how_to_download_data(self):
        (self.directory / 'data').write_text(
            'version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 140224\n',
            encoding='utf-8',
        )
        with self.assertRaisesRegex(ValueError, 'git lfs pull'):
            Dataset_Loader_Jokes(data_dir=self.directory).load()


if __name__ == '__main__':
    unittest.main()
