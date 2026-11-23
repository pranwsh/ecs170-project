"""Verify shared split membership, sentiment balance, and grouped duplicate jokes."""

import unittest

import numpy as np

from local_code.stage_4_code.Data_Splits import create_split_ids, select_records


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.reviews = {'X': [f'review {index}' for index in range(20)],
                        'ids': [f'train/{index}.txt' for index in range(20)],
                        'y': np.asarray([0] * 10 + [1] * 10, dtype=np.int64)}
        self.jokes = {'X': [f'joke {index}' for index in range(10)] + ['joke 0'],
                      'ids': [str(index) for index in range(11)]}

    def test_splits_are_repeatable_disjoint_and_cover_all_records(self):
        splits = create_split_ids(self.reviews, self.jokes)
        self.assertEqual(splits, create_split_ids(self.reviews, self.jokes))
        for task, data in (('reviews', self.reviews), ('jokes', self.jokes)):
            train, validation = set(splits[task]['train']), set(splits[task]['val'])
            self.assertFalse(train & validation)
            self.assertEqual(train | validation, set(data['ids']))
        validation = select_records(self.reviews, splits['reviews']['val'])
        self.assertEqual(np.bincount(validation['y']).tolist(), [1, 1])

    def test_identical_jokes_do_not_cross_partitions(self):
        splits = create_split_ids(self.reviews, self.jokes)['jokes']
        train = select_records(self.jokes, splits['train'])
        validation = select_records(self.jokes, splits['val'])
        self.assertFalse(set(train['X']) & set(validation['X']))
        self.assertEqual('0' in splits['train'], '10' in splits['train'])

    def test_identical_reviews_stay_together_with_matching_sentiment(self):
        reviews = {'X': self.reviews['X'] * 2,
                   'ids': self.reviews['ids'] + [f'train/copy{index}.txt' for index in range(20)],
                   'y': np.tile(self.reviews['y'], 2)}
        splits = create_split_ids(reviews, self.jokes)['reviews']
        training = select_records(reviews, splits['train'])
        validation = select_records(reviews, splits['val'])

        self.assertFalse(set(training['X']) & set(validation['X']))
        self.assertEqual(set(training['ids']) | set(validation['ids']), set(reviews['ids']))
        self.assertEqual(np.bincount(validation['y']).tolist(), [2, 2])

    def test_selection_preserves_requested_order_and_label_alignment(self):
        selected = select_records(self.reviews, ['train/15.txt', 'train/2.txt'])
        self.assertEqual(selected['X'], ['review 15', 'review 2'])
        self.assertEqual(selected['ids'], ['train/15.txt', 'train/2.txt'])
        self.assertEqual(selected['y'].tolist(), [1, 0])


if __name__ == '__main__':
    unittest.main()
