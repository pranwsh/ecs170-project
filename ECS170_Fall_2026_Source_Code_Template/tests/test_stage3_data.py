"""Daniel's loader tests: python -m unittest discover -s tests -p test_stage3_data.py -v."""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset
from local_code.stage_3_code.Dataset_Loader_MNIST import Dataset_Loader_MNIST
from local_code.stage_3_code.Dataset_Loader_ORL import Dataset_Loader_ORL
from local_code.stage_3_code.Dataset_Loader_CIFAR import Dataset_Loader_CIFAR


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write(self, cls, train, test=None):
        with (self.directory / cls.file_name).open('wb') as target:
            pickle.dump({'train': train, 'test': test}, target)

    def test_shapes_types_scaling_and_labels(self):
        for cls, output_shape in ((Dataset_Loader_MNIST, (1, 28, 28)),
                                  (Dataset_Loader_ORL, (1, 112, 92)),
                                  (Dataset_Loader_CIFAR, (3, 32, 32))):
            with self.subTest(dataset=cls.file_name):
                dtype = np.float64 if cls is Dataset_Loader_MNIST else np.uint8
                image = np.zeros(cls.image_shape, dtype=dtype)
                image.flat[-1] = 255
                if cls is Dataset_Loader_ORL:
                    image[:, :, :] = 128
                label = cls.num_classes - 1 + cls.label_offset
                self.write(cls, [{'image': image, 'label': label}])
                loader = cls('name', 'description', data_dir=self.directory)
                self.assertIsInstance(loader, dataset)
                data = loader.load()
                self.assertIs(data, loader.data)
                self.assertEqual(set(data), {'X', 'y'})
                self.assertEqual(data['X'].shape, (1, *output_shape))
                self.assertEqual(data['X'].dtype, np.float32)
                self.assertTrue(data['X'].flags.c_contiguous)
                self.assertEqual(data['y'].dtype, np.int64)
                self.assertEqual(data['y'].tolist(), [cls.num_classes - 1])
                self.assertEqual(loader.display_labels(data['y']).tolist(), [label])
                expected = image.astype(np.float32) / 255
                if cls is Dataset_Loader_MNIST:
                    expected = expected[None]
                elif cls is Dataset_Loader_ORL:
                    expected = expected[:, :, 0][None]
                else:
                    expected = expected.transpose(2, 0, 1)
                np.testing.assert_array_equal(data['X'][0], expected)

    def test_rgb_channel_order(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, :, :] = [0, 127, 255]
        self.write(Dataset_Loader_CIFAR, [{'image': image, 'label': 0}])
        result = Dataset_Loader_CIFAR(data_dir=self.directory).load()
        np.testing.assert_array_equal(result['X'][0, :, 0, 0],
                                      np.array([0, 127, 255], dtype=np.float32) / 255)

    def test_multiple_images_keep_label_order_and_orl_identities(self):
        images = [np.full((112, 92, 3), value, dtype=np.uint8) for value in (0, 255)]
        records = [{'image': image, 'label': label} for image, label in zip(images, (40, 1))]
        self.write(Dataset_Loader_ORL, records)
        loader = Dataset_Loader_ORL(data_dir=self.directory)
        result = loader.load()
        self.assertEqual(result['y'].tolist(), [39, 0])
        self.assertEqual(loader.display_labels(result['y']).tolist(), [40, 1])
        self.assertTrue(np.all(result['X'][0] == 0))
        self.assertTrue(np.all(result['X'][1] == 1))

    def test_template_source_path_fields(self):
        image = np.zeros((28, 28))
        self.write(Dataset_Loader_MNIST, [{'image': image, 'label': 0}])
        (self.directory / 'MNIST').rename(self.directory / 'custom_digits')
        loader = Dataset_Loader_MNIST('digits', 'Stage 3')
        loader.dataset_source_folder_path = str(self.directory)
        loader.dataset_source_file_name = 'custom_digits'
        self.assertEqual(loader.dataset_name, 'digits')
        self.assertEqual(loader.dataset_descrition, 'Stage 3')
        self.assertEqual(loader.load()['y'].tolist(), [0])

    def test_partitions_are_selected_separately(self):
        image = np.zeros((28, 28))
        self.write(Dataset_Loader_MNIST, [{'image': image, 'label': 2}],
                   [{'image': image + 255, 'label': 9}])
        for split, label, pixel in (('train', 2, 0), ('test', 9, 1)):
            result = Dataset_Loader_MNIST(split=split, data_dir=self.directory).load()
            self.assertEqual(result['y'].tolist(), [label])
            self.assertTrue(np.all(result['X'] == pixel))
        # A malformed, unused test partition must not affect a training load.
        self.write(Dataset_Loader_MNIST, [{'image': image, 'label': 2}], 'unused')
        self.assertEqual(Dataset_Loader_MNIST(data_dir=self.directory).load()['y'].tolist(), [2])

    def test_bad_data_is_rejected(self):
        for image, label in ((np.zeros((27, 28)), 0), (np.full((28, 28), np.nan), 0),
                             (np.full((28, 28), np.inf), 0),
                             (np.full((28, 28), 256), 0), (np.full((28, 28), -1), 0),
                             (np.zeros((28, 28)), 1.5), (np.zeros((28, 28)), 10),
                             (np.zeros((28, 28)), -1), (np.zeros((28, 28)), np.nan),
                             (np.zeros((28, 28)), np.inf)):
            with self.subTest(shape=image.shape, label=label):
                self.write(Dataset_Loader_MNIST, [{'image': image, 'label': label}])
                with self.assertRaises(ValueError):
                    Dataset_Loader_MNIST(data_dir=self.directory).load()
        for label in (0, 41):
            self.write(Dataset_Loader_ORL, [{'image': np.zeros((112, 92, 3)), 'label': label}])
            with self.assertRaises(ValueError):
                Dataset_Loader_ORL(data_dir=self.directory).load()

    def test_missing_empty_lfs_and_invalid_split(self):
        with self.assertRaises(ValueError):
            Dataset_Loader_MNIST(split='val')
        with self.assertRaises(FileNotFoundError):
            Dataset_Loader_MNIST(data_dir=self.directory).load()
        self.write(Dataset_Loader_MNIST, [])
        with self.assertRaises(ValueError):
            Dataset_Loader_MNIST(data_dir=self.directory).load()
        (self.directory / 'MNIST').write_bytes(b'version https://git-lfs.github.com/spec/v1\n')
        with self.assertRaisesRegex(ValueError, 'Git LFS pointer'):
            Dataset_Loader_MNIST(data_dir=self.directory).load()


if __name__ == '__main__':
    unittest.main()
