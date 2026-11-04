"""Read one partition of the supplied Stage 3 pickle files."""

import pickle
from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset


# Find the data relative to this file, regardless of the launch directory.
DATA_DIRECTORY = Path(__file__).resolve().parents[2] / 'data' / 'stage_3_data'


class ImageDatasetLoader(dataset):
    """Small shared loader; subclasses specify each dataset's stored layout."""

    # Each subclass supplies its file name, stored image shape, and label range.
    file_name = None
    image_shape = None
    num_classes = None
    label_offset = 0

    def __init__(self, dName=None, dDescription=None, *, split='train', data_dir=None):
        super().__init__(dName, dDescription)
        # Validation splits are made later from training data, not loaded here.
        if split not in ('train', 'test'):
            raise ValueError("split must be 'train' or 'test'")
        self.split = split
        # Keep the source-path fields used by the professor's dataset template.
        self.dataset_source_folder_path = DATA_DIRECTORY if data_dir is None else Path(data_dir)
        self.dataset_source_file_name = self.file_name

    def load(self):
        """Return normalized float32 NCHW images and zero-based int64 labels.

        The professor's pickle contains both partitions in one file. Unpickling
        reads that container, but only the requested partition is preprocessed
        and returned. Load only the supplied, trusted project pickle files.
        """
        path = Path(self.dataset_source_folder_path) / self.dataset_source_file_name
        with path.open('rb') as source:
            # An LFS pointer contains placeholder text instead of dataset bytes.
            if source.read(80).startswith(b'version https://git-lfs.github.com/spec/'):
                raise ValueError(f'{path} is a Git LFS pointer; run git lfs pull')
            # Rewind after the header check so pickle reads from the beginning.
            source.seek(0)
            # Read the container, then select only the requested partition.
            records = pickle.load(source)[self.split]
        if not records:
            raise ValueError('The requested partition is empty')

        # Stack images together and collect labels in the same example order.
        images = np.stack([record['image'] for record in records])
        labels = np.asarray([record['label'] for record in records])

        # Check the stored dimensions and pixel range before changing the data.
        if images.shape[1:] != self.image_shape:
            raise ValueError(f'Expected stored image shape {self.image_shape}, got {images.shape[1:]}')
        if not np.isfinite(images).all() or np.any(images < 0) or np.any(images > 255):
            raise ValueError('Pixels must be finite values from 0 to 255')
        # Check before casting so fractional labels cannot become valid integers.
        if (labels.shape != (len(records),) or not np.isfinite(labels).all()
                or np.any(labels != np.floor(labels))
                or np.any(labels < self.label_offset)
                or np.any(labels >= self.num_classes + self.label_offset)):
            raise ValueError('Labels must be integer class IDs in the dataset range')

        # PyTorch convolutions expect channels before height and width.
        if self.file_name == 'MNIST':
            # Add the missing channel dimension: (N, 28, 28) -> (N, 1, 28, 28).
            images = images[:, None, :, :]
        elif self.file_name == 'ORL':
            # The supplied ORL images have equal RGB channels, so keep just one.
            images = images[:, :, :, 0][:, None, :, :]
        else:
            # Preserve RGB values and move channels first: (N, H, W, 3) -> (N, 3, H, W).
            images = images.transpose(0, 3, 1, 2)

        # Use float32 in contiguous memory and scale pixels from 0-255 to 0-1.
        images = np.ascontiguousarray(images, dtype=np.float32)
        images /= 255.0

        # Pair images with integer labels; subtract the offset to map ORL IDs to 0-39.
        self.data = {'X': images, 'y': labels.astype(np.int64) - self.label_offset}
        return self.data

    def display_labels(self, labels):
        """Restore supplied identities, particularly ORL's person IDs 1-40."""
        return np.asarray(labels) + self.label_offset
