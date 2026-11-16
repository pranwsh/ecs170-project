"""Load the supplied movie reviews and their binary sentiment labels."""

from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset


# Resolve the default location independently of the script's working directory.
DATA_DIRECTORY = Path(__file__).resolve().parents[2] / 'data' / 'stage_4_data' / 'text_classification'


class Dataset_Loader_Reviews(dataset):
    def __init__(self, dName=None, dDescription=None, *, split='train', data_dir=None):
        super().__init__(dName, dDescription)
        if split not in ('train', 'test'):
            raise ValueError("split must be 'train' or 'test'")
        self.split = split
        self.dataset_source_folder_path = DATA_DIRECTORY if data_dir is None else Path(data_dir)

    def load(self):
        """Return raw review strings, int64 labels, and relative source IDs."""
        directory = Path(self.dataset_source_folder_path)
        texts, labels, source_ids = [], [], []

        # Sentiment folders determine labels; ratings in file names are not targets.
        for label, sentiment in enumerate(('neg', 'pos')):
            folder = directory / self.split / sentiment
            if not folder.is_dir():
                raise FileNotFoundError(
                    f'{folder} is missing; extract text_classification/reviews.zip '
                    'as described in Stage 4 Split.md'
                )

            # Sorted paths keep each review, label, and source ID in a stable order.
            for path in sorted(folder.glob('*.txt')):
                texts.append(path.read_text(encoding='utf-8'))
                labels.append(label)
                source_ids.append(path.relative_to(directory).as_posix())

        # Keep raw text separate from metadata; tokenization happens after splitting.
        self.data = {'X': texts, 'y': np.asarray(labels, dtype=np.int64), 'ids': source_ids}
        return self.data
