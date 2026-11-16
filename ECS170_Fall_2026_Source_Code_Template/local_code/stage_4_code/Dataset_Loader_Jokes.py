"""Load the supplied joke CSV and preserve its source IDs."""

import csv
from pathlib import Path

from local_code.base_class.dataset import dataset


# The supplied CSV is named data and has no file extension.
DATA_DIRECTORY = Path(__file__).resolve().parents[2] / 'data' / 'stage_4_data' / 'text_generation'


class Dataset_Loader_Jokes(dataset):
    def __init__(self, dName=None, dDescription=None, *, data_dir=None):
        super().__init__(dName, dDescription)
        self.dataset_source_folder_path = DATA_DIRECTORY if data_dir is None else Path(data_dir)
        self.dataset_source_file_name = 'data'

    def load(self):
        """Return raw joke strings and CSV IDs without constructing token targets."""
        path = Path(self.dataset_source_folder_path) / self.dataset_source_file_name
        texts, source_ids = [], []
        with path.open(encoding='utf-8', newline='') as source:
            # An LFS pointer is placeholder text rather than the downloaded CSV.
            if source.read(80).startswith('version https://git-lfs.github.com/spec/'):
                raise ValueError(f'{path} is a Git LFS pointer; run git lfs pull')
            source.seek(0)

            # CSV parsing preserves commas, quoted dialogue, and multiline jokes.
            for record in csv.DictReader(source):
                texts.append(record['Joke'])
                source_ids.append(record['ID'])

        # IDs identify original jokes for comparison; they are not prediction labels.
        self.data = {'X': texts, 'ids': source_ids}
        return self.data
