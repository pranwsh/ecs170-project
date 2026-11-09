"""Load one face channel and map person IDs 1-40 to training IDs 0-39."""

from .Dataset_Loader import ImageDatasetLoader


class Dataset_Loader_ORL(ImageDatasetLoader):
    # Stored faces have three equal channels and person IDs from 1 through 40.
    file_name = 'ORL'
    image_shape = (112, 92, 3)
    num_classes = 40
    # Subtract one for training; display_labels() restores the original person IDs.
    label_offset = 1
