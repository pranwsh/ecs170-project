"""Load the supplied CIFAR images, preserving all three color channels."""

from .Dataset_Loader import ImageDatasetLoader


class Dataset_Loader_CIFAR(ImageDatasetLoader):
    # The shared loader preserves RGB channels for ten object categories.
    file_name = 'CIFAR'
    image_shape = (32, 32, 3)
    num_classes = 10
