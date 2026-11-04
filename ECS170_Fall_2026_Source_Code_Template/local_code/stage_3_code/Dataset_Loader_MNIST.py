"""Load the supplied handwritten digits using the professor's dataset class."""

from .Dataset_Loader import ImageDatasetLoader


class Dataset_Loader_MNIST(ImageDatasetLoader):
    # The shared loader reads 28x28 grayscale images labeled with digits 0-9.
    file_name = 'MNIST'
    image_shape = (28, 28)
    num_classes = 10
