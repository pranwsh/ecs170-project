"""Convolutional network for 28x28 grayscale digit images."""

from torch import nn

from .Method_CNN import Method_CNN


class MNISTCNN(Method_CNN):
    """Two convolution blocks followed by a ten-class classifier."""

    def __init__(self, mName='mnist', mDescription='MNIST CNN', *, kernel_size=3,
                 pooling='max', loss_function=None, seed=2):
        super().__init__(mName, mDescription, loss_function=loss_function, seed=seed)

        # Support the kernel and pooling choices used in the MNIST comparisons.
        if kernel_size not in (3, 5):
            raise ValueError('kernel_size must be 3 or 5')
        if pooling not in ('max', 'average'):
            raise ValueError("pooling must be 'max' or 'average'")
        self.kernel_size = kernel_size
        self.pooling = pooling
        pooling_class = nn.MaxPool2d if pooling == 'max' else nn.AvgPool2d

        # Padding preserves image size during convolution; each pooling step halves it.
        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            pooling_class(2),  # (N, 16, 14, 14)
            nn.Conv2d(16, 32, kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            pooling_class(2),  # (N, 32, 7, 7)
        )

        # Flatten the feature maps, learn 64 hidden features, and score ten digits.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, images):
        # Return raw scores; cross-entropy loss applies the probability normalization.
        return self.classifier(self.features(images))
