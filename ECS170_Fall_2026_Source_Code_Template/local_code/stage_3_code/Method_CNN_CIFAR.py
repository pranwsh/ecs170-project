"""CNN for the ten-class CIFAR color image dataset."""
from torch import nn
from .Method_CNN import Method_CNN


class CIFARCNN(Method_CNN):
    def __init__(self, mName='cifar', mDescription='CIFAR CNN', *, kernel_size=3,
                 pooling='max', hidden_dim=128, loss_function=None, seed=2):
        super().__init__(mName, mDescription, loss_function=loss_function, seed=seed)
        if kernel_size not in (3, 5) or pooling not in ('max', 'average') or hidden_dim <= 0:
            raise ValueError('Invalid kernel_size, pooling, or hidden_dim')
        pool = nn.MaxPool2d if pooling == 'max' else nn.AvgPool2d
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size, padding=kernel_size // 2), nn.ReLU(), pool(2),
            nn.Conv2d(32, 64, kernel_size, padding=kernel_size // 2), nn.ReLU(), pool(2),
            nn.Conv2d(64, 128, kernel_size, padding=kernel_size // 2), nn.ReLU(), pool(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, hidden_dim),
                                        nn.ReLU(), nn.Linear(hidden_dim, 10))

    def forward(self, images):
        return self.classifier(self.features(images))
