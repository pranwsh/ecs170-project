"""Save labeled training examples and class counts without opening windows."""

from pathlib import Path

import matplotlib
# Save figures without opening a plotting window.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_visualizations(data, dataset_name, output_dir, *, label_offset=0):
    """Save one training sample per class and a chart of training class counts.

    Pass normalized, channel-first training images from a Stage 3 loader.
    Set label_offset=1 to display ORL's original person IDs from 1 through 40.
    """
    # Count the labels and arrange up to ten classes in each row of the grid.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images, labels = data['X'], data['y']
    classes, counts = np.unique(labels, return_counts=True)
    columns = min(10, len(classes))
    rows = (len(classes) + columns - 1) // columns
    paths = (output_dir / f'{dataset_name.lower()}_samples.png',
             output_dir / f'{dataset_name.lower()}_class_counts.png')
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.5, rows * 1.8), squeeze=False)
    axes = axes.flatten()

    # Hide axes and leave any unused cells blank.
    for ax in axes:
        ax.axis('off')

    for index, class_id in enumerate(classes):
        # Pick the first example of each class for reproducible sample figures.
        sample = images[np.flatnonzero(labels == class_id)[0]]

        # Display normalized grayscale pixels directly; move RGB channels last.
        if sample.shape[0] == 1:
            axes[index].imshow(sample[0], cmap='gray', vmin=0, vmax=1)
        else:
            axes[index].imshow(sample.transpose(1, 2, 0))
        axes[index].set_title(f'Class {class_id + label_offset}')

    fig.suptitle(f'{dataset_name}: one training image per class')
    fig.tight_layout()
    fig.savefig(paths[0], dpi=150)
    plt.close(fig)

    # Plot the number of training examples in each class.
    display_labels = classes + label_offset
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(display_labels, counts)
    ax.set_xticks(display_labels)
    ax.set_xlabel('Person ID' if dataset_name == 'ORL' else 'Class ID')
    ax.set_ylabel('Number of examples')
    ax.set_title(f'{dataset_name}: training class counts')
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(paths[1], dpi=150)
    plt.close(fig)
    return paths
