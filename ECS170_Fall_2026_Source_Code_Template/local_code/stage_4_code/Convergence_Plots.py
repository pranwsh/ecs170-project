"""Reusable learning curves for classification and generation histories."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def save_convergence_plots(history, path, title='', dpi=150):
    """Plot loss and, when recorded, accuracy; final fits may omit validation."""
    if len(history.get('train_loss', [])) == 0:
        raise ValueError('history has no train_loss')
    metrics = [('loss', 'Loss')]
    if len(history.get('train_acc', [])) or len(history.get('val_acc', [])):
        metrics.append(('acc', 'Accuracy'))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.2), squeeze=False)
    try:
        for ax, (key, label) in zip(axes[0], metrics):
            for split in ('train', 'val'):
                values = history.get(f'{split}_{key}', [])
                if len(values):
                    ax.plot(range(1, len(values) + 1), values, marker='o', markersize=3,
                            label='train' if split == 'train' else 'validation')
            ax.set(xlabel='Epoch', ylabel=label, title=f'{label} per epoch')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.3)
            ax.legend()
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        return path
    finally:
        plt.close(fig)
