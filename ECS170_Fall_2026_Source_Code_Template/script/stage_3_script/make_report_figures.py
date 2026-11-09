"""Make the two Stage 3 report figures from saved runs; run from the template root:

    python -m script.stage_3_script.make_report_figures

Writes cnn_architectures.png and convergence_all_datasets.png to result/stage_3_result/report/figures/.
The convergence panels use the final model's training loss and the matching validation run's
validation loss (MNIST and ORL from experiments/, CIFAR from report_runs/).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RESULTS = Path(__file__).resolve().parents[2] / 'result' / 'stage_3_result'
OUT = RESULTS / 'report' / 'figures'

# (dataset, final run, matching validation run, label); kernel size shown in the architecture figure
CURVES = [('MNIST', 'experiments/mnist/final', 'experiments/mnist/kernel5', '5x5 kernels'),
          ('ORL', 'experiments/orl/final', 'experiments/orl/baseline', 'baseline, seed 2'),
          ('CIFAR-10', 'report_runs/cifar/final', 'report_runs/cifar/kernel5', '5x5 kernels')]
KERNELS = {'MNIST': 5, 'ORL': 3, 'CIFAR-10': 5}
CONV, POOL, FC, IO = '#d4e4f7', '#f7e6d4', '#dff0d8', '#e8eef7'


def architectures(path):
    km, ko, kc = KERNELS['MNIST'], KERNELS['ORL'], KERNELS['CIFAR-10']
    rows = [
        ('MNIST', [('Input\n1x28x28', IO), (f'Conv {km}x{km}\n16 + ReLU', CONV), ('MaxPool 2\n16x14x14', POOL),
                   (f'Conv {km}x{km}\n32 + ReLU', CONV), ('MaxPool 2\n32x7x7', POOL),
                   ('Flatten\n1,568', IO), ('Linear 64\n+ ReLU', FC), ('Linear 10\nlogits', FC)]),
        ('ORL', [('Input\n1x112x92', IO), (f'Conv {ko}x{ko}\n16 + ReLU', CONV), ('MaxPool 2\n16x56x46', POOL),
                 (f'Conv {ko}x{ko}\n32 + ReLU', CONV), ('MaxPool 2\n32x28x23', POOL),
                 ('AdaptiveAvgPool\n32x7x6', POOL), ('Linear 128\n+ ReLU', FC), ('Linear 40\nlogits', FC)]),
        ('CIFAR-10', [('Input\n3x32x32', IO), (f'Conv {kc}x{kc}\n32 + ReLU', CONV), ('MaxPool 2\n32x16x16', POOL),
                      (f'Conv {kc}x{kc}\n64 + ReLU', CONV), ('MaxPool 2\n64x8x8', POOL),
                      (f'Conv {kc}x{kc}\n128 + ReLU', CONV), ('MaxPool 2\n128x4x4', POOL),
                      ('Linear 128 + ReLU\n-> Linear 10', FC)]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 6.2))
    for ax, (name, boxes) in zip(axes, rows):
        ax.set_xlim(-1.1, len(boxes) * 1.5 + 0.35)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(-1.05, 0.5, name, ha='left', va='center', fontsize=11, weight='bold')
        for i, (label, color) in enumerate(boxes):
            x = i * 1.5 + 0.35
            ax.add_patch(FancyBboxPatch((x, 0.15), 1.15, 0.7, boxstyle='round,pad=0.03',
                                        facecolor=color, edgecolor='#333333'))
            ax.text(x + 0.575, 0.5, label, ha='center', va='center', fontsize=8)
            if i < len(boxes) - 1:
                ax.annotate('', xy=(x + 1.5, 0.5), xytext=(x + 1.2, 0.5),
                            arrowprops=dict(arrowstyle='->', lw=1.2))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def convergence(path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (name, final, val, label) in zip(axes, CURVES):
        train = json.loads((RESULTS / final / 'history.json').read_text())['train_loss']
        valid = json.loads((RESULTS / val / 'history.json').read_text())['val_loss']
        ax.plot(range(1, len(train) + 1), train, marker='o', ms=3,
                label='training loss (final model, full training set)')
        ax.plot(range(1, len(valid) + 1), valid, marker='s', ms=3, ls='--',
                label='validation loss (validation run)')
        ax.set_title(f'{name} ({label})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cross-entropy loss')
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    architectures(OUT / 'cnn_architectures.png')
    convergence(OUT / 'convergence_all_datasets.png')
    print(f'saved figures to {OUT}')
