"""Stage 2 MLP pipeline: load data, train, evaluate, and save everything for one run.

Run from the template root:
    python -m script.stage_2_script.script_mlp                    # final run on test.csv
    python -m script.stage_2_script.script_mlp --validate --name lr_1e-2 --lr 1e-2
    python -m script.stage_2_script.script_mlp --validate --name deep --hidden 512,256,128

Each run gets its own folder in result/stage_2_result/<name>/ with the config,
metrics, saved predictions, convergence plots, confusion matrix, and error analysis.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

# let the script run from PyCharm or the terminal without changing the working directory
TEMPLATE_ROOT = Path(__file__).resolve().parents[2]
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Stage2 import Setting_Stage2
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code import Error_Analysis


RESULT_ROOT = TEMPLATE_ROOT / 'result' / 'stage_2_result'

OPTIMIZERS = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD, 'rmsprop': torch.optim.RMSprop}

# Baseline config everyone uses unless their experiment changes one of these.
# Defaults match Method_MLP's own defaults.
BASELINE = {
    'seed': 2,
    'normalize': True,
    'hidden_dims': [256, 128],
    'dropout': 0.0,
    'optimizer': 'adam',
    'learning_rate': 1e-3,
    'batch_size': 128,
    'max_epoch': 20,
    'device': 'cpu',
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Stage 2 MLP pipeline')
    parser.add_argument('--name', default=None, help='run folder name (default: baseline / baseline_validation)')
    parser.add_argument('--validate', action='store_true', help='evaluate on 20%% of train.csv instead of test.csv')
    parser.add_argument('--seed', type=int, default=BASELINE['seed'])
    parser.add_argument('--no-normalize', action='store_true', help='keep raw 0-255 pixels')
    parser.add_argument('--hidden', default=','.join(map(str, BASELINE['hidden_dims'])),
                        help='hidden layer sizes, e.g. 512,256,128')
    parser.add_argument('--dropout', type=float, default=BASELINE['dropout'])
    parser.add_argument('--optimizer', choices=OPTIMIZERS, default=BASELINE['optimizer'])
    parser.add_argument('--lr', type=float, default=BASELINE['learning_rate'])
    parser.add_argument('--batch-size', type=int, default=BASELINE['batch_size'])
    parser.add_argument('--epochs', type=int, default=BASELINE['max_epoch'])
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=BASELINE['device'])
    return parser.parse_args()


def save_convergence_plots(history, path):
    """Loss and accuracy per epoch; validation curves are included when they exist."""
    epochs = np.arange(1, len(history['train_loss']) + 1)
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(10, 4))

    # Plotting loss
    loss_ax.plot(epochs, history['train_loss'], marker='o', markersize=3, label='train')
    if history.get('val_loss'):
        loss_ax.plot(epochs, history['val_loss'], marker='o', markersize=3, label='validation')
    loss_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Loss')
    loss_ax.set_title('Training loss')
    loss_ax.grid(alpha=0.3)
    loss_ax.legend()

    # Plotting accuracy separately just to be safe with loop structures
    if 'train_acc' in history:
        acc_ax.plot(epochs, history['train_acc'], marker='o', markersize=3, label='train')
    if history.get('val_acc'):
        acc_ax.plot(epochs, history['val_acc'], marker='o', markersize=3, label='validation')
    acc_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    acc_ax.set_xlabel('Epoch')
    acc_ax.set_ylabel('Accuracy')
    acc_ax.set_title('Training accuracy')
    acc_ax.grid(alpha=0.3)
    acc_ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    config = {
        'seed': args.seed,
        'normalize': not args.no_normalize,
        'hidden_dims': [int(h) for h in args.hidden.split(',') if h.strip()],
        'dropout': args.dropout,
        'optimizer': args.optimizer,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'max_epoch': args.epochs,
        'device': args.device,
        'evaluated_on': 'validation (20% of train.csv)' if args.validate else 'test.csv',
    }
    name = args.name or ('baseline_validation' if args.validate else 'baseline')
    run_dir = RESULT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)

    #---- parameter section -------------------------------
    set_seed(config['seed'])
    #------------------------------------------------------

    # ---- object initialization section -------------------
    train_obj = Dataset_Loader('digits (train)', 'train.csv', split='train', normalize=config['normalize'])
    test_obj = Dataset_Loader('digits (test)', 'test.csv', split='test', normalize=config['normalize'])

    method_obj = Method_MLP('multi-layer perceptron', '', hidden_dims=tuple(config['hidden_dims']),
                            dropout=config['dropout'])
    method_obj.optimizer_class = OPTIMIZERS[config['optimizer']]
    method_obj.learning_rate = config['learning_rate']
    method_obj.batch_size = config['batch_size']
    method_obj.max_epoch = config['max_epoch']
    method_obj.device = config['device']

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = run_dir
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Stage2('provided train/test split', '')
    setting_obj.use_validation = args.validate
    setting_obj.seed = config['seed']

    evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    print('config:', config)
    setting_obj.prepare(train_obj, method_obj, result_obj, evaluate_obj, test_obj)
    setting_obj.print_setup_summary()
    start = time.time()
    scores, _ = setting_obj.load_run_save_evaluate()
    config['run_seconds'] = round(time.time() - start, 1)

    if not isinstance(scores, dict):
        scores = {'accuracy': scores}

    # Clean up scores dictionary
    temp_scores = {}
    for key, value in scores.items():
        temp_scores[key] = float(value)
    scores = temp_scores

    print('************ Overall Performance ************')
    for key, value in scores.items():
        print(f'{key}: {value:.4f}')

    # ---- save outputs ------------------------------------
    learned = setting_obj.learned_result
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (run_dir / 'metrics.json').write_text(json.dumps(scores, indent=2) + '\n')

    if learned['history'].get('train_loss'):
        save_convergence_plots(learned['history'], run_dir / 'convergence.png')
    else:
        print('note: Method_MLP did not record a training history, skipping convergence plots')

    cm = Error_Analysis.compute_confusion_matrix(learned['true_y'], learned['pred_y'])
    np.savetxt(run_dir / 'confusion_matrix.csv', cm, fmt='%d', delimiter=',')
    Error_Analysis.save_confusion_matrix(cm, run_dir / 'confusion_matrix.png',
                                         title=f'Confusion matrix ({config["evaluated_on"]})')
    Error_Analysis.save_misclassified_examples(setting_obj.eval_data['X'], learned['true_y'],
                                               learned['pred_y'], run_dir / 'misclassified_examples.png')
    print(Error_Analysis.write_report(cm, run_dir / 'error_analysis.txt'))
    print(f'saved run to {run_dir}')
    print('************ Finish ************')
    # ------------------------------------------------------


if __name__ == '__main__':
    main()
