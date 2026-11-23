"""Run one Stage 4 validation comparison or final review test fit."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from local_code.stage_4_code.Setting_Stage4 import Stage4Settings
from local_code.stage_4_code.Experiment_Data import prepare_data, make_loaders
from local_code.stage_4_code.Result_Saver import save_results
from local_code.stage_4_code.Training import (train_classification, train_generation,
                                                evaluate_classification, evaluate_generation)
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_4_code.Evaluate_Classification import classification_metrics
from local_code.stage_4_code.Convergence_Plots import save_convergence_plots


def run_experiment(settings, *, data_dir, handoff_dir, output):
    if Path(output).exists():
        raise FileExistsError(f'Output already exists: {output}')
    train, evaluation, vocabulary, split = prepare_data(settings, data_dir, handoff_dir)
    train_loader, eval_loader = make_loaders(settings, train, evaluation)
    cls = Method_RNN_Classification if settings.task == 'classification' else Method_RNN_Generation
    model = cls(len(vocabulary['id_to_token']), cell=settings.cell, embed_dim=settings.embed_dim,
                hidden_dim=settings.hidden_dim, num_layers=settings.num_layers, seed=settings.seed)
    model.to(settings.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    val = eval_loader if settings.mode == 'validation' else None
    if settings.task == 'classification':
        history = train_classification(model, train_loader, val, optimizer, settings.epochs, settings.device)
        metrics = evaluate_classification(model, eval_loader, settings.device)
        model.eval()
        predictions, labels = [], []
        with torch.no_grad():
            for batch in eval_loader:
                scores = model(batch['input_ids'].to(settings.device), batch['lengths'].to(settings.device))
                predictions.extend(scores.argmax(-1).cpu().tolist())
                labels.extend(batch['labels'].tolist())
        metrics.update(classification_metrics(labels, predictions))
        predictions = {'source_ids': split['evaluation_ids'], 'labels': labels, 'predictions': predictions}
    else:
        history = train_generation(model, train_loader, val, optimizer, settings.epochs, settings.device)
        metrics = evaluate_generation(model, eval_loader, settings.device)
        predictions = None
    config = settings.to_dict()
    config.update(vocabulary_size=len(vocabulary['id_to_token']),
                  parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  evaluated_partition=split['evaluated_partition'])
    save_results(output, config=config, vocabulary=vocabulary, split=split, history=history,
                 metrics=metrics, model=model, predictions=predictions)
    save_convergence_plots(history, Path(output) / 'learning_curves.png')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['classification', 'generation'], required=True)
    parser.add_argument('--cell', choices=['rnn', 'lstm', 'gru'], required=True)
    parser.add_argument('--mode', choices=['validation', 'test'], default='validation')
    for name, default in [('seed', 2), ('epochs', 5), ('batch-size', 64), ('embed-dim', 128),
                          ('hidden-dim', 256), ('num-layers', 1), ('review-length', 256)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/stage_4_data')
    parser.add_argument('--handoff-dir', type=Path, default=ROOT / 'result/stage_4_result/data_inspection')
    parser.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args())
    paths = {key: args.pop(key) for key in ('data_dir', 'handoff_dir', 'output')}
    print(run_experiment(Stage4Settings(**args), **paths))


if __name__ == '__main__':
    main()
