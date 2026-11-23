"""Generate report-ready Markdown from saved runs, without training.

Run from the template root: python -m script.stage_3_script.summarize_experiments
"""
import argparse
import json
from pathlib import Path

DEFAULT_ROOT = Path('result/stage_3_result/experiments')
METRICS = ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro')
SETTINGS = ('seed', 'split_seed', 'epochs', 'batch_size', 'learning_rate',
            'device', 'train_count', 'evaluation_count')


def load_runs(root):
    """Read all run artifacts, rejecting incomplete or inconsistent runs."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'Experiment directory does not exist: {root}')
    directories = sorted({p.parent for name in ('config.json', 'metrics.json')
                          for p in root.rglob(name)})
    if not directories:
        raise ValueError(f'No experiment runs found in {root}')
    runs = []
    for directory in directories:
        try:
            config = json.loads((directory / 'config.json').read_text())
            metrics = json.loads((directory / 'metrics.json').read_text())
            for key in ('dataset', 'mode', 'variant', *SETTINGS):
                config[key]
            for key in (*METRICS, 'correct', 'total'):
                metrics[key]
            if config['mode'] not in ('validation', 'test'):
                raise ValueError(f"unknown mode {config['mode']!r}")
            if metrics['total'] != config['evaluation_count']:
                raise ValueError('metric total differs from evaluation_count')
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            raise ValueError(f'Invalid run {directory}: {exc}') from exc
        runs.append({'path': directory.relative_to(root).as_posix(),
                     'config': config, 'metrics': metrics})
    return runs


def _table(headers, rows):
    def cell(value):
        return str(value).replace('|', r'\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join('---' for _ in headers) + ' |']
    lines.extend('| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows)
    return '\n'.join(lines)


def render_summary(runs):
    lines = ['# Stage 3 experiment summary', '',
             'Metrics are fractions (0–1); precision, recall and F1 are macro-averaged.',
             'Validation is for model selection; test results are final evaluations and must not be used to select variants.', '']
    for mode, heading in (('validation', 'Validation ablations'), ('test', 'Final test evaluations')):
        selected = [run for run in runs if run['config']['mode'] == mode]
        lines += [f'## {heading}', '']
        if not selected:
            lines += ['No runs found.', '']
            continue
        rows = []
        for run in selected:
            c, m = run['config'], run['metrics']
            rows.append([c['dataset'], c['variant'], run['path'],
                         *(f"{m[k]:.4f}" for k in METRICS), f"{m['correct']}/{m['total']}"])
        lines += [_table(['Dataset', 'Variant', 'Run', 'Accuracy', 'Macro precision',
                          'Macro recall', 'Macro F1', 'Correct / total'], rows), '']
    lines += ['## Run settings', '',
              _table(['Run', 'Dataset', 'Split', 'Variant', 'Model options', 'Seed',
                      'Split seed', 'Epochs', 'Batch size', 'Learning rate', 'Device',
                      'Train count', 'Evaluation count'],
                     [[r['path'], r['config']['dataset'], r['config']['mode'],
                       r['config']['variant'], json.dumps(r['config'].get('model_options', {}), sort_keys=True),
                       *(r['config'][k] for k in SETTINGS)] for r in runs]), '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, help='Write Markdown here instead of stdout')
    args = parser.parse_args(argv)
    try:
        summary = render_summary(load_runs(args.root))
    except ValueError as exc:
        parser.error(str(exc))
    if args.output:
        args.output.write_text(summary, encoding='utf-8')
    else:
        print(summary, end='')


if __name__ == '__main__':
    main()
