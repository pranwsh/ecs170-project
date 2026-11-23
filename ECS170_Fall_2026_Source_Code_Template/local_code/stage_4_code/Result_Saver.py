"""Persist complete Stage 4 run artifacts without overwriting prior results."""
import json
from pathlib import Path

import torch


def save_results(output, *, config, vocabulary, split, history, metrics, model,
                 predictions=None, examples=None):
    """Save reproducibility metadata and weights. Output must not already exist."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    items = {'config': config, 'vocabulary': vocabulary, 'split': split,
             'history': history, 'metrics': metrics}
    if predictions is not None:
        items['predictions'] = predictions
    if examples is not None:
        items['examples'] = examples
    for name, value in items.items():
        (output / (name + '.json')).write_text(
            json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    torch.save(model.state_dict(), output / 'model.pt')
