"""Compare review-token retention using the saved split and training vocabulary."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Use the same project imports regardless of the script's launch directory.
TEMPLATE_ROOT = Path(__file__).resolve().parents[2]
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from local_code.stage_4_code.Data_Splits import select_records
from local_code.stage_4_code.Dataset_Loader_Reviews import Dataset_Loader_Reviews
from local_code.stage_4_code.Text_Preprocessing import tokenize


TOKEN_LIMITS = {'128': 128, '256': 256, 'unlimited': None}


def compare_lengths(texts, vocabulary):
    """Compare capped and full reviews with identical texts and vocabulary."""
    results = {name: {'truncated_count': 0, 'retained_tokens': 0, 'unknown_tokens': 0}
               for name in TOKEN_LIMITS}
    original_tokens = 0
    for text in texts:
        tokens = tokenize(text)
        original_tokens += len(tokens)
        for name, limit in TOKEN_LIMITS.items():
            # A slice ending at None retains the complete token sequence.
            retained = tokens[:limit]
            details = results[name]
            details['truncated_count'] += int(limit is not None and len(tokens) > limit)
            details['retained_tokens'] += len(retained)
            details['unknown_tokens'] += sum(token not in vocabulary['token_to_id'] for token in retained)

    # Fractions use review counts, original tokens, and retained tokens respectively.
    for details in results.values():
        details.update({'review_count': len(texts), 'original_tokens': original_tokens,
                        'truncated_fraction': details['truncated_count'] / len(texts),
                        'retained_fraction': details['retained_tokens'] / original_tokens,
                        'unknown_fraction': details['unknown_tokens'] / details['retained_tokens']})
    return results


def save_figure(partitions, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    positions = np.arange(len(TOKEN_LIMITS))
    for offset, split in ((-0.18, 'train'), (0.18, 'val')):
        values = [partitions[split][name] for name in TOKEN_LIMITS]
        axes[0].bar(positions + offset, [100 * value['truncated_fraction'] for value in values],
                    width=0.36, label=split)
        axes[1].bar(positions + offset, [100 * value['retained_fraction'] for value in values],
                    width=0.36, label=split)
    for axis, title in zip(axes, ('Reviews truncated', 'Original tokens retained')):
        axis.set_xticks(positions, ['128 tokens', '256 tokens', 'No truncation'])
        axis.set_ylim(0, 100)
        axis.set_ylabel('Percent')
        axis.set_title(title)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / 'review_length_comparison.png', dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=TEMPLATE_ROOT / 'data' / 'stage_4_data')
    parser.add_argument('--prepared-dir', type=Path,
                        default=TEMPLATE_ROOT / 'result' / 'stage_4_result' / 'data_inspection')
    parser.add_argument('--output-dir', type=Path,
                        default=TEMPLATE_ROOT / 'result' / 'stage_4_result' / 'review_length_comparison')
    args = parser.parse_args()

    # Load the same saved split and training vocabulary for all three cases.
    splits = json.loads((args.prepared_dir / 'splits.json').read_text(encoding='utf-8'))
    vocabulary = json.loads((args.prepared_dir / 'review_vocabulary.json').read_text(encoding='utf-8'))
    reviews = Dataset_Loader_Reviews(data_dir=args.data_dir / 'text_classification').load()
    partitions = {}
    for split in ('train', 'val'):
        records = select_records(reviews, splits['reviews'][split])
        partitions[split] = compare_lengths(records['X'], vocabulary)

    # Save the comparison summary and figure in their own output directory.
    summary = {'split_seed': splits['seed'], 'split_file': 'splits.json',
               'vocabulary_file': 'review_vocabulary.json',
               'vocabulary_size': len(vocabulary['id_to_token']), 'partitions': partitions}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    save_figure(partitions, args.output_dir)
    print(f'Saved review-length retention comparison to {args.output_dir}')


if __name__ == '__main__':
    main()
