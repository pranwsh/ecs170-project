"""Prepare shared text splits and vocabularies, and save data checks and figures."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Resolve project imports independently of the directory used to launch the script.
TEMPLATE_ROOT = Path(__file__).resolve().parents[2]
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from local_code.stage_4_code.Dataset_Loader_Jokes import Dataset_Loader_Jokes
from local_code.stage_4_code.Dataset_Loader_Reviews import Dataset_Loader_Reviews
from local_code.stage_4_code.Data_Splits import create_split_ids, select_records
from local_code.stage_4_code.Text_Preprocessing import (
    build_vocabulary, collate_jokes, collate_reviews, encode, tokenize,
)


def summarize_lengths(lengths):
    """Describe token counts before truncation, padding, or adding EOS."""
    return {'min': int(lengths.min()), 'mean': float(lengths.mean()),
            'median': float(np.median(lengths)), 'p95': float(np.percentile(lengths, 95)),
            'max': int(lengths.max()), 'total_tokens': int(lengths.sum())}


def token_coverage(texts, vocabulary):
    """Measure token occurrences covered by a fixed training vocabulary."""
    total, unknown = 0, 0
    for text in texts:
        tokens = tokenize(text)
        total += len(tokens)
        unknown += sum(token not in vocabulary['token_to_id'] for token in tokens)
    return {'total_tokens': total, 'unknown_tokens': unknown,
            'unknown_fraction': unknown / total, 'covered_fraction': (total - unknown) / total}


def save_figures(class_counts, review_lengths, joke_lengths, output_dir):
    # Counts and token-length plots use only supplied training text.
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(['Negative', 'Positive'], class_counts)
    axis.set_ylabel('Number of reviews')
    axis.set_title('Movie reviews: supplied training class counts')
    figure.tight_layout()
    figure.savefig(output_dir / 'review_class_counts.png', dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, lengths, title in zip(
            axes, (review_lengths, joke_lengths),
            ('Movie reviews: supplied training token lengths', 'Jokes: supplied corpus token lengths')):
        axis.hist(lengths, bins=50)
        axis.set_xlabel('Tokens before truncation or EOS')
        axis.set_ylabel('Number of texts')
        axis.set_title(title)
    axes[0].axvline(256, color='darkred', linestyle='--', label='256-token baseline')
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(output_dir / 'training_token_lengths.png', dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=TEMPLATE_ROOT / 'data' / 'stage_4_data')
    parser.add_argument('--output-dir', type=Path,
                        default=TEMPLATE_ROOT / 'result' / 'stage_4_result' / 'data_inspection')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Read supplied partitions separately; test reviews are counted, not tokenized.
    review_directory = args.data_dir / 'text_classification'
    reviews = Dataset_Loader_Reviews(data_dir=review_directory).load()
    test_reviews = Dataset_Loader_Reviews(split='test', data_dir=review_directory).load()
    test_details = {'count': len(test_reviews['X']),
                    'class_counts': np.bincount(test_reviews['y'], minlength=2).tolist()}
    del test_reviews
    jokes = Dataset_Loader_Jokes(data_dir=args.data_dir / 'text_generation').load()

    # Full-corpus descriptions are separate from the model's training/validation splits.
    review_lengths = np.asarray([len(tokenize(text)) for text in reviews['X']])
    joke_lengths = np.asarray([len(tokenize(text)) for text in jokes['X']])
    class_counts = np.bincount(reviews['y'], minlength=2).tolist()
    review_details = {'count': len(reviews['X']), 'class_counts': class_counts,
                      'token_lengths': summarize_lengths(review_lengths), 'truncation': {}}
    truncated = int(np.count_nonzero(review_lengths > 256))
    review_details['truncation']['256'] = {
        'count': truncated, 'fraction': truncated / len(review_lengths),
    }

    # Keep a short training example per sentiment and one original joke for reference.
    examples = []
    for label in (0, 1):
        index = int(np.flatnonzero(reviews['y'] == label)[0])
        examples.append({'source_id': reviews['ids'][index], 'label': label,
                         'text_preview': reviews['X'][index][:200],
                         'tokens_preview': tokenize(reviews['X'][index])[:24]})
    review_details['examples'] = examples
    joke_details = {'count': len(jokes['X']), 'token_lengths': summarize_lengths(joke_lengths),
                    'example': {'source_id': jokes['ids'][0], 'text_preview': jokes['X'][0][:200],
                                'tokens_preview': tokenize(jokes['X'][0])[:24]}}

    # Persist one shared split before fitting either task's training vocabulary.
    splits = create_split_ids(reviews, jokes)
    (args.output_dir / 'splits.json').write_text(json.dumps(splits, indent=2) + '\n', encoding='utf-8')
    summary = {'reviews': {'train': review_details, 'test': test_details},
               'jokes': {'supplied_corpus': joke_details},
               'vocabulary_checks': {'status': 'complete', 'split_file': 'splits.json'}}
    for task, data in (('reviews', reviews), ('jokes', jokes)):
        is_joke = task == 'jokes'
        parts = {split: select_records(data, splits[task][split]) for split in ('train', 'val')}
        vocabulary = build_vocabulary(parts['train']['X'], max_tokens=None if is_joke else 20000,
                                      include_eos=is_joke)
        name = 'joke_vocabulary.json' if is_joke else 'review_vocabulary.json'
        (args.output_dir / name).write_text(json.dumps(vocabulary, indent=2) + '\n', encoding='utf-8')
        prepared = {'vocabulary_size': len(vocabulary['id_to_token']), 'partitions': {}}

        # Prepare real batch examples and measure coverage without changing the fitted mapping.
        for split, part in parts.items():
            sequences = [encode(text, vocabulary, max_length=None if is_joke else 256,
                                add_eos=is_joke) for text in part['X'][:3]]
            batch = (collate_jokes(sequences) if is_joke else
                     collate_reviews(list(zip(sequences, part['y'][:3]))))
            prepared['partitions'][split] = {
                'count': len(part['X']), 'coverage': token_coverage(part['X'], vocabulary),
                'batch_example': {key: tensor.tolist() for key, tensor in batch.items()},
            }
        summary[task]['prepared'] = prepared

    save_figures(class_counts, review_lengths, joke_lengths, args.output_dir)
    (args.output_dir / 'dataset_checks.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8',
    )
    print(f'Reviews: train={review_details["count"]}, test={test_details["count"]}')
    print(f'Jokes: {joke_details["count"]}; shared splits, vocabularies, and batch examples prepared')
    print(f'Saved dataset checks and figures to {args.output_dir}')


if __name__ == '__main__':
    main()
