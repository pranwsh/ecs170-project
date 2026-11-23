"""Load authoritative Stage 4 splits and encode partitions without leakage."""
import hashlib
import json
from pathlib import Path

from torch.utils.data import DataLoader

from .Data_Splits import select_records
from .Dataset_Loader_Jokes import Dataset_Loader_Jokes
from .Dataset_Loader_Reviews import Dataset_Loader_Reviews
from .Text_Preprocessing import build_vocabulary, collate_jokes, collate_reviews, encode


def prepare_data(settings, data_dir, handoff_dir):
    """Return encoded train/evaluation examples, vocabulary, and split identity.

    Validation runs use the saved training-only vocabulary, never fitting one here.
    Test runs fit reviews' vocabulary only on all supplied training reviews.
    """
    data_dir, handoff_dir = Path(data_dir), Path(handoff_dir)
    split_bytes = (handoff_dir / 'splits.json').read_bytes()
    splits = json.loads(split_bytes)
    task_key = 'reviews' if settings.task == 'classification' else 'jokes'
    if settings.task == 'classification':
        directory = data_dir / 'text_classification'
        source = Dataset_Loader_Reviews(data_dir=directory).load()
    else:
        source = Dataset_Loader_Jokes(data_dir=data_dir / 'text_generation').load()
    ids = splits[task_key]
    train_ids, val_ids = ids['train'], ids['val']
    if (len(set(source['ids'])) != len(source['ids']) or
            len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids) or
            set(train_ids) & set(val_ids) or set(train_ids) | set(val_ids) != set(source['ids'])):
        raise ValueError('saved split IDs must uniquely partition the supplied training records')
    if settings.mode == 'validation':
        train, evaluation = select_records(source, train_ids), select_records(source, val_ids)
        vocab_name = 'review_vocabulary.json' if task_key == 'reviews' else 'joke_vocabulary.json'
        vocabulary = json.loads((handoff_dir / vocab_name).read_text(encoding='utf-8'))
        partition = 'validation'
    else:
        train = source
        evaluation = Dataset_Loader_Reviews(split='test', data_dir=directory).load()
        vocabulary = build_vocabulary(train['X'], max_tokens=20000)
        partition = 'supplied_test'
    if settings.task == 'classification':
        def convert(records):
            return [(encode(text, vocabulary, max_length=settings.review_length), int(label))
                    for text, label in zip(records['X'], records['y'])]
    else:
        def convert(records):
            return [encode(text, vocabulary, add_eos=True) for text in records['X']]
    identity = {'source': str(handoff_dir / 'splits.json'),
                'sha256': hashlib.sha256(split_bytes).hexdigest(), 'split_seed': splits['seed'],
                'train_ids': train['ids'], 'evaluation_ids': evaluation['ids'],
                'evaluated_partition': partition}
    return convert(train), convert(evaluation), vocabulary, identity


def make_loaders(settings, train, evaluation):
    import torch
    collate = collate_reviews if settings.task == 'classification' else collate_jokes
    generator = torch.Generator().manual_seed(settings.seed)
    return (DataLoader(train, batch_size=settings.batch_size, shuffle=True,
                       collate_fn=collate, generator=generator),
            DataLoader(evaluation, batch_size=settings.batch_size, shuffle=False,
                       collate_fn=collate))
