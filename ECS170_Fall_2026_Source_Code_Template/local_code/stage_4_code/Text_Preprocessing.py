"""Tokenize text and encode it using a training-only vocabulary."""

import html
import re
from collections import Counter


LINE_BREAK_PATTERN = re.compile(r'<br\s*/?>', re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"\w+(?:'\w+)*|[^\w\s]")


def tokenize(text):
    """Keep words, internal apostrophes, and separate punctuation tokens."""
    # Decode entities before removing line breaks, including escaped break tags.
    text = LINE_BREAK_PATTERN.sub(' ', html.unescape(text))
    text = text.lower().replace('\u2018', "'").replace('\u2019', "'")
    return TOKEN_PATTERN.findall(text)


def build_vocabulary(training_texts, *, max_tokens=20000, include_eos=False):
    """Build both ID mappings; max_tokens counts ordinary tokens only.

    Pass only training texts. For jokes, use max_tokens=None and include_eos=True.
    """
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError('max_tokens must be positive or None')

    # Count the supplied training text without reading any other partition.
    counts = Counter()
    for text in training_texts:
        counts.update(tokenize(text))

    # Frequency ordering favors common tokens; alphabetical ties keep IDs stable.
    ordinary_tokens = sorted(counts, key=lambda token: (-counts[token], token))
    if max_tokens is not None:
        ordinary_tokens = ordinary_tokens[:max_tokens]
    id_to_token = ['<PAD>', '<UNK>']
    if include_eos:
        id_to_token.append('<EOS>')
    id_to_token.extend(ordinary_tokens)
    token_to_id = {token: token_id for token_id, token in enumerate(id_to_token)}
    return {'token_to_id': token_to_id, 'id_to_token': id_to_token}


def encode(text, vocabulary, *, max_length=None, add_eos=False):
    """Return token IDs, optionally limiting text tokens before appending EOS.

    Reviews use max_length=256; jokes use add_eos=True with no length limit.
    """
    if max_length is not None and max_length <= 0:
        raise ValueError('max_length must be positive or None')

    # Apply truncation to this example without changing the fitted vocabulary.
    tokens = tokenize(text)
    if max_length is not None:
        tokens = tokens[:max_length]
    token_to_id = vocabulary['token_to_id']
    encoded = [token_to_id.get(token, token_to_id['<UNK>']) for token in tokens]

    # Recurrent models need at least one input token, even for a blank review.
    if not encoded:
        encoded = [token_to_id['<UNK>']]
    if add_eos:
        encoded.append(token_to_id['<EOS>'])
    return encoded
