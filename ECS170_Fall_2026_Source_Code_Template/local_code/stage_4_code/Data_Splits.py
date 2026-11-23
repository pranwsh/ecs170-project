"""Create the shared training-only split IDs and select matching raw records."""

from sklearn.model_selection import train_test_split


def _split_ids(data, seed):
    # Identical texts stay together so copies cannot cross the holdout boundary.
    groups = {}
    for source_id, text in zip(data['ids'], data['X']):
        groups.setdefault(text, []).append(source_id)

    # Review copies share a sentiment label; jokes have no class labels to stratify.
    group_labels = None
    if 'y' in data:
        labels = dict(zip(data['X'], data['y']))
        group_labels = [labels[text] for text in groups]
    training, validation = train_test_split(
        list(groups), test_size=0.1, random_state=seed, stratify=group_labels,
    )
    return {'train': [source_id for text in training for source_id in groups[text]],
            'val': [source_id for text in validation for source_id in groups[text]]}


def create_split_ids(reviews, jokes, *, seed=2):
    """Hold out approximately 10% of each corpus, keeping identical texts together."""
    return {'seed': seed, 'reviews': _split_ids(reviews, seed), 'jokes': _split_ids(jokes, seed)}


def select_records(data, source_ids):
    """Select raw texts and matching labels in the saved split's ID order."""
    positions = {source_id: index for index, source_id in enumerate(data['ids'])}
    indices = [positions[source_id] for source_id in source_ids]
    selected = {'X': [data['X'][index] for index in indices], 'ids': list(source_ids)}
    if 'y' in data:
        selected['y'] = data['y'][indices]
    return selected
