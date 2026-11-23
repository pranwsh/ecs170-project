"""Finish the same three-word prompts w/ the trained RNN, LSTM and GRU joke models.

Run it after the three 20-epoch generation runs, e.g.
    python script/stage_4_script/run_experiments.py --task generation --cell lstm \
        --epochs 20 --output result/stage_4_result/validation_generation_lstm_20ep
"""

import argparse
import json
import sys
from pathlib import Path
import torch

# resolve project imports no matter where the script gets launched from
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_code.stage_4_code.Dataset_Loader_Jokes import Dataset_Loader_Jokes
from local_code.stage_4_code.Generate_Text import find_source_jokes, generate
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_4_code.Text_Preprocessing import tokenize

# same prompts for every model so the comparison is fair. train ones show if it just memorized,
# val ones are jokes it never trained on, and the new ones aren't in the data at all
PROMPTS = [
    ('why are bears', 'train'),
    ("what's a pirate's", 'train'),
    ('why does thor', 'train'),
    ('which cheese is', 'validation'),
    ('how do porcupines', 'validation'),
    ('why do java', 'validation'),
    ('my dog said', 'new'),
    ('why is pizza', 'new'),
    ('a robot walks', 'new'),
]

# sharing a run of words this long w/ a training joke (past the prompt) counts as partly copied
COPY_RUN = 6

def longest_shared_run(a, b):
    """Length of the longest stretch of tokens that shows up in both lists, in order."""
    best = 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best

def build_table(rows, cells):
    """Markdown table w/ one row per prompt and one column per model, ready for the report."""
    lines = ['| Prompt | Source | Original joke | ' + ' | '.join(c.upper() for c in cells) + ' |',
             '|' + ' --- |' * (3 + len(cells))]
    for row in rows:
        # escape pipes so a joke w/ a | in it doesn't break the table
        outputs = [f"{row['outputs'][c]['text']} *({row['outputs'][c]['label']})*".replace('|', '\\|')
                   for c in cells]
        original = (row['original'] or '-').replace('|', '\\|')
        lines.append(f"| {row['prompt']} | {row['source']} | {original} | " + ' | '.join(outputs) + ' |')
    return '\n'.join(lines) + '\n'

def check_copying(text, train_tokens):
    """Label a completion as copied, partly copied or new compared to the training jokes."""
    tokens = tokenize(text)
    if tokens in train_tokens:
        return 'copied', len(tokens)

    # skip the prompt, otherwise every dataset prompt looks like a copy. only count real words too,
    # bc a model spamming "....." would otherwise "match" jokes that have "..."
    words = [t for t in tokens[3:] if any(ch.isalnum() for ch in t)]
    longest = max((longest_shared_run(words, [t for t in joke if any(ch.isalnum() for ch in t)])
                   for joke in train_tokens), default=0)
    return ('partly copied' if longest >= COPY_RUN else 'new'), longest

def load_model(run_dir):
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    vocab = json.loads((run_dir / 'vocabulary.json').read_text(encoding='utf-8'))
    if config['task'] != 'generation':
        raise ValueError(f'{run_dir} is a {config["task"]} run, not a generation run')

    model = Method_RNN_Generation(len(vocab['id_to_token']), cell=config['cell'],
                                  embed_dim=config['embed_dim'], hidden_dim=config['hidden_dim'],
                                  num_layers=config['num_layers'], seed=config['seed'])
    model.load_state_dict(torch.load(run_dir / 'model.pt', map_location='cpu'))
    return model, vocab, config

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runs', type=Path, nargs='+', default=[
        ROOT / f'result/stage_4_result/validation_generation_{cell}_20ep' for cell in ('rnn', 'lstm', 'gru')])
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/stage_4_data/text_generation')
    parser.add_argument('--max-new-tokens', type=int, default=30)
    parser.add_argument('--output', type=Path, default=ROOT / 'result/stage_4_result/generation_prompt_comparison')
    args = parser.parse_args()

    jokes = Dataset_Loader_Jokes(data_dir=args.data_dir).load()
    models = {}
    train_ids = None
    for run_dir in args.runs:
        model, vocab, config = load_model(run_dir)
        models[config['cell']] = (model, vocab)

        # every run has to share the same training split, otherwise the copy check means nothing
        split = json.loads((run_dir / 'split.json').read_text(encoding='utf-8'))
        if train_ids is not None and split['train_ids'] != train_ids:
            raise ValueError(f'{run_dir} was trained on a different split than the other runs')
        train_ids = split['train_ids']

    train_set = set(train_ids)
    train_tokens = [tokenize(text) for text, joke_id in zip(jokes['X'], jokes['ids'])
                    if joke_id in train_set]

    rows = []
    for prompt, source in PROMPTS:
        # grab the original joke to compare against. if a few jokes start the same way, keep the first
        # and record how many matched
        matches = find_source_jokes(prompt, jokes['X'], jokes['ids'])
        if source != 'new' and not matches:
            raise ValueError(f'no joke in the dataset starts with {prompt!r}')
        row = {'prompt': prompt, 'source': source, 'match_count': len(matches),
               'original_id': matches[0][0] if matches else None,
               'original': matches[0][1] if matches else None, 'outputs': {}}

        for cell, (model, vocab) in models.items():
            text = generate(model, prompt, vocab, max_new_tokens=args.max_new_tokens)
            label, shared = check_copying(text, train_tokens)
            row['outputs'][cell] = {'text': text, 'label': label, 'longest_shared_run': shared}
        rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    # keep run paths relative to the project so the saved file doesn't depend on whose machine ran it
    runs = [str(r.resolve().relative_to(ROOT)) if r.resolve().is_relative_to(ROOT) else str(r)
            for r in args.runs]
    summary = {'prompts': len(rows), 'copy_run_threshold': COPY_RUN,
               'max_new_tokens': args.max_new_tokens, 'runs': runs, 'examples': rows}

    # save the full data plus a table that can go straight into the report
    (args.output / 'examples.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (args.output / 'examples.md').write_text(build_table(rows, list(models)), encoding='utf-8')
    print(build_table(rows, list(models)))

if __name__ == '__main__':
    main()
