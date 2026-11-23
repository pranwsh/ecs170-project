"""Generate comparable greedy completions from held-out joke prefixes."""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_code.stage_4_code.Dataset_Loader_Jokes import Dataset_Loader_Jokes
from local_code.stage_4_code.Generate_Text import generate
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_4_code.Text_Preprocessing import tokenize

base = ROOT / 'result/stage_4_result'
source = Dataset_Loader_Jokes().load()
split = json.loads((base / 'data_inspection/splits.json').read_text())['jokes']
heldout = set(split['val'])
examples, seen = [], set()
for source_id, text in zip(source['ids'], source['X']):
    tokens = tokenize(text)
    prefix = tuple(tokens[:3])
    if source_id in heldout and len(tokens) >= 6 and prefix not in seen and all(t.isalpha() for t in prefix):
        seen.add(prefix)
        examples.append({'source_id': source_id, 'prompt': ' '.join(prefix), 'original': text})
        if len(examples) == 3:
            break
for cell in ('rnn', 'lstm', 'gru'):
    directory = base / f'validation_generation_{cell}'
    config = json.loads((directory / 'config.json').read_text())
    vocab = json.loads((directory / 'vocabulary.json').read_text())
    model = Method_RNN_Generation(len(vocab['id_to_token']), cell=cell, embed_dim=config['embed_dim'], hidden_dim=config['hidden_dim'], num_layers=config['num_layers'], seed=config['seed'])
    model.load_state_dict(torch.load(directory / 'model.pt', map_location='cpu', weights_only=True))
    output = [{**item, 'completion': generate(model, item['prompt'], vocab, max_new_tokens=30)} for item in examples]
    (directory / 'examples.json').write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
