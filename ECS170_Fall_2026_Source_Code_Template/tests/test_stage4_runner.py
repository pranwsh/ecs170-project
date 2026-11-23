import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_code.stage_4_code.Setting_Stage4 import Stage4Settings
from local_code.stage_4_code.Text_Preprocessing import build_vocabulary
from script.stage_4_script.run_experiments import run_experiment


@pytest.mark.parametrize('task', ['classification', 'generation'])
@pytest.mark.parametrize('cell', ['rnn', 'lstm', 'gru'])
def test_runner_tiny_fixture(tmp_path, task, cell):
    data = tmp_path / 'data'
    handoff = tmp_path / 'handoff'
    handoff.mkdir()
    reviews = data / 'text_classification' / 'train'
    for sentiment, text in [('neg', 'bad film'), ('pos', 'good film')]:
        directory = reviews / sentiment
        directory.mkdir(parents=True)
        for n in range(2):
            (directory / f'{n}.txt').write_text(text)
    jokes = data / 'text_generation'
    jokes.mkdir()
    (jokes / 'data').write_text('ID,Joke\n1,one funny joke\n2,another funny joke\n')
    split = {'seed': 2, 'reviews': {'train': ['train/neg/0.txt', 'train/pos/0.txt'],
                                   'val': ['train/neg/1.txt', 'train/pos/1.txt']},
             'jokes': {'train': ['1'], 'val': ['2']}}
    (handoff / 'splits.json').write_text(json.dumps(split))
    for filename, vocab in [('review_vocabulary.json', build_vocabulary(['bad film', 'good film'])),
                            ('joke_vocabulary.json', build_vocabulary(['one funny joke'], max_tokens=None, include_eos=True))]:
        (handoff / filename).write_text(json.dumps(vocab))
    output = tmp_path / 'run'
    settings = Stage4Settings(task=task, cell=cell, epochs=1, batch_size=2, embed_dim=4, hidden_dim=5)
    metrics = run_experiment(settings, data_dir=data, handoff_dir=handoff, output=output)
    assert metrics['loss'] >= 0
    assert json.loads((output / 'config.json').read_text())['cell'] == cell
    assert (output / 'model.pt').exists()
    assert (output / 'learning_curves.png').exists()
    with pytest.raises(FileExistsError):
        run_experiment(settings, data_dir=data, handoff_dir=handoff, output=output)
