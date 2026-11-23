"""Tests for the joke generation model and three-word decoding."""

import unittest

import torch
from torch import nn

from local_code.stage_4_code.Generate_Text import find_source_jokes, generate, join_tokens
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generation
from local_code.stage_4_code.Text_Preprocessing import build_vocabulary, collate_jokes, encode


JOKES = [
    'why did the chicken cross the road ? to get to the other side .',
    'what do you call a fish with no eyes ? a fsh .',
    'knock knock who is there ? lettuce .',
]


def small_model(cell, vocab_size=20, seed=2):
    return Method_RNN_Generation(vocab_size, cell=cell, embed_dim=8, hidden_dim=16, seed=seed)


class ModelShapeTests(unittest.TestCase):
    def test_every_cell_returns_one_score_per_word_per_position(self):
        input_ids = torch.randint(1, 20, (4, 7))
        lengths = torch.full((4,), 7)
        for cell in ('rnn', 'lstm', 'gru'):
            with self.subTest(cell=cell):
                model = small_model(cell)
                self.assertIsInstance(model.rnn, {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[cell])
                self.assertFalse(model.rnn.bidirectional)
                self.assertEqual(model(input_ids, lengths).shape, (4, 7, 20))

    def test_unknown_cell_is_rejected(self):
        with self.assertRaises(ValueError):
            small_model('transformer')

    def test_same_seed_gives_same_weights(self):
        first, second = small_model('gru', seed=5), small_model('gru', seed=5)
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_later_words_cannot_change_earlier_scores(self):
        model = small_model('lstm').eval()
        base = torch.tensor([[3, 4, 5, 6, 7]])
        changed = base.clone()
        changed[0, 3:] = torch.tensor([9, 10])
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(base)[0, :3], model(changed)[0, :3]))

    def test_right_padding_does_not_change_real_positions(self):
        model = small_model('rnn').eval()
        short = torch.tensor([[3, 4, 5]])
        padded = torch.tensor([[3, 4, 5, 0, 0]])
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(short)[0], model(padded)[0, :3]))


class LossAndTrainingTests(unittest.TestCase):
    def setUp(self):
        self.vocab = build_vocabulary(JOKES, max_tokens=None, include_eos=True)
        self.batch = collate_jokes([encode(text, self.vocab, add_eos=True) for text in JOKES])

    def test_padded_targets_do_not_count_toward_the_loss(self):
        model = small_model('gru', vocab_size=len(self.vocab['id_to_token']))
        scores = model(self.batch['input_ids'], self.batch['lengths'])
        targets = self.batch['target_ids']
        loss = nn.CrossEntropyLoss(ignore_index=0)(scores.flatten(0, 1), targets.flatten())

        # rebuild the same number by hand using only the real (non-pad) targets
        real = targets.flatten() != 0
        manual = nn.functional.cross_entropy(scores.flatten(0, 1)[real], targets.flatten()[real])
        self.assertTrue(torch.allclose(loss, manual))

    def test_tiny_model_memorizes_and_finishes_a_training_joke(self):
        torch.manual_seed(0)
        model = Method_RNN_Generation(len(self.vocab['id_to_token']), cell='lstm',
                                      embed_dim=32, hidden_dim=64, seed=0)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(300):
            optimizer.zero_grad()
            scores = model(self.batch['input_ids'], self.batch['lengths'])
            loss = loss_fn(scores.flatten(0, 1), self.batch['target_ids'].flatten())
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), 0.05)
        self.assertEqual(generate(model, 'knock knock who', self.vocab),
                         'knock knock who is there? lettuce.')


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.vocab = build_vocabulary(JOKES, max_tokens=None, include_eos=True)
        self.model = small_model('rnn', vocab_size=len(self.vocab['id_to_token']))

    def test_prompt_must_be_exactly_three_tokens(self):
        for prompt in ('why did', 'why did the chicken', ''):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                generate(self.model, prompt, self.vocab)

    def test_never_emits_pad_or_unk_and_respects_the_limit(self):
        # push all the weight onto <PAD> and <UNK> so the model really wants to pick them
        with torch.no_grad():
            self.model.output.bias.fill_(-10)
            self.model.output.bias[0] = 50
            self.model.output.bias[1] = 40
        text = generate(self.model, 'why did the', self.vocab, max_new_tokens=5)
        self.assertNotIn('<PAD>', text)
        self.assertNotIn('<UNK>', text)
        self.assertLessEqual(len(text.split()), 3 + 5)

    def test_stops_as_soon_as_eos_is_predicted(self):
        with torch.no_grad():
            self.model.output.bias.fill_(-10)
            self.model.output.bias[self.vocab['token_to_id']['<EOS>']] = 50
        self.assertEqual(generate(self.model, 'why did the', self.vocab), 'why did the')

    def test_unseen_prompt_words_still_work(self):
        text = generate(self.model, 'zebras eat pizza', self.vocab, max_new_tokens=3)
        self.assertTrue(text.startswith('zebras eat pizza'))

    def test_join_tokens_attaches_punctuation(self):
        self.assertEqual(join_tokens(['why', '?', 'to', 'win', ',', 'duh', '!']), 'why? to win, duh!')

    def test_find_source_jokes_returns_every_match(self):
        texts = JOKES + ['Knock knock, who...']
        self.assertEqual(find_source_jokes('knock knock who', texts, ['1', '2', '3', '4']),
                         [('3', JOKES[2])])
        self.assertEqual(find_source_jokes('why did the', texts, ['1', '2', '3', '4']),
                         [('1', JOKES[0])])


if __name__ == '__main__':
    unittest.main()
