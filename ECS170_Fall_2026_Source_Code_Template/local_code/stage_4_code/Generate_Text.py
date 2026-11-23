"""Finish a joke from three starting words, one predicted word at a time."""

import torch

from local_code.stage_4_code.Text_Preprocessing import tokenize


# no space goes before these when we turn tokens back into a sentence
NO_SPACE_BEFORE = {'.', ',', '!', '?', ';', ':', ')', '%'}


def join_tokens(tokens):
    """Glue tokens back into readable text, attaching punctuation to the word before it."""
    text = ''
    for token in tokens:
        if text and token not in NO_SPACE_BEFORE:
            text += ' '
        text += token
    return text


@torch.no_grad()
def generate(model, prompt, vocabulary, max_new_tokens=30):
    """Greedy completion: always pick the most likely next word until <EOS> or the limit.

    The prompt has to be exactly three tokens after tokenizing (same tokenizer as training).
    Returns the whole joke, prompt included, as readable text.
    """
    prompt_tokens = tokenize(prompt)
    if len(prompt_tokens) != 3:
        raise ValueError(f'prompt needs exactly 3 tokens, got {len(prompt_tokens)}: {prompt_tokens}')
    if max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be positive')

    token_to_id = vocabulary['token_to_id']
    id_to_token = vocabulary['id_to_token']
    unk_id, eos_id = token_to_id['<UNK>'], token_to_id['<EOS>']

    # unseen prompt words become <UNK>, same as validation text does
    ids = [token_to_id.get(token, unk_id) for token in prompt_tokens]
    device = next(model.parameters()).device

    # eval mode + no_grad so nothing gets updated while we're generating
    model.eval()
    new_tokens = []
    for _ in range(max_new_tokens):
        # we just rerun the whole growing prefix each step, it's short so caching isn't worth it
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        scores = model(input_ids, torch.tensor([len(ids)]))[0, -1].clone()

        # never let it pick <PAD>, and skip <UNK> too so the output doesn't fill up w/ blanks
        scores[token_to_id['<PAD>']] = float('-inf')
        scores[unk_id] = float('-inf')

        next_id = int(scores.argmax())
        if next_id == eos_id:
            break
        ids.append(next_id)
        new_tokens.append(id_to_token[next_id])

    return join_tokens(prompt_tokens + new_tokens)


def find_source_jokes(prompt, texts, ids):
    """Find every joke whose first three tokens match the prompt, so we can show the original.

    Returns a list of (source_id, joke_text). More than one match means the prompt is
    ambiguous, so the caller should record which source it actually compared against.
    """
    prompt_tokens = tokenize(prompt)
    return [(source_id, text) for source_id, text in zip(ids, texts)
            if tokenize(text)[:3] == prompt_tokens]
