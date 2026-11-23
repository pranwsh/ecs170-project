"""Recompute one-epoch validation error counts; run from the template root."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json, pathlib, collections
from local_code.stage_4_code.Text_Preprocessing import tokenize
root=pathlib.Path('result/stage_4_result')
for c in ('rnn','lstm','gru'):
 p=json.loads((root/f'validation_classification_{c}/predictions.json').read_text())
 bins=collections.defaultdict(lambda:[0,0]); errors=collections.Counter()
 for id,y,yp in zip(p['source_ids'],p['labels'],p['predictions']):
  n=len(tokenize((pathlib.Path('data/stage_4_data/text_classification')/id).read_text()))
  label='<=128' if n<=128 else '129-256' if n<=256 else '>256'
  bins[label][0]+=1; bins[label][1]+=y!=yp
  if y!=yp: errors['false negative' if y else 'false positive']+=1
 print(c,dict(bins),dict(errors))
