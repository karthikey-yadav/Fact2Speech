"""Overwrite train.txt/val.txt from splits.json (per-speaker, no overlap).
python English\make_splits.py English\preprocessed_data\Svarah English\raw_data\splits.json
"""
import json, sys
from pathlib import Path

d = Path(sys.argv[1])
sp = json.load(open(sys.argv[2]))
train, val = set(sp['train']), set(sp['val'])
lines = []
for n in ('train.txt', 'val.txt'):
    lines += (d / n).read_text(encoding='utf-8').splitlines()
tr = [l for l in lines if l.split('|')[0] in train]
va = [l for l in lines if l.split('|')[0] in val]
assert not ({l.split('|')[0] for l in tr} & {l.split('|')[0] for l in va})
(d / 'train.txt').write_text('\n'.join(tr) + '\n', encoding='utf-8')
(d / 'val.txt').write_text('\n'.join(va) + '\n', encoding='utf-8')
print('train', len(tr), 'val', len(va), 'dropped (no alignment)', len(train) + len(val) - len(tr) - len(va))
