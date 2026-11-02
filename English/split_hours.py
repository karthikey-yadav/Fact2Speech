import numpy as np, sys
from pathlib import Path
d = Path(sys.argv[1])
for s in ('train', 'val'):
    t = 0.0
    n = 0
    for l in (d / (s + '.txt')).read_text(encoding='utf-8').splitlines():
        b, spk = l.split('|')[:2]
        m = np.load(d / 'mel' / (spk + '-mel-' + b + '.npy'), mmap_mode='r')
        fr = m.shape[0] if m.shape[1] == 80 else m.shape[1]
        t += fr * 256 / 22050
        n += 1
    print(s, n, 'utts', round(t / 3600, 2), 'h')
