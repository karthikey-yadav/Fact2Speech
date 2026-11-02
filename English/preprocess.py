"""Svarah -> FastSpeech2 raw_data (multi-speaker, proxy speaker IDs).
python English\preprocess.py --src C:\dev\data\svarah\data --out C:\dev\project\English\raw_data
"""
import argparse, io, json, random, re, sys
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, librosa
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from text import _clean_text  # repo text frontend

SR = 22050
META = ['duration', 'text', 'gender', 'age-group', 'primary_language',
        'native_place_state', 'native_place_district',
        'highest_qualification', 'job_category', 'occupation_domain']
SPK_COLS = META[2:]
OK_TEXT = re.compile(r"^[A-Za-z0-9 .,?!'\-]+$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--target_hours', type=float, default=6.0)
    ap.add_argument('--val_frac', type=float, default=1 / 6)
    ap.add_argument('--min_dur', type=float, default=1.0)
    ap.add_argument('--max_dur', type=float, default=12.0)
    ap.add_argument('--seed', type=int, default=1234)
    a = ap.parse_args()

    shards = sorted(Path(a.src).glob('*.parquet'))
    parts = []
    for si, f in enumerate(shards):
        m = pd.read_parquet(f, columns=META)
        m['shard'] = si
        m['row'] = np.arange(len(m))
        parts.append(m)
    df = pd.concat(parts, ignore_index=True)
    stats = {'raw_utts': len(df), 'raw_hours': round(df.duration.sum() / 3600, 2)}

    df['text'] = df['text'].astype(str).str.strip()
    df = df[(df.duration >= a.min_dur) & (df.duration <= a.max_dur)]
    df = df[df.text.map(lambda t: bool(OK_TEXT.match(t)))].copy()
    df['clean'] = df.text.map(lambda t: _clean_text(t, ['english_cleaners']).strip())
    df = df[~df.clean.str.contains(r'\d')]
    df = df[df.clean.str.split().str.len() >= 2]
    cps = df.clean.str.len() / df.duration
    df = df[(cps >= 4) & (cps <= 25)].reset_index(drop=True)
    stats['pool_utts'] = len(df)
    stats['pool_hours'] = round(df.duration.sum() / 3600, 2)

    df['speaker_key'] = df[SPK_COLS].astype(str).agg('|'.join, axis=1)
    rng = random.Random(a.seed)
    frac = min(1.0, a.target_hours / (df.duration.sum() / 3600))
    assign = {}
    for key, g in df.groupby('speaker_key'):
        idx = list(g.index)
        rng.shuffle(idx)
        budget, acc, sel = frac * g.duration.sum(), 0.0, []
        for i in idx:
            if acc >= budget:
                break
            sel.append(i)
            acc += df.at[i, 'duration']
        if len(sel) < 4:
            continue
        vbudget, vacc, nval = a.val_frac * acc, 0.0, 0
        for i in reversed(sel):
            if vacc >= vbudget or nval >= len(sel) - 2:
                break
            assign[i] = 'val'
            vacc += df.at[i, 'duration']
            nval += 1
        for i in sel:
            assign.setdefault(i, 'train')

    sel = df.loc[list(assign)].copy()
    sel['split'] = pd.Series(assign)
    keys = sorted(sel.speaker_key.unique())
    spk_id = {k: 'spk%03d' % n for n, k in enumerate(keys)}
    sel['spk'] = sel.speaker_key.map(spk_id)
    sel = sel.sort_values(['spk', 'shard', 'row'])
    sel['utt'] = sel.spk + '_' + sel.groupby('spk').cumcount().map('{:04d}'.format)

    out = Path(a.out) / 'Svarah'
    done = []
    for si, f in enumerate(shards):
        need = sel[sel.shard == si]
        if need.empty:
            continue
        col = pd.read_parquet(f, columns=['audio_filepath'])['audio_filepath']
        for r in tqdm(need.itertuples(), total=len(need), desc=f.name):
            y, sr = sf.read(io.BytesIO(col.iloc[r.row]['bytes']), dtype='float32')
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            peak = float(np.abs(y).max())
            if peak < 1e-4:
                continue
            y = y / peak * 0.95
            d = out / r.spk
            d.mkdir(parents=True, exist_ok=True)
            sf.write(str(d / (r.utt + '.wav')), y, SR, subtype='PCM_16')
            (d / (r.utt + '.lab')).write_text(r.clean, encoding='utf-8')
            done.append((r.utt, r.spk, r.split, len(y) / SR))

    res = pd.DataFrame(done, columns=['utt', 'spk', 'split', 'dur'])
    base = Path(a.out)
    json.dump({s: res[res.split == s].utt.tolist() for s in ('train', 'val')},
              open(base / 'splits.json', 'w'), indent=1)
    sm = pd.DataFrame({'spk': list(spk_id.values()), 'key': list(spk_id.keys())})
    sm.to_csv(base / 'speaker_map.csv', index=False)
    for s in ('train', 'val'):
        x = res[res.split == s]
        stats[s] = {'utts': len(x), 'hours': round(x.dur.sum() / 3600, 2),
                    'speakers': int(x.spk.nunique())}
    stats['val_speakers_not_in_train'] = int(len(set(res[res.split == 'val'].spk) - set(res[res.split == 'train'].spk)))
    stats['filters'] = {'dur': [a.min_dur, a.max_dur], 'chars_per_sec': [4, 25], 'sr': SR,
                        'seed': a.seed, 'speaker_id': 'proxy: metadata tuple'}
    json.dump(stats, open(Path(__file__).parent / 'data_stats.json', 'w'), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
