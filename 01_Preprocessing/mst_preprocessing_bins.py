"""
mst_preprocessing_bins.py
--------------------------
Builds stimulus_bin_mapping.csv — maps every Lure image_path to its
lure-similarity bin (1 = very similar to target, 5 = least similar).

Bin file logic
--------------
Set6 bins.txt   : Objects stimuli, indices 1–192
SetScC bins.txt : Scenes  stimuli, indices 1–192

Objects folder (indices 1–384 in the data):
  1–192   → Set6   bins  (Set6 objects)
  193–384 → SetScC bins  (SetScC objects, offset −192)

Scenes folder (indices 1–192 in the data):
  1–192   → SetScC bins  (SetScC scenes)

task_only exception:
  Uses Set6 bins_ob.txt (indices 1–384, both sets combined in one file).
  Objects and Scenes both look up directly by image index — no offset.

item_only and Both_item_task share identical bin files (verified with diff).
"""

from pathlib import Path
import re
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_CSV    = ROOT / '01_Preprocessing' / 'compiled_preprocessed.csv'
OUT_CSV     = ROOT / '01_Preprocessing' / 'stimulus_bin_mapping.csv'

SHARED_BASE = ROOT / 'MAIN_DATA' / 'MST_Data' / 'item_only'
TASK_BASE   = ROOT / 'MAIN_DATA' / 'MST_Data' / 'task_only'


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_bin_file(path: Path) -> dict:
    """Return {index (int): bin (int)} from a tab-separated bin file."""
    mapping = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        mapping[int(parts[0])] = int(parts[1])
    return mapping


def get_folder(image_path: str) -> str:
    return re.split(r'[/\\]', image_path)[0]


def extract_index(image_path: str) -> int | None:
    m = re.search(r'(\d+)', Path(image_path).name)
    return int(m.group(1)) if m else None


# ── Load bin maps ─────────────────────────────────────────────────────────────
set6   = load_bin_file(SHARED_BASE / 'Set6 bins.txt')
setsc  = load_bin_file(SHARED_BASE / 'SetScC bins.txt')
task_ob = load_bin_file(TASK_BASE  / 'Set6 bins_ob.txt')


# ── Bin lookup ────────────────────────────────────────────────────────────────
def lookup_bin(image_path: str, condition: str) -> int | None:
    folder = get_folder(image_path)
    idx    = extract_index(image_path)
    if idx is None:
        return None

    if 'task_only' in str(condition).lower():
        return task_ob.get(idx)

    if folder == 'Objects':
        if 1 <= idx <= 192:
            return set6.get(idx)
        if 193 <= idx <= 384:
            return setsc.get(idx - 192)

    if folder == 'Scenes':
        return setsc.get(idx)

    return None


# ── Build mapping ─────────────────────────────────────────────────────────────
def build_mapping(df: pd.DataFrame) -> pd.DataFrame:
    lures = df[df['trial_type'] == 'Lure'][['image_path', 'condition']].drop_duplicates()

    rows = []
    for _, row in lures.iterrows():
        path      = row['image_path']
        condition = row['condition']
        rows.append({
            'image_path':    path,
            'condition':     condition,
            'stimulus_type': get_folder(path),
            'stimulus_index': extract_index(path),
            'bin':           lookup_bin(path, condition),
        })

    return (pd.DataFrame(rows)
            .sort_values(['condition', 'stimulus_type', 'stimulus_index'])
            .reset_index(drop=True))


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df  = pd.read_csv(DATA_CSV)
    out = build_mapping(df)

    missing = out['bin'].isna().sum()
    print(f'Total Lure stimulus-condition pairs : {len(out)}')
    print(f'Missing bin assignments             : {missing}')
    print()
    print('Bin distribution per condition:')
    print(out.groupby('condition')['bin'].value_counts().unstack().fillna(0).astype(int))
    print()
    print('Bin distribution per stimulus_type:')
    print(out.groupby('stimulus_type')['bin'].value_counts().unstack().fillna(0).astype(int))

    out.to_csv(OUT_CSV, index=False)
    print(f'\nSaved → {OUT_CSV}')
    print()
    print(out.head(10).to_string(index=False))
