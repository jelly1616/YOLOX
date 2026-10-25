"""Create a REVIEWABLE chronological 4000/1000 split draft.
WARNING: assumes alphabetically sorted file names are chronological;
if false, do NOT use output. Train/val from same long video are not
independent, especially around the split boundary. Prefer separate sessions.
"""
import argparse
import csv
from pathlib import Path
from final_config import load_final_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group-size', type=int, default=100)
    p.add_argument('--config', default='config_final.yaml')
    args = p.parse_args()
    if args.group_size <= 0:
        raise ValueError('group-size must be positive')
    cfg = load_final_config(args.config)
    data = cfg['data']
    files = sorted(x for x in Path(data['raw_images']).iterdir() if x.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
    ntrain, nval = data['train_count'], data['val_count']
    if len(files) != ntrain+nval:
        raise ValueError(f'Expected {ntrain+nval} images, found {len(files)}')
    path = Path(data['manifest'])
    if path.exists():
        raise FileExistsError(f'Will not overwrite an existing split manifest: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['file_name', 'group', 'split'])
        for i, file in enumerate(files):
            # 4000 and 1000 divisible by 100: no group crosses boundary by default.
            group = ('train' if i < ntrain else 'val') + f'_block_{i // args.group_size:04}'
            writer.writerow([file.name, group, 'train' if i < ntrain else 'val'])
    print('Wrote:', path)
    print('IMPORTANT: inspect chronological order and group identifiers before training.')
    print('If files originate from one continuous recording, adjacent frames near boundary can cause leakage.')
    print('A genuinely independent validation recording/session is strongly preferred.')


if __name__ == '__main__':
    main()
