"""Fetch the public sources listed in league/public/sources.csv and verify every SHA-256.

  python league/scripts/fetch_sources.py            # download into data/public/, verify, write fetch_log.csv
  python league/scripts/fetch_sources.py make-list  # (maintainers only) write league/public/sources.csv

Sources: official NBA injury-report PDFs (ak-static.cms.nba.com) and a pinned CC0 Kaggle box-score dataset.
Nothing is fetched from ESPN or stats.nba.com. Files that already exist and verify are skipped, so the script can be
re-run after an interruption.
"""
import sys, time, hashlib, csv, os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
LIST = ROOT / 'league/public/sources.csv'
DEST = Path(os.environ.get('P2_PUBLIC', ROOT / 'data/public'))
PDF_URL = 'https://ak-static.cms.nba.com/referee/injury/'
KAGGLE = 'https://www.kaggle.com/api/v1/datasets/download/eoinamoore/historical-nba-data-and-player-box-scores/'
KAGGLE_VERSION = 515
PAUSE = 0.5


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def make_list():
    """(maintainers) Complete report series: every file found on the league server for 2021-22...2025-26 game days,
    from 00:00 ET to 90 minutes before the day's last tip-off."""
    import pandas as pd
    log = pd.read_csv(ROOT / 'data/league/paper2_ext/full_series/fetch_log.csv', dtype=str)
    log = log[log.status == '200'].drop_duplicates('file', keep='last')
    rows = [dict(kind='injury_pdf', file='injury_reports/' + f, url=PDF_URL + f, bytes=int(b), sha256=s)
            for f, b, s in sorted(zip(log.file, log.bytes, log.sha256))]
    kdir = ROOT / f'data/external/kaggle/eoinamoore_v{KAGGLE_VERSION}'
    for f in ('Games.csv', 'PlayerStatistics.csv'):
        rows.append(dict(kind='kaggle_cc0', file=f'kaggle_v{KAGGLE_VERSION}/{f}', url=f'{KAGGLE}{f}?datasetVersionNumber={KAGGLE_VERSION}',
                         bytes=(kdir / f).stat().st_size, sha256=sha256(kdir / f)))
    LIST.parent.mkdir(parents=True, exist_ok=True)
    with open(LIST, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['kind', 'file', 'url', 'bytes', 'sha256']); w.writeheader(); w.writerows(rows)
    print(len(rows), 'sources ->', LIST)


def fetch():
    rows = list(csv.DictReader(open(LIST)))
    log = []
    for i, r in enumerate(rows):
        out = DEST / r['file']; out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and sha256(out) == r['sha256']:
            log.append(dict(file=r['file'], result='ok_cached')); continue
        res = 'failed'
        for attempt in range(3):
            try:
                req = urllib.request.Request(r['url'], headers={'User-Agent': 'Mozilla/5.0 (research rebuild; paper2)'})
                with urllib.request.urlopen(req, timeout=300) as resp, open(str(out) + '.part', 'wb') as f:
                    while True:
                        b = resp.read(1 << 20)
                        if not b: break
                        f.write(b)
                os.replace(str(out) + '.part', out)
                res = 'ok' if sha256(out) == r['sha256'] else 'sha_mismatch'
                break
            except Exception as e:
                res = f'failed:{type(e).__name__}'
                time.sleep(2 * (attempt + 1))
        log.append(dict(file=r['file'], result=res))
        if (i + 1) % 200 == 0: print(i + 1, 'of', len(rows), flush=True)
        time.sleep(PAUSE)
    with open(DEST / 'fetch_log.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'result']); w.writeheader(); w.writerows(log)
    from collections import Counter
    c = Counter(x['result'] for x in log); print(dict(c))
    return c


if __name__ == '__main__':
    if sys.argv[1:] == ['make-list']:
        make_list()
    else:
        c = fetch()
        sys.exit(0 if set(c) <= {'ok', 'ok_cached'} else 1)
