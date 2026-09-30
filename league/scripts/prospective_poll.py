"""Prospective collection of official NBA injury reports, recording when each file first becomes retrievable.

The benchmark in this repository is built from reports dated by the time printed on them. That tells you when a
report says it was issued, not when it could first be read. This collector records the second quantity for the
2026-27 season: it probes the league's published file names every few minutes and writes down, for each file, the
moment it first answered, the server's own clock, and the validators it returned.

  python league/scripts/prospective_poll.py                       # one polling run (schedule it every 5 minutes)
  python league/scripts/prospective_poll.py --test-date 2026-03-15  # probe one past day, to test a deployment
  P2_PROSPECTIVE=<dir> overrides the output directory (default data/prospective_2627)

Each run probes the candidate file names for the quarter-hour slots in the last LOOKBACK hours, in both of the
naming formats the league has used; downloads the ones it has not seen; re-checks recently seen files for a
re-upload (a changed ETag or length); and appends one line to poll_log.jsonl. Nothing is parsed here.

Politeness: one request at a time, a pause between requests, a per-run request cap, and an identifying
User-Agent. The output directory holds the PDFs, seen.csv (one row per file version) and poll_log.jsonl.
"""
import csv, hashlib, json, os, sys, time, urllib.request, urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('P2_PROSPECTIVE', ROOT / 'data/prospective_2627'))
BASE = 'https://ak-static.cms.nba.com/referee/injury/'
UA = 'Mozilla/5.0 (research: NBA injury-report publication times; contact via repository)'
ET = ZoneInfo('America/New_York')
LOOKBACK_H, RECHECK_H, PAUSE, MAX_REQ = 6, 2, 0.3, 120
SEEN_FIELDS = ['slot_et', 'file', 'version', 'first_seen_utc', 'http_date', 'last_modified', 'etag', 'content_length',
               'bytes', 'sha256', 'stored_as']


def names_for(slot):
    """Candidate file names for a quarter-hour slot (current format first, legacy hour-only format on the hour)."""
    h12 = slot.hour % 12 or 12; ap = 'AM' if slot.hour < 12 else 'PM'; d = slot.strftime('%Y-%m-%d')
    out = [f'Injury-Report_{d}_{h12:02d}_{slot.minute:02d}{ap}.pdf']
    if slot.minute == 0: out.append(f'Injury-Report_{d}_{h12:02d}{ap}.pdf')
    return out


def request(url, method):
    req = urllib.request.Request(url, method=method, headers={'User-Agent': UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), (r.read() if method == 'GET' else b'')
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), b''


def load_seen():
    p = OUT / 'seen.csv'
    return list(csv.DictReader(open(p, newline=''))) if p.exists() else []


def append_seen(rec):
    p = OUT / 'seen.csv'; new = not p.exists()
    with open(p, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SEEN_FIELDS)
        if new: w.writeheader()
        w.writerow(rec)


def store(name, body, version):
    d = OUT / 'pdfs'; d.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(body).hexdigest()
    fn = name if version == 1 else name.replace('.pdf', f'.v{version}_{sha[:10]}.pdf')
    (d / fn).write_bytes(body)
    return sha, fn


def run(slots, recheck=True):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc)
    seen = load_seen()
    latest = {}
    for r in seen: latest[r['file']] = r  # last version per file
    n_req = n_new = n_changed = 0; errors = []; clock_skew = None
    for slot in slots:
        for name in names_for(slot):
            if name in latest or n_req >= MAX_REQ: continue
            n_req += 1
            try:
                st, hd, _ = request(BASE + name, 'HEAD')
                if st == 200:
                    st, hd, body = request(BASE + name, 'GET'); n_req += 1
                    now = datetime.now(timezone.utc)
                    if st == 200 and body:
                        sha, fn = store(name, body, 1)
                        rec = dict(slot_et=slot.isoformat(), file=name, version=1, first_seen_utc=now.isoformat(timespec='seconds'),
                                   http_date=hd.get('Date', ''), last_modified=hd.get('Last-Modified', ''), etag=hd.get('ETag', ''),
                                   content_length=hd.get('Content-Length', ''), bytes=len(body), sha256=sha, stored_as=fn)
                        append_seen(rec); latest[name] = rec; n_new += 1
                if hd.get('Date') and clock_skew is None:
                    srv = datetime.strptime(hd['Date'], '%a, %d %b %Y %H:%M:%S GMT').replace(tzinfo=timezone.utc)
                    clock_skew = round((datetime.now(timezone.utc) - srv).total_seconds(), 1)
            except Exception as e:
                errors.append(f'{name}:{type(e).__name__}')
            time.sleep(PAUSE)
    if recheck:  # recently seen files: detect re-uploads (ETag or size change)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=RECHECK_H)
        for name, r in list(latest.items()):
            if datetime.fromisoformat(r['first_seen_utc']) < cutoff or n_req >= MAX_REQ: continue
            n_req += 1
            try:
                st, hd, _ = request(BASE + name, 'HEAD')
                if st == 200 and (hd.get('ETag', '') != r['etag'] or hd.get('Content-Length', '') != r['content_length']):
                    st, hd, body = request(BASE + name, 'GET'); n_req += 1
                    v = int(r['version']) + 1
                    sha, fn = store(name, body, v)
                    rec = dict(slot_et=r['slot_et'], file=name, version=v, first_seen_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                               http_date=hd.get('Date', ''), last_modified=hd.get('Last-Modified', ''), etag=hd.get('ETag', ''),
                               content_length=hd.get('Content-Length', ''), bytes=len(body), sha256=sha, stored_as=fn)
                    append_seen(rec); latest[name] = rec; n_changed += 1
            except Exception as e:
                errors.append(f'recheck {name}:{type(e).__name__}')
            time.sleep(PAUSE)
    log = dict(run_start_utc=t0.isoformat(timespec='seconds'), run_end_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
               slots=len(slots), first_slot_et=slots[0].isoformat() if slots else '', last_slot_et=slots[-1].isoformat() if slots else '',
               requests=n_req, new_files=n_new, changed_files=n_changed, errors=errors[:20], n_errors=len(errors),
               local_minus_server_clock_s=clock_skew, hit_request_cap=n_req >= MAX_REQ)
    with open(OUT / 'poll_log.jsonl', 'a') as f: f.write(json.dumps(log) + '\n')
    return log


def slots_between(start, end):
    s = start.replace(minute=start.minute - start.minute % 15, second=0, microsecond=0)
    out = []
    while s <= end:
        out.append(s); s += timedelta(minutes=15)
    return out


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--test-date':
        MAX_REQ = 400
        d = datetime.strptime(sys.argv[2], '%Y-%m-%d').replace(tzinfo=ET)
        print(json.dumps(run(slots_between(d, d + timedelta(hours=23, minutes=45)), recheck=False), indent=1))
    else:
        now = datetime.now(ET)
        print(json.dumps(run(slots_between(now - timedelta(hours=LOOKBACK_H), now))))
