"""Rebuild the point-in-time availability features from the fetched sources, and rerun the
headline leakage experiment from public sources only (official injury-report PDFs + pinned CC0 Kaggle box scores),
as fetched by fetch_sources.py. No ESPN, stats.nba.com, private cache or betting data is read.

  python league/scripts/rebuild.py parse [budget_seconds]   # resumable; re-run until it prints 'done'
  python league/scripts/rebuild.py build                    # -> data/public/out/*.csv
  python league/scripts/rebuild.py model                    # -> results/paper2_public_path.json

Rules follow the project's main build path, with two
differences: tip-off is the scheduled time in the Kaggle game table, and the roster at the cutoff uses box appearances
only (no transaction log).
"""
import sys, os, re, json, time, bisect
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'league/public'))
from pit_rules import name_key, name_keys, reason_class, reference_minutes

PUB = Path(os.environ.get('P2_PUBLIC', ROOT / 'data/public'))
PDF, PARTS, OUTD = PUB / 'injury_reports', PUB / 'parse', PUB / 'out'
KAG = PUB / 'kaggle_v515'
SEASONS = (2022, 2023, 2024, 2025, 2026)
GAME_TYPES = {'Regular Season', 'NBA Emirates Cup', 'Emirates NBA Cup', 'NBA Cup'}
CUTS = {'6h': pd.Timedelta(minutes=360), '90m': pd.Timedelta(minutes=90)}
CFG = dict(max_games=20, min_games=5, min_minutes=100, max_opportunities=20, q=0.9, cap=40)
HIST_DAYS = pd.Timedelta(days=365)
TEAM_BY_NAME = {'Hawks': 'ATL', 'Celtics': 'BOS', 'Nets': 'BKN', 'Hornets': 'CHA', 'Bulls': 'CHI', 'Cavaliers': 'CLE',
                'Mavericks': 'DAL', 'Nuggets': 'DEN', 'Pistons': 'DET', 'Warriors': 'GSW', 'Rockets': 'HOU', 'Pacers': 'IND',
                'Clippers': 'LAC', 'Lakers': 'LAL', 'Grizzlies': 'MEM', 'Heat': 'MIA', 'Bucks': 'MIL', 'Timberwolves': 'MIN',
                'Pelicans': 'NOP', 'Knicks': 'NYK', 'Thunder': 'OKC', 'Magic': 'ORL', '76ers': 'PHI', 'Suns': 'PHX',
                'Trail Blazers': 'POR', 'Kings': 'SAC', 'Spurs': 'SAS', 'Raptors': 'TOR', 'Jazz': 'UTA', 'Wizards': 'WAS'}
HDR = re.compile(r'Injury Report:\s*(\d\d/\d\d/\d\d)\s+(\d\d:\d\d)\s*([AP]M)')
NAME = re.compile(r'Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})(?:_(\d{2}))?(AM|PM)\.pdf$')


# ---------------------------------------------------------------- parse
def parse_one(f):
    """Parse one report: rows, team blocks and the time printed in the header."""
    import pdfplumber
    from league.scripts.parse_injury_pdf import parse
    try:
        r, b = parse(str(PDF / f)); err = '' if (r or b) else 'empty'
    except Exception as e:
        r, b, err = [], [], repr(e)[:200]
    hdr = ''
    try:
        with pdfplumber.open(PDF / f) as pdf:
            pg = pdf.pages[0]; t = (pg.crop((0, 0, pg.width, 80)).extract_text() or '').replace('\n', ' ')
        m = HDR.search(t)
        if m: hdr = pd.Timestamp(f'{m.group(1)} {m.group(2)} {m.group(3)}').strftime('%Y-%m-%dT%H:%M:00')
    except Exception as e:
        err = err or repr(e)[:200]
    n = NAME.match(f); h = int(n.group(2)) % 12 + (12 if n.group(4) == 'PM' else 0)
    return dict(file=f, name_ts_et=f'{n.group(1)}T{h:02d}:{n.group(3) or "00"}:00', header_ts_et=hdr,
                n_rows=len(r), n_blocks=len(b), error=err), r, b


def parse_step(budget, workers=2):  # 2 processes: this machine runs the model reruns at the same time
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time(); PARTS.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in os.listdir(PDF) if NAME.match(f))
    done = set()
    for p in PARTS.glob('files_*.csv'): done |= set(pd.read_csv(p).file)
    todo = [f for f in files if f not in done]
    k = len(list(PARTS.glob('files_*.csv'))); parsed = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for c in range(0, len(todo), 400):
            if time.time() - t0 > budget: break
            res = list(ex.map(parse_one, todo[c:c + 400], chunksize=20))
            meta = [x[0] for x in res]; rows = [r for x in res for r in x[1]]; blocks = [b for x in res for b in x[2]]
            for name, d in (('files', meta), ('rows', rows), ('blocks', blocks)):
                pd.DataFrame(d).to_csv(PARTS / f'{name}_{k:04d}.csv', index=False)
            k += 1; parsed += len(res)
            print(json.dumps(dict(parsed=parsed, of=len(todo), seconds=round(time.time() - t0))), flush=True)
    left = len(todo) - parsed
    print(json.dumps(dict(files=len(files), parsed_now=parsed, remaining=left, seconds=round(time.time() - t0))))
    if left == 0: print('done')


# ---------------------------------------------------------------- inputs
def season_of(ts_et):
    return np.where(ts_et.dt.month >= 9, ts_et.dt.year + 1, ts_et.dt.year)


def load_games_box():
    kg = pd.read_csv(KAG / 'Games.csv', low_memory=False)
    kg = kg[kg.gameType.isin(GAME_TYPES) & (kg.gameDateTimeEst.astype(str) >= '2019-10-01')].copy()
    kg['home'] = kg.hometeamName.map(TEAM_BY_NAME); kg['away'] = kg.awayteamName.map(TEAM_BY_NAME)
    kg = kg.dropna(subset=['home', 'away'])
    et = pd.to_datetime(kg.gameDateTimeEst).dt.tz_localize('US/Eastern', ambiguous=True, nonexistent='shift_forward')
    g = pd.DataFrame(dict(game_id=kg.gameId.astype(str), season_year=season_of(et), start=et.dt.tz_convert('UTC'),
                          et_date=et.dt.strftime('%m/%d/%Y'), home=kg.home, away=kg.away,
                          home_score=kg.homeScore.astype(float), away_score=kg.awayScore.astype(float)))
    cols = ['personId', 'gameId', 'firstName', 'lastName', 'playerteamName', 'numMinutes', 'points', 'assists', 'blocks', 'steals',
            'fieldGoalsAttempted', 'fieldGoalsMade', 'freeThrowsAttempted', 'freeThrowsMade', 'reboundsTotal', 'turnovers']
    kp = pd.read_csv(KAG / 'PlayerStatistics.csv', low_memory=False, usecols=cols)
    kp = kp[kp.gameId.astype(str).isin(set(g.game_id))].copy()
    box = pd.DataFrame(dict(game_id=kp.gameId.astype(str), pid=kp.personId.astype(str), team=kp.playerteamName.map(TEAM_BY_NAME),
                            player_name=(kp.firstName.fillna('') + ' ' + kp.lastName.fillna('')).str.strip()))
    num = lambda c: pd.to_numeric(kp[c], errors='coerce').fillna(0.0).values
    box['min'] = num('numMinutes')
    box['eff'] = (num('points') + num('reboundsTotal') + num('assists') + num('steals') + num('blocks') - num('turnovers')
                  - (num('fieldGoalsAttempted') - num('fieldGoalsMade')) - (num('freeThrowsAttempted') - num('freeThrowsMade')))
    box['pts'] = num('points')
    assert box.team.notna().all()
    # Games.csv shows 0-0 for a few completed games: take team points from the box score instead
    pts = box.groupby(['game_id', 'team']).pts.sum()
    bad = (g.home_score.fillna(0) == 0) & (g.away_score.fillna(0) == 0)
    g.loc[bad, 'home_score'] = [pts.get((a, b), np.nan) for a, b in zip(g.game_id[bad], g.home[bad])]
    g.loc[bad, 'away_score'] = [pts.get((a, b), np.nan) for a, b in zip(g.game_id[bad], g.away[bad])]
    g = g.dropna(subset=['home_score', 'away_score'])
    g = g[g.game_id.isin(set(box.game_id))]
    box = box.merge(g[['game_id', 'start', 'season_year']], on='game_id')
    return g.sort_values('start').reset_index(drop=True), box, int(bad.sum())


def load_reports(g):
    # a file parsed twice (overlapping runs) keeps the output of its first part only
    rd = lambda kind: pd.concat([pd.read_csv(p, dtype=str).assign(part=p.stem[-4:]) for p in sorted(PARTS.glob(f'{kind}_*.csv'))
                                 if p.stat().st_size > 5])
    files = rd('files').drop_duplicates('file').reset_index(drop=True)
    first = files.set_index('file').part
    rows, blocks = [d[d.part == d.file.map(first)] for d in (rd('rows'), rd('blocks'))]
    ts = files.header_ts_et.fillna('')
    no_hdr = ts == ''
    ts[no_hdr] = (pd.to_datetime(files.name_ts_et[no_hdr]) + pd.Timedelta(minutes=30)).dt.strftime('%Y-%m-%dT%H:%M:00')
    # same rule as the main path: on a fall-back date the ambiguous hour is read as its first (DST) occurrence
    files['ts'] = pd.to_datetime(ts).dt.tz_localize('US/Eastern', ambiguous=True, nonexistent='shift_forward').dt.tz_convert('UTC')
    fts = files.set_index('file').ts
    out = {}
    for name, d in (('rows', rows), ('blocks', blocks)):
        d = d[d.matchup.fillna('').str.contains('@')].copy()
        d['away'] = d.matchup.str.split('@').str[0]; d['home'] = d.matchup.str.split('@').str[1]
        d = d.merge(g[['game_id', 'et_date', 'home', 'away']], left_on=['date', 'home', 'away'], right_on=['et_date', 'home', 'away'])
        d['ts'] = d.file.map(fts)
        out[name] = d
    b = out['blocks'].groupby(['ts', 'game_id', 'team']).state.agg(lambda s: 'listed' if 'listed' in set(s) else s.iloc[0]).reset_index()
    r = out['rows'].drop_duplicates(['ts', 'game_id', 'team', 'player'])
    info = dict(pdf_files=len(files), files_no_header=int(no_hdr.sum()), files_error=int((files.error.fillna('') != '').sum()))
    return r, b, info


# ---------------------------------------------------------------- point-in-time rules
def status_at(g, rows, blocks, box, cut):
    """Latest report at or before the cutoff in which the team's block is listed; names matched to NBA person ids."""
    gg = g[['game_id', 'start', 'season_year', 'home', 'away']].copy(); gg['cutoff'] = gg.start - cut
    b = blocks.merge(gg, on='game_id'); b = b[b.ts <= b.cutoff]
    best = b[b.state == 'listed'].sort_values('ts').groupby(['game_id', 'team']).tail(1)[['game_id', 'team', 'ts']]
    r = rows.merge(best, on=['game_id', 'team', 'ts']).merge(gg[['game_id', 'season_year']], on='game_id')
    r = r[(r.team == r.home) | (r.team == r.away)].copy()
    rn = box[['pid', 'player_name', 'team', 'season_year']].drop_duplicates()
    luts, globs = [], []
    for level in range(3):
        rn['k'] = rn.player_name.map(lambda n: name_keys(n)[level])
        luts.append(rn.groupby(['team', 'season_year', 'k']).pid.agg(lambda s: sorted(set(s))).to_dict())
        globs.append(rn.groupby('k').pid.agg(lambda s: sorted(set(s))).to_dict())

    def match(name, tm, sy):
        keys = name_keys(name)
        for level in range(3):
            for c in (luts[level].get((tm, sy, keys[level])), globs[level].get(keys[level]) if level < 2 else None):
                if c and len(c) == 1: return c[0]
        return ''
    r['pid'] = [match(n, t, s) for n, t, s in zip(r.player, r.team, r.season_year)]
    r['reason_class'] = r.reason.map(reason_class)
    r['k'] = r.player.map(name_key)
    r = r.drop_duplicates(['game_id', 'team', 'k'])
    return best, r[['game_id', 'team', 'player', 'pid', 'status', 'reason', 'reason_class', 'ts']]


class League:
    """Last-known-team rule from box appearances (no transaction log) and the main path's player profile."""

    def __init__(self, g, box):
        self.hist = {}
        pos = box[box['min'] > 0].sort_values('start')
        for pid, d in pos.groupby('pid'):
            self.hist[pid] = dict(t=d.start.values.astype('datetime64[ns]'), m=d['min'].values, e=d.eff.values)
        ev = box[['pid', 'start', 'team']].sort_values('start')
        self.ev = {pid: (d.start.tolist(), d.team.tolist()) for pid, d in ev.groupby('pid')}
        self.sched = {tm: sorted(pd.concat([g.start[g.home == tm], g.start[g.away == tm]]).tolist()) for tm in set(g.home) | set(g.away)}
        self.team_players = box.groupby('team').pid.agg(set).to_dict()

    def team_at(self, pid, cutoff):
        ts, tms = self.ev.get(pid, ([], []))
        i = bisect.bisect_left(ts, cutoff) - 1
        if i < 0 or cutoff - ts[i] > HIST_DAYS: return None
        return tms[i]

    def roster(self, tm, cutoff, listed):
        return sorted(p for p in self.team_players[tm] | set(listed)
                      if (a := self.team_at(p, cutoff)) == tm or (p in listed and a is None))

    def misattributed(self, tm, cutoff, listed):
        return sorted(p for p in listed if self.team_at(p, cutoff) not in (tm, None))

    def profile(self, pid, tm, cutoff):
        h = self.hist.get(pid)
        base = dict(canonical_player_id=pid, history_n=0, supported=False)
        if h is None: return base
        c = np.datetime64(cutoff.tz_convert(None))
        hi = np.searchsorted(h['t'] + np.timedelta64(3, 'h'), c, side='left')
        lo = np.searchsorted(h['t'], np.datetime64((cutoff - HIST_DAYS).tz_convert(None)), side='left')
        idx = np.arange(lo, hi)[-CFG['max_games']:]
        n = len(idx)
        if n <= 0: return base
        m = h['m'][idx]; mm = float(m.sum())
        last = pd.Timestamp(h['t'][idx[-1]], tz='UTC'); s = self.sched[tm]
        opp = bisect.bisect_left(s, cutoff) - bisect.bisect_right(s, last)
        ok = n >= CFG['min_games'] and mm >= CFG['min_minutes'] and opp <= CFG['max_opportunities']
        return dict(canonical_player_id=pid, history_n=n, mean_minutes=mm / n, supported=ok,
                    capacity_minutes=min(CFG['cap'], float(np.quantile(m, CFG['q'], method='linear'))))

    def eff_prior(self, pid, cutoff):
        h = self.hist.get(pid)
        if h is None: return np.nan
        c = np.datetime64(cutoff.tz_convert(None))
        hi = np.searchsorted(h['t'] + np.timedelta64(3, 'h'), c)
        lo = max(np.searchsorted(h['t'], np.datetime64((cutoff - HIST_DAYS).tz_convert(None))), hi - 20)
        mm = h['m'][lo:hi].sum()
        return h['e'][lo:hi].sum() / mm if mm > 0 else np.nan


def team_games(g, lg, best, st, cut, variant):
    rep = set(zip(best.game_id, best.team))
    stg = {k: d for k, d in st.groupby(['game_id', 'team'])}
    tg, ros = [], []
    for r in g[g.season_year.isin(SEASONS)].itertuples():
        cutoff = r.start - cut
        for tm in (r.home, r.away):
            s = stg.get((r.game_id, tm))
            listed = {} if s is None else {x.pid: x for x in s.itertuples() if x.pid}
            wrong = lg.misattributed(tm, cutoff, listed)
            for p in wrong: listed.pop(p)
            roster = lg.roster(tm, cutoff, listed)
            prof = [lg.profile(p, tm, cutoff) for p in roster]
            sup = [p for p in prof if p['history_n'] > 0 and p['supported']]
            ref = reference_minutes(sup) if sup else None
            missing = ((r.game_id, tm) not in rep) or s is None or len(s) == len(wrong)
            tg.append(dict(game_id=r.game_id, team=tm, variant=variant, cutoff=cutoff, block_missing=missing,
                           listed_n=0 if s is None else len(s), listed_unmatched=0 if s is None else int((s.pid == '').sum()),
                           roster_n=len(roster)))
            for p in prof:
                x = listed.get(p['canonical_player_id'])
                ros.append(dict(game_id=r.game_id, team=tm, pid=p['canonical_player_id'], status=x.status if x else 'not_listed',
                                reason_class=x.reason_class if x else '', base_minutes=(ref or {}).get(p['canonical_player_id'], 0.0)))
    return pd.DataFrame(tg), pd.DataFrame(ros)


def team_frame(g):
    rows = []
    for r in g.itertuples():
        for tm, opp, home, pf, pa in ((r.home, r.away, 1, r.home_score, r.away_score), (r.away, r.home, 0, r.away_score, r.home_score)):
            rows.append(dict(game_id=r.game_id, team=tm, opp=opp, start=r.start, margin=pf - pa, home=home))
    t = pd.DataFrame(rows).sort_values(['team', 'start'])
    grp = t.groupby('team')
    t['last10_margin'] = grp.margin.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    et = t.start.dt.tz_convert('US/Eastern').dt.tz_localize(None).dt.normalize()
    t['rest'] = (et - et.groupby(t.team).shift(1)).dt.days.clip(upper=3)
    return t


def build():
    t0 = time.time(); OUTD.mkdir(parents=True, exist_ok=True)
    g, box, zero_score = load_games_box()
    rows, blocks, info = load_reports(g)
    lg = League(g, box)
    parts = {}
    for v, cut in CUTS.items():
        best, st = status_at(g, rows, blocks, box, cut)
        parts[v] = team_games(g, lg, best, st, cut, v) + (st,)
        print(v, 'done', round(time.time() - t0), 's', flush=True)
    (a, ra, sa), (b, rb, sb) = parts['6h'], parts['90m']
    use90 = set(zip(a.game_id[a.block_missing], a.team[a.block_missing]))
    k = lambda d: pd.Series(list(zip(d.game_id, d.team)), index=d.index).isin(use90)
    tg = pd.concat([a[~k(a)], b[k(b)]]); ros = pd.concat([ra[~k(ra)], rb[k(rb)]])
    tg['season_year'] = tg.game_id.map(g.set_index('game_id').season_year)
    # features, defined as in the main build
    ref = ros[ros.base_minutes > 0].merge(tg[['game_id', 'team', 'cutoff', 'season_year', 'block_missing']], on=['game_id', 'team'])
    ref['eff_prior'] = [lg.eff_prior(p, c) for p, c in zip(ref.pid, ref.cutoff)]
    pos = box[box['min'] > 0]
    ps = pos.groupby(['pid', 'season_year']).agg(m=('min', 'sum'), e=('eff', 'sum'))
    ref = ref.merge((ps.e / ps.m).rename('eff_season').reset_index(), on=['pid', 'season_year'], how='left')
    p5 = ref.season_year.isin((2024, 2025, 2026))
    n_missing_eff = int(ref.eff_prior.isna().sum())
    for c in ('eff_prior', 'eff_season'):
        med_p5, med_all = ref.loc[p5, c].median(), ref[c].median()
        ref.loc[p5, c] = ref.loc[p5, c].fillna(med_p5); ref.loc[~p5, c] = ref.loc[~p5, c].fillna(med_all)
    played = set(zip(pos.game_id, pos.pid))
    ref['out_pre'] = ref.status == 'Out'
    ref['out_real'] = [(gm, p) not in played for gm, p in zip(ref.game_id, ref.pid)]
    ref['F1'] = ref.out_pre * ref.base_minutes * ref.eff_prior
    ref['F2'] = ref.out_real * ref.base_minutes * ref.eff_prior
    ref['F3'] = ref.out_pre * ref.base_minutes * ref.eff_season
    ref['F4'] = ref.out_real * ref.base_minutes * ref.eff_season
    V = ('F1', 'F2', 'F3', 'F4')
    T = ref.groupby(['game_id', 'team']).agg(**{x: (x, 'sum') for x in V}, n_out_pre=('out_pre', 'sum'), n_out_real=('out_real', 'sum'),
                                             block_missing=('block_missing', 'first')).reset_index()
    T = T.merge(team_frame(g)[['game_id', 'team', 'home', 'margin', 'last10_margin', 'rest']], on=['game_id', 'team'])
    H = T[T.home == 1].merge(T[T.home == 0], on='game_id', suffixes=('_h', '_a'))
    gi = g.set_index('game_id')
    G = pd.DataFrame(dict(game_id=H.game_id, season_year=H.game_id.map(gi.season_year),
                          game_date_et=H.game_id.map(gi.et_date), home=H.team_h, away=H.team_a,
                          margin=H.margin_h, any_block_missing=H.block_missing_h | H.block_missing_a,
                          strength_diff=H.last10_margin_h - H.last10_margin_a, rest_diff=H.rest_h.fillna(3) - H.rest_a.fillna(3),
                          n_out_pre_h=H.n_out_pre_h, n_out_pre_a=H.n_out_pre_a, n_out_real_h=H.n_out_real_h, n_out_real_a=H.n_out_real_a,
                          **{f'{x}_diff': H[f'{x}_h'] - H[f'{x}_a'] for x in V}))
    G = G.sort_values(['season_year', 'game_id']).reset_index(drop=True)
    G.to_csv(OUTD / 'bench_games_public.csv', index=False)
    tg.to_csv(OUTD / 'team_games_public.csv', index=False)
    ros.to_csv(OUTD / 'rosters_public.csv', index=False)
    pd.concat([sa.assign(variant='6h'), sb.assign(variant='90m')]).to_csv(OUTD / 'pregame_status_public.csv', index=False)
    # Table 1 analogue (2023-24...2025-26, team-games with a report): pregame status vs play
    R = ros.merge(tg[['game_id', 'team', 'season_year', 'block_missing']], on=['game_id', 'team'])
    R = R[R.season_year.isin((2024, 2025, 2026)) & ~R.block_missing].copy()
    R['played'] = [(gm, p) in played for gm, p in zip(R.game_id, R.pid)]
    rot = R[R.base_minutes > 0]; absent = rot[~rot.played]
    tab = [dict(status=s, player_games=int((R.status == s).sum()), played_pct=round(100 * R.played[R.status == s].mean(), 1),
                share_of_lost_ref_minutes_pct=round(100 * absent.base_minutes[absent.status == s].sum() / absent.base_minutes.sum(), 1))
           for s in ('Out', 'Doubtful', 'Questionable', 'Probable', 'Available', 'not_listed')]
    pd.DataFrame(tab).to_csv(OUTD / 'status_table_public.csv', index=False)
    stats = dict(**info, status_table=tab,
                 lost_ref_minutes_not_reported_out_pct=round(100 * absent.base_minutes[absent.status != 'Out'].sum() / absent.base_minutes.sum(), 1),
                 games=len(g), games_by_season=g.season_year.value_counts().sort_index().to_dict(),
                 games_zero_score_fixed_from_box=zero_score, bench_games=len(G), bench_by_season=G.season_year.value_counts().sort_index().to_dict(),
                 bench_excluded_block_missing=int(G.any_block_missing.sum()), team_games_from_90m=int(k(b).sum()),
                 team_games_block_missing=int(tg.block_missing.sum()), ref_rows=len(ref), ref_missing_eff_prior=n_missing_eff,
                 listed_rows=int(tg.listed_n.sum()), listed_unmatched=int(tg.listed_unmatched.sum()), seconds=round(time.time() - t0))
    json.dump(stats, open(OUTD / 'build_stats.json', 'w'), indent=1, default=str)
    print(json.dumps(stats, indent=1, default=str))


# ---------------------------------------------------------------- model (headline specification, no market)
def model(bench=None, out=None):
    """bench: a released benchmark CSV (quick check without rebuilding); out: where to write the JSON.
    Without arguments: the locally rebuilt table, written to results/paper2_public_path.json."""
    G = pd.read_csv(bench or OUTD / 'bench_games_public.csv', dtype={'game_id': str})
    G = G[~G.any_block_missing]
    BASE = ['strength_diff', 'rest_diff']
    VERS = {'F0': [], 'F1': ['F1_diff'], 'F2': ['F2_diff'], 'F3': ['F3_diff'], 'F4': ['F4_diff']}
    X = lambda d, cols: np.column_stack([np.ones(len(d))] + [d[c].values for c in BASE + cols])
    _in = Path(bench) if bench else OUTD / 'bench_games_public.csv'
    try:      # record where the input came from, relative to the repository, not the machine
        _in = _in.resolve().relative_to(ROOT)
    except ValueError:
        _in = _in.name
    res = dict(input=str(_in).replace('\\', '/'))
    if not bench and (OUTD / 'build_stats.json').exists(): res['stats'] = json.load(open(OUTD / 'build_stats.json'))
    for label, train in (('main_train_2024_2025', (2024, 2025)), ('train_2022_2025', (2022, 2023, 2024, 2025))):
        tr, ho = G[G.season_year.isin(train)].reset_index(drop=True), G[G.season_year == 2026].reset_index(drop=True)
        y = ho.margin.values
        pred = {k: X(ho, c) @ np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0] for k, c in VERS.items()}

        def stats(idx):
            r = {k: np.sqrt(np.mean((y[idx] - p[idx]) ** 2)) for k, p in pred.items()}
            out = {f'gain_{k}': r['F0'] - r[k] for k in ('F1', 'F2', 'F3', 'F4')}
            out['inflation_F2'] = r['F1'] - r['F2']
            out['ratio_F2'] = out['gain_F2'] / out['gain_F1'] if out['gain_F1'] > 0 else np.nan
            return out, r
        point, rmse = stats(np.arange(len(ho)))
        rng = np.random.default_rng(20260921)
        teams = ho.home.unique(); groups = {t: np.flatnonzero(ho.home.values == t) for t in teams}
        B = [stats(np.concatenate([groups[t] for t in rng.choice(teams, len(teams))]))[0] for _ in range(2000)]
        ci = {k: [float(np.nanpercentile([b[k] for b in B], q)) for q in (2.5, 97.5)] for k in point}
        res[label] = dict(n_train=len(tr), n_holdout=len(ho), holdout_rmse={k: round(float(v), 4) for k, v in rmse.items()},
                          estimates={k: dict(point=round(float(v), 4), ci95=[round(c, 4) for c in ci[k]]) for k, v in point.items()})
    dest = Path(out) if out else (ROOT / 'results/paper2_public_path_check.json' if bench else ROOT / 'results/paper2_public_path.json')
    json.dump(res, open(dest, 'w'), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)); print('written to', dest)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'parse': parse_step(float(sys.argv[2]) if len(sys.argv) > 2 else 600)
    elif cmd == 'build': build()
    elif cmd == 'model':
        a = sys.argv[2:]
        model(bench=a[a.index('--bench') + 1] if '--bench' in a else None, out=a[a.index('--out') + 1] if '--out' in a else None)
