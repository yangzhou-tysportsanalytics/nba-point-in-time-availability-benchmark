"""Selecting each report by the server's modification time instead of the time printed on it.

Published unchanged below its original header. It reads the originating project's paths, so
it is not rerun here; its output is `results/paper2_server_time.json`, which carries RMSE differences, intervals and counts and no player rows.

The original header follows.

Paper 2, plan V item V1: the two timestamp conventions on a sample that does not change between them.

Section 4.6 reports that taking the later of a report's printed header time and the server's Last-Modified
time moves the report's value from 0.185 to 0.137 and the premium from 0.050 to 0.099. That run evaluated the
strict feature on the games the *header* convention can serve, so the 428 team-games whose qualifying report
the strict rule removes contributed no absentees at all -- a missing report read as a healthy roster. Two
changes are mixed into one number, and the paper attributed the movement to the second without computing the
first.

This script holds the sample fixed: the holdout games where both teams have a qualifying report under both
conventions, with the same eligibility rule applied to training, and the difference between the conventions
taken inside each bootstrap draw. It also reports what the published run could not: how often the two
conventions select a different report and how far the absence value moves when they do, and the coverage of
each rule in its own right.

Control, before anything new is reported: the same code must reproduce the two published figures exactly,
the header convention's six-hour point and the strict run as it was computed. If either differs the script
refuses to write.

  python league/scripts/p6_ds_server_time.py  -> results/paper2_server_time.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.boot_designs import cluster_keys, weight_bank                         # noqa: E402
from src.server_time import common_sample, disagreement, paired_contrast, selected_report  # noqa: E402

EXT = ROOT / 'data/league/paper2_ext'
DATA = EXT / 'rebuild/data'
FS = EXT / 'full_series'
PLAN = ROOT / 'docs/plans/paper2_V_server_time_and_strict.md'
PLAN_SHA = 'eb870b9cd8ec4296'
OUT = ROOT / 'results/paper2_server_time.json'
SEED, REPS = 20260921, 2000
SEASONS = (2023, 2024, 2025, 2026)
MIN_COMMON = 3000          # plan V rule V-b

sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
if not sha.startswith(PLAN_SHA):
    raise SystemExit(f'the plan has changed since the rules were fixed: {sha[:16]} != {PLAN_SHA}')

# ---------------------------------------------------------------- inputs, as p6_ds_cutoff_curve.py reads them
S_ = dict(dtype={'game_id': str, 'espn_player_id': str})
G = pd.read_csv(EXT / 'bench_games.csv', dtype={'game_id': str})
R = pd.read_csv(EXT / 'bench_ref_players.csv', **S_)
games = pd.read_csv(DATA / 'games.csv', dtype={'game_id': str})
games['start'] = pd.to_datetime(games.start_utc, utc=True)
rows = pd.read_csv(FS / 'injury_rows.csv', dtype={'game_id': str})
blocks = pd.read_csv(FS / 'injury_blocks.csv', dtype={'game_id': str})
rows['ts'] = pd.to_datetime(rows.report_timestamp_et, utc=True)
blocks['ts'] = pd.to_datetime(blocks.report_timestamp_et, utc=True)

lm = pd.concat([pd.read_csv(FS / 'fetch_log.csv', dtype=str)[['file', 'last_modified']],
                pd.read_csv(EXT / 'lastmod_scan.csv', dtype=str)[['file', 'last_modified']]])
lm = lm[lm.last_modified.fillna('') != ''].drop_duplicates('file', keep='last')
lm['lm'] = pd.to_datetime(lm.last_modified, format='%a, %d %b %Y %H:%M:%S GMT', utc=True)
file_ts = rows[['file', 'ts']].drop_duplicates('file')
strict = file_ts.merge(lm[['file', 'lm']], on='file', how='left')
strict['ts_strict'] = strict[['ts', 'lm']].max(axis=1)
ts_map = dict(zip(strict.ts, strict.ts_strict))
lm_by_ts = dict(zip(strict.ts, strict.lm))

st = pd.concat([pd.read_csv(DATA / f, **S_) for f in ('pregame_status.csv', 'pregame_status_90m.csv')])
st = st.dropna(subset=['espn_player_id']).merge(games[['game_id', 'season_year']], on='game_id')
name_id = (st.drop_duplicates(['team', 'season_year', 'player_name'])
             .set_index(['team', 'season_year', 'player_name']).espn_player_id)
rows = rows.merge(games[['game_id', 'season_year', 'start']], on='game_id')
rows['pid'] = [name_id.get((t, s, n)) for t, s, n in zip(rows.team, rows.season_year, rows.player_name)]
blocks = blocks.merge(games[['game_id', 'start']], on='game_id')
listed = blocks[blocks.state == 'listed']
R['val'] = (R.base_minutes * R.eff_prior).values

# the report series the selection rule reads, with both times on every row
LISTED = listed[['game_id', 'team', 'ts', 'start']].copy()
LISTED['lm'] = LISTED.ts.map(lm_by_ts)


def feature(hours: float, convention: str) -> tuple[pd.Series, dict]:
    """Per team-game absence value under one convention, and which report each team-game selected."""
    sel = selected_report(LISTED, hours=hours, convention=convention)
    rr = rows.copy()
    # a row belongs to the selected report when its own effective time matches that team-game's selection,
    # which is the published rule; the selection is keyed by the report's header time
    rr['sel_header'] = [sel.get((g, t)) for g, t in zip(rr.game_id, rr.team)]
    on = rr[(rr.ts == rr.sel_header) & rr.pid.notna()]
    out = on[on.availability_status == 'Out'][['game_id', 'team', 'pid']].drop_duplicates()
    o = set(zip(out.game_id, out.team, out.pid))
    flag = np.array([(g, t, p) in o for g, t, p in zip(R.game_id, R.team, R.espn_player_id)])
    v = pd.Series(np.where(flag, R.val, 0.0)).groupby([R.game_id, R.team]).sum()
    n_out = pd.Series(flag.astype(float)).groupby([R.game_id, R.team]).sum()
    return v, dict(selection=sel, n_out=n_out)


def gdiff(v: pd.Series, frame: pd.DataFrame) -> np.ndarray:
    return np.array([v.get((g, h), 0.0) - v.get((g, a), 0.0)
                     for g, h, a in zip(frame.game_id, frame.home, frame.away)])


BASE = ['strength_diff', 'rest_diff']
v_head, aux_head = feature(6, 'header')
v_srv, aux_srv = feature(6, 'server')
have_head, have_srv = set(aux_head['selection']), set(aux_srv['selection'])

# the published sample: both teams served under the header convention
pub = np.array([((g, h) in have_head) and ((g, a) in have_head)
                for g, h, a in zip(G.game_id, G.home, G.away)]) & ~G.any_block_missing.values
# plan V's sample: both teams served under *both* conventions
com = common_sample(G, have_head, have_srv) & ~G.any_block_missing.values


def fit_and_score(mask: np.ndarray, f1_values: pd.Series, reps: int = REPS) -> dict:
    """Rolling-origin RMSEs and the home-cluster interval for one convention on one sample."""
    H = G[mask].reset_index(drop=True)
    H['F1'] = gdiff(f1_values, H)
    y, P, keep = [], {k: [] for k in ('F0', 'F1', 'F2')}, []
    X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])
    for s in SEASONS:
        tr, ho = H[H.season_year < s], H[H.season_year == s]
        for k, c in (('F0', []), ('F1', ['F1']), ('F2', ['F2_diff'])):
            P[k].append(X(ho, c) @ np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0])
        y.append(ho.margin.values)
        keep.append(ho)
    y = np.concatenate(y)
    P = {k: np.concatenate(v) for k, v in P.items()}
    Rr = pd.concat(keep)
    sq = {k: (y - P[k]) ** 2 for k in P}
    rng = np.random.default_rng(SEED)
    W = weight_bank(rng, 'home_season_cluster', reps, cluster_keys(Rr))

    def st(w):
        tot = w.sum()
        r = {k: np.sqrt(np.sum(w * sq[k]) / tot) for k in sq}
        return r['F0'] - r['F1'], r['F1'] - r['F2']

    pt = st(np.ones(len(y)))
    B = [st(w) for w in W]
    ci = lambda v: [round(float(np.percentile(v, q)), 4) for q in (2.5, 97.5)]
    return {'n': int(len(y)), 'gain_F1': round(pt[0], 4), 'inflation': round(pt[1], 4),
            'gain_F1_ci': ci([b[0] for b in B]), 'inflation_ci': ci([b[1] for b in B]),
            'sq': sq, 'rows': Rr}


out = {'plan': 'docs/plans/paper2_V_server_time_and_strict.md', 'plan_sha256': sha,
       'seed': SEED, 'reps': REPS,
       'note': 'The header convention uses a report\'s printed time; the server convention uses the later '
               'of that and the file\'s Last-Modified time. The published run in Section 4.6 evaluated the '
               'server-time feature on the header convention\'s sample, so a team whose qualifying report '
               'the server rule removes contributed no absentees. The common sample keeps only games where '
               'both teams are served under both conventions.'}

# ---------------------------------------------------------------- control: reproduce what is published
pub_head = fit_and_score(pub, v_head)
pub_srv = fit_and_score(pub, v_srv)
CC = json.load(open(ROOT / 'results/paper2_cutoff_curve.json'))
want_head = CC['curve']['6']['pooled_rolling']
want_srv = CC['strict_lastmodified_6h']['pooled_rolling']
control = {}
for lab, got, want in (('header_six_hour', pub_head, want_head), ('server_time_as_published', pub_srv, want_srv)):
    control[lab] = {
        'reproduced': all(abs(got[k] - want[k]) < 5e-4 for k in ('gain_F1', 'inflation'))
                      and got['n'] == want['n'],
        'got': {k: got[k] for k in ('n', 'gain_F1', 'inflation')},
        'published': {k: want[k] for k in ('n', 'gain_F1', 'inflation')}}
out['control'] = control
if not all(c['reproduced'] for c in control.values()):
    print(json.dumps(control, indent=1))
    raise SystemExit('the control did not reproduce the published figures; nothing written')

# ---------------------------------------------------------------- V1a: both conventions on the common sample
com_head = fit_and_score(com, v_head)
com_srv = fit_and_score(com, v_srv)
out['common_sample'] = {
    'n_games': int(com.sum()),
    'n_holdout_rows': com_head['n'],
    'header': {k: com_head[k] for k in ('n', 'gain_F1', 'inflation', 'gain_F1_ci', 'inflation_ci')},
    'server': {k: com_srv[k] for k in ('n', 'gain_F1', 'inflation', 'gain_F1_ci', 'inflation_ci')}}

rng = np.random.default_rng(SEED)
W = weight_bank(rng, 'home_season_cluster', REPS, cluster_keys(com_head['rows']))
for lab, a, b in (('gain_F1', 'F0', 'F1'), ('inflation', 'F1', 'F2')):
    pc = paired_contrast(com_srv['sq'][a], com_srv['sq'][b], com_head['sq'][a], com_head['sq'][b], W)
    out['common_sample'][f'server_minus_header_{lab}'] = {
        'point': round(pc['point'], 4), 'ci95_home': [round(x, 4) for x in pc['ci95']]}

# How far the server time sits after the printed one, over every file that carries both. The paper said
# "never more than 59" seconds; twelve files exceed it and one exceeds an hour, on the daylight-saving
# fall-back date, and two are served a second before their printed time (review, 2026-10-16).
_pair = strict.dropna(subset=['ts', 'lm'])
_d = (_pair.lm - _pair.ts).dt.total_seconds()
# One file sits on the US daylight-saving fall-back date, when 01:00 local happens twice: its header is
# recorded in EDT and the server wrote it during the EST hour, so the hour between them is which 1 a.m. the
# header was assigned to, not a delay. Reading that one file in the later hour, the largest lag is the 142
# seconds paper2_publication_times.json also reports, so the two measurements agree.
_dst = _d > 3000
_resolved = _d.where(~_dst, _d - 3600)
out['server_lag_seconds'] = {
    'files': int(len(_d)),
    'median': round(float(_resolved.median()), 1),
    'max': round(float(_resolved.max()), 1),
    'max_file': str(_pair.loc[_resolved.idxmax(), 'file']),
    'over_59s': int((_resolved > 59).sum()),
    'served_before_printed': int((_resolved < 0).sum()),
    'min': round(float(_resolved.min()), 1),
    'ambiguous_hour_files': int(_dst.sum()),
    'ambiguous_hour_file': str(_pair.loc[_d.idxmax(), 'file']) if _dst.any() else '',
    'raw_max_before_resolving': round(float(_d.max()), 1),
    'note': 'The selection rule takes the later of the two times either way, so no estimate depends on '
            'this. The ambiguous-hour file is on the fall-back date; its apparent hour is a clock, not a '
            'delay.'}

# ---------------------------------------------------------------- V1b: what each rule covers
tg = pd.concat([pd.read_csv(DATA / f'team_games_{s}_hybrid.csv', dtype={'game_id': str})
                for s in range(2022, 2027)], ignore_index=True)
tg_in = tg[tg.game_id.isin(set(G.game_id[~G.any_block_missing]))]
out['coverage'] = {
    'team_games_in_scope': int(len(tg_in)),
    'header_served': len(have_head), 'server_served': len(have_srv),
    'header_share': round(len(have_head) / len(tg_in), 4),
    'server_share': round(len(have_srv) / len(tg_in), 4),
    'lost_to_server_rule': int(len(have_head - have_srv)),
    'gained_by_server_rule': int(len(have_srv - have_head)),
    'games_both_teams_header': int(pub.sum()),
    'games_both_teams_server': int((common_sample(G, have_srv, have_srv) & ~G.any_block_missing.values).sum()),
    'games_both_teams_both_conventions': int(com.sum())}

# ---------------------------------------------------------------- V1c: how far the features move
dis = disagreement(aux_head['selection'], aux_srv['selection'])
shared = set(aux_head['selection']) & set(aux_srv['selection'])
changed = [k for k in shared if aux_head['selection'][k] != aux_srv['selection'][k]]
dv = np.array([float(v_srv.get(k, 0.0) - v_head.get(k, 0.0)) for k in changed])
do = np.array([float(aux_srv['n_out'].get(k, 0.0) - aux_head['n_out'].get(k, 0.0)) for k in changed])
q = lambda a: {'mean': round(float(a.mean()), 4), 'median': round(float(np.median(a)), 4),
               'p05': round(float(np.percentile(a, 5)), 4), 'p95': round(float(np.percentile(a, 95)), 4),
               'share_zero': round(float((np.abs(a) < 1e-9).mean()), 4),
               'mean_abs': round(float(np.abs(a).mean()), 4)} if len(a) else {}
out['disagreement'] = {**dis, 'absence_value_change_on_changed_team_games': q(dv),
                       'out_count_change_on_changed_team_games': q(do)}

# ---------------------------------------------------------------- the rules
mc = 0.005          # Section 3.1's Monte-Carlo error of a bound at this replicate count
infl = out['common_sample']['server_minus_header_inflation']
verdicts = {}
if out['common_sample']['n_games'] < MIN_COMMON:
    verdicts['V-b: the common sample is large enough to attribute'] = False
    out['rule_V_b_failed'] = (f"the common sample has {out['common_sample']['n_games']} games, below the "
                              f"{MIN_COMMON} the plan fixed, so the comparison is underpowered and no "
                              f"attribution is made either way")
else:
    verdicts['V-b: the common sample is large enough to attribute'] = True
    moved = abs(infl['point']) > mc
    # stated so that True is the outcome the paper's existing sentence needs, and False withdraws it
    verdicts['V-a: the timestamp convention leaves the premium unchanged on a fixed sample'] = bool(not moved)
    out['rule_V_a'] = (
        'The convention moves the premium on a fixed sample, so the attribution of the published rise to '
        'missing-report handling is withdrawn and both conventions are reported on the common sample.'
        if moved else
        'The convention does not move the premium by more than the Monte-Carlo error of a bound on a fixed '
        'sample, so the published rise is the missing-report handling and the paper may say so.')
out['rule_verdicts'] = verdicts

for k in ('common_sample',):
    pass
json.dump(out, open(OUT, 'w'), indent=1, default=str)

print(f"control: header {control['header_six_hour']['reproduced']}, "
      f"server {control['server_time_as_published']['reproduced']}")
print(f"\ncoverage: header serves {out['coverage']['header_served']:,} team-games, "
      f"server {out['coverage']['server_served']:,} of {out['coverage']['team_games_in_scope']:,}; "
      f"the server rule loses {out['coverage']['lost_to_server_rule']} and gains "
      f"{out['coverage']['gained_by_server_rule']}")
print(f"games: {out['coverage']['games_both_teams_header']:,} under the header rule, "
      f"{out['coverage']['games_both_teams_server']:,} under the server rule, "
      f"{out['coverage']['games_both_teams_both_conventions']:,} under both")
print(f"\ndisagreement: {dis['different_report']:,} of {dis['team_games_compared']:,} team-games select a "
      f"different report ({100 * dis['share_different']:.1f}%)")
if len(dv):
    print(f"  absence value moves by {q(dv)['mean_abs']:.3f} on average there, "
          f"{100 * q(dv)['share_zero']:.1f}% not at all; Out count by {q(do)['mean_abs']:.3f}")
print(f"\non the published sample ({pub_head['n']:,} holdout games):")
print(f"  header  report {pub_head['gain_F1']:+.4f}  premium {pub_head['inflation']:+.4f}")
print(f"  server  report {pub_srv['gain_F1']:+.4f}  premium {pub_srv['inflation']:+.4f}   <- mixes the two changes")
print(f"\non the common sample ({com_head['n']:,} holdout games):")
print(f"  header  report {com_head['gain_F1']:+.4f} {com_head['gain_F1_ci']}  "
      f"premium {com_head['inflation']:+.4f} {com_head['inflation_ci']}")
print(f"  server  report {com_srv['gain_F1']:+.4f} {com_srv['gain_F1_ci']}  "
      f"premium {com_srv['inflation']:+.4f} {com_srv['inflation_ci']}")
for lab in ('gain_F1', 'inflation'):
    d = out['common_sample'][f'server_minus_header_{lab}']
    print(f"  paired server - header, {lab}: {d['point']:+.4f} {d['ci95_home']}")
print()
for k, v in verdicts.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print(f"\n-> {OUT.relative_to(ROOT)}")
