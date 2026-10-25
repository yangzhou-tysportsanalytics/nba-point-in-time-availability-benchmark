"""Plan Y as it was run: what reading the report later is worth once the early report is read in full.

This is the main build's script, published unchanged below its original header so the analysis can be read
and checked line by line. It reads the originating project's paths and does **not** run against this
repository's rebuild, which makes it the one analysis in this release that cannot be rerun here. The reason
is the input: the all-status feature needs each listed player's status on the report selected at each cutoff,
and the rebuild writes its player-level table on the reader's own machine, so this repository holds no
rebuilt copy and no public-path result for it. `results/paper2_cutoff_f7.json` is the main build's result
file.

What can be checked here is the part that goes wrong quietly. The encoding is shipped as
`src/cutoff_encoding.py` with the thirteen tests it was built against, in
`league/tests/test_cutoff_encoding.py`, which run in this repository: a report printed exactly on the cutoff
counts, a team that had not filed is unknown rather than healthy, a player the report does not list is
charged nothing even though "Available" carries a sit-out rate of its own, a status that softens between two
cutoffs lowers the value instead of keeping it, and a status the league has not used before raises instead
of scoring zero.

The plan is `docs/plans/paper2_Y_f7_at_the_cutoffs.md` in the main project, pinned by hash below, and
`ANALYSIS_HISTORY.md` records its controls, its rules fixed before the run, and which one fired.

The original header follows.

Plan Y (docs/plans/paper2_Y_f7_at_the_cutoffs.md): what reading the report later is worth once the early
report is read in full.

  python league/scripts/p6_ds_cutoff_f7.py  -> results/paper2_cutoff_f7.json

The paper holds three of four cells. The cutoff curve re-reads only the **Out** status at each cutoff, so
`+0.067 (0.030--0.103)` is what waiting from six hours to one buys a model that reads the report as
out-or-not. Section 3.4's stronger baseline reads every status the report publishes, weighted by that
status's sit-out rate, but only at six hours. The missing cell decides whether waiting carries information a
full reading at six hours does not, or whether most of it is a status hardening from Questionable to Out,
which a full reading has already priced.

Conventions, following the cutoff curve so the two can be compared: the reference rotation, the reference
minutes and the prior efficiency are the six-hour build's, and only the status is re-read. The sit-out rates
are likewise held at their six-hour values --- rule Y-g --- and the version that re-estimates them at one
hour is reported beside it as a sensitivity.

Controls first, rules Y-a and Y-b: this script's own F1 path must reproduce the published cutoff curve at the
fourth decimal, and its F7 at six hours must reproduce the published strict F7 gain to within 0.002. Nothing
is interpreted if the first fails.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.boot_designs import cluster_keys as _cluster_keys, weight_bank as _weight_bank  # noqa: E402
from src.cutoff_encoding import NOT_LISTED, weight                                       # noqa: E402

PLAN = ROOT / 'docs/plans/paper2_Y_f7_at_the_cutoffs.md'
PLAN_SHA = '86e1f0558da1746a'
_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
if not _sha.startswith(PLAN_SHA):
    raise SystemExit(f'the plan has changed since the rules were fixed: {_sha[:16]} != {PLAN_SHA}')

EXT = ROOT / 'data/league/paper2_ext'
DATA = EXT / 'rebuild/data'
FS = EXT / 'full_series'
OUT = ROOT / 'results/paper2_cutoff_f7.json'
SEED, REPS = 20260921, 2000
SEASONS = (2023, 2024, 2025, 2026)
BASE = ['strength_diff', 'rest_diff']
STATUSES = ['Out', 'Doubtful', 'Questionable', 'Probable', 'Available']
PRIOR_A, PRIOR_N = 1.0, 2.0
CUTS = (6, 1)

# what the controls must reproduce, read off the committed result files before this script existed
CONTROL = {'gain_F1_6h': 0.1852, 'infl_6h': 0.0502, 'paired_F1': 0.0666,
           'paired_F1_ci': [0.0305, 0.1031], 'n': 4131, 'strict_F7_gain': 0.2198}

# ---------------------------------------------------------------- the build, as the cutoff curve loads it
G = pd.read_csv(EXT / 'bench_games.csv', dtype={'game_id': str})
R = pd.read_csv(EXT / 'bench_ref_players.csv', dtype={'game_id': str, 'espn_player_id': str})
games = pd.read_csv(DATA / 'games.csv', dtype={'game_id': str})
games['start'] = pd.to_datetime(games.start_utc, utc=True)
rows = pd.read_csv(FS / 'injury_rows.csv', dtype={'game_id': str})
blocks = pd.read_csv(FS / 'injury_blocks.csv', dtype={'game_id': str})
rows['ts'] = pd.to_datetime(rows.report_timestamp_et, utc=True)
blocks['ts'] = pd.to_datetime(blocks.report_timestamp_et, utc=True)

st = pd.concat([pd.read_csv(DATA / f, dtype={'game_id': str, 'espn_player_id': str})
                for f in ('pregame_status.csv', 'pregame_status_90m.csv')])
st = st.dropna(subset=['espn_player_id']).merge(games[['game_id', 'season_year']], on='game_id')
name_id = dict(zip(zip(st.team, st.season_year, st.player_name), st.espn_player_id))
rows = rows.merge(games[['game_id', 'season_year', 'start']], on='game_id')
blocks = blocks.merge(games[['game_id', 'start']], on='game_id')
listed = blocks[blocks.state == 'listed']

# The series holds 1.9 million rows and this analysis reads two cutoffs, so the selected report per
# team-game is found first and the rows are cut to those reports before any per-row work. Mapping names to
# player ids over the whole series, as the cutoff-curve script does, is most of its runtime.
def _selected(hours):
    b = listed[listed.ts <= listed.start - pd.Timedelta(hours=hours)]
    best = b.sort_values('ts').groupby(['game_id', 'team']).ts.max()
    return dict(zip(best.index, best.values)), set(best.index)


_SEL = {h: _selected(h) for h in CUTS}
_needed = {(g, t, ts) for h in CUTS for (g, t), ts in _SEL[h][0].items()}
rows = rows[[(g, t, ts) in _needed
             for g, t, ts in zip(rows.game_id, rows.team, rows.ts.values)]].reset_index(drop=True)
rows['pid'] = [name_id.get((t, s, n)) for t, s, n in zip(rows.team, rows.season_year, rows.player_name)]
print(f'  report rows on a selected report: {len(rows):,}', flush=True)

R = R.merge(games[['game_id', 'start']], on='game_id')
R['val'] = R.base_minutes * R.eff_prior
R['out_real'] = R.out_real.astype(bool)

# ---------------------------------------------------------------- the sit-out rates, at the six-hour cutoff
# Built exactly as p6_ds_pooled_secondary.py builds q_roll: among reference-rotation player-games whose
# six-hour status was s and whose game ended before this cutoff, the share who did not play, with a
# Beta(1,1) prior. Evaluated at every game's six-hour cutoff for every status, not only the status the
# player happened to carry there, because a later report can move him to any of them.
def rate_table(cutoff_per_game: pd.Series) -> pd.DataFrame:
    end = R.start + pd.Timedelta(hours=2.5)
    cuts = cutoff_per_game.reindex(G.game_id).values.astype('datetime64[ns]')
    tab = {}
    for s in STATUSES:
        d = R[R.status == s].assign(end=end[R.status == s]).sort_values('end')
        ends = d.end.values.astype('datetime64[ns]')
        cum = np.concatenate([[0], np.cumsum(d.out_real.values)])
        k = np.searchsorted(ends, cuts, side='right')
        tab[s] = (cum[k] + PRIOR_A) / (k + PRIOR_N)
    return pd.DataFrame(tab, index=G.game_id.values)


start_of = games.set_index('game_id').start
CUT6 = start_of - pd.Timedelta(hours=6)
Q6 = rate_table(CUT6)

# ---------------------------------------------------------------- the two encodings at a cutoff
_CACHE: dict = {}


def statuses_at(hours: float):
    """Per team-game, the status of every listed player on the last report filed by that cutoff.

    Returns the status frame and the set of team-games that had filed, so a team that had not is unknown
    rather than healthy. Selection is at or before the cutoff: most selected reports are printed exactly on
    the cutoff moment.
    """
    if hours in _CACHE:
        return _CACHE[hours]
    best, have = _SEL[hours] if hours in _SEL else _selected(hours)
    on = np.array([best.get((g, t)) == ts
                   for g, t, ts in zip(rows.game_id, rows.team, rows.ts.values)])
    keep = rows[on & rows.pid.notna().values]
    _CACHE[hours] = (keep[['game_id', 'team', 'pid', 'availability_status']].drop_duplicates(
        ['game_id', 'team', 'pid']), have)
    return _CACHE[hours]


def team_values(hours: float, encoding: str, rates: pd.DataFrame):
    """The absence value of each team-game under one encoding, summed over the reference rotation."""
    S, have = statuses_at(hours)
    m = R.merge(S, left_on=['game_id', 'team', 'espn_player_id'],
                right_on=['game_id', 'team', 'pid'], how='left')
    status = m.availability_status.fillna(NOT_LISTED).values
    cols = {k: rates[k].reindex(m.game_id.values).values for k in STATUSES}
    w = np.array([weight(s, {k: cols[k][i] for k in STATUSES}, encoding)
                  for i, s in enumerate(status)])
    v = pd.Series(m.val.values * w).groupby([m.game_id.values, m.team.values]).sum()
    return v, have


def fdiff(v: pd.Series) -> np.ndarray:
    d = dict(zip(v.index, v.values))
    return np.array([d.get((g, h), 0.0) - d.get((g, a), 0.0)
                     for g, h, a in zip(G.game_id, G.home, G.away)])


# ---------------------------------------------------------------- the sample, as the cutoff curve fixes it
_, have6 = statuses_at(6)
fixed = np.array([((g, h) in have6) and ((g, a) in have6)
                  for g, h, a in zip(G.game_id, G.home, G.away)]) & ~G.any_block_missing.values
H = G[fixed].reset_index(drop=True)

for enc, tag in (('out_only', 'F1'), ('all_statuses', 'F7')):
    for h in CUTS:
        v, _ = team_values(h, enc, Q6)
        H[f'{tag}_{h}'] = fdiff(v)[fixed]
# the sensitivity of rule Y-g: the rates re-estimated at the one-hour cutoff instead of held at six
Q1 = rate_table(start_of - pd.Timedelta(hours=1))
v, _ = team_values(1, 'all_statuses', Q1)
H['F7_1_reest'] = fdiff(v)[fixed]
# F2, the participation feature, for the six-hour control
H['F2_diff'] = G.F2_diff.values[fixed]


# ---------------------------------------------------------------- inference, as the cutoff curve reads it
def _fit(cols, mask=None):
    y_all, P, keep = [], {k: [] for k in cols}, []
    for s in SEASONS:
        tr = H[H.season_year < s] if mask is None else H[(H.season_year < s) & mask]
        sel = (H.season_year == s) if mask is None else ((H.season_year == s) & mask)
        ho = H[sel]
        X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])
        for k, c in cols.items():
            P[k].append(X(ho, c) @ np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0])
        y_all.append(ho.margin.values)
        keep.append(ho)
    return np.concatenate(y_all), {k: np.concatenate(v) for k, v in P.items()}, pd.concat(keep)


def _draws(Rk, reps=REPS, seed=SEED):
    rng = np.random.default_rng(seed)
    hs = (Rk.home + '_' + Rk.season_year.astype(str)).values
    code, keys = pd.factorize(pd.Series(hs))
    nk = len(keys)
    teams = np.array(sorted(set(Rk.home) | set(Rk.away)))
    hi_, ai_ = np.searchsorted(teams, Rk.home.values), np.searchsorted(teams, Rk.away.values)
    W1 = [np.bincount(rng.integers(0, nk, nk), minlength=nk).astype(float)[code] for _ in range(reps)]
    W2 = []
    for _ in range(reps):
        c = np.bincount(rng.integers(0, len(teams), len(teams)), minlength=len(teams)).astype(float)
        W2.append(c[hi_] * c[ai_])
    W3 = list(_weight_bank(rng, 'season_then_cluster', reps, _cluster_keys(Rk)))
    # The away mirror and the iid game bootstrap, which the calibration scores beside the reported design
    # and cannot order against it. The paper shows elsewhere that the side can move the premium across
    # zero, so the quantities it leads on are reported under all of them (eleventh review).
    _k = _cluster_keys(Rk)
    W4 = list(_weight_bank(rng, 'away_season_cluster', reps, _k))
    W5 = list(_weight_bank(rng, 'iid_game', reps, _k))
    return W1, W2, W3, W4, W5


def _ci(v):
    return [round(float(np.percentile(v, q)), 4) for q in (2.5, 97.5)]


def _rmse(y, p, w):
    return np.sqrt(np.sum(w * (y - p) ** 2) / w.sum())


def gain_and_inflation(col, mask=None):
    y, P, Rk = _fit({'F0': [], 'F': [col], 'F2': ['F2_diff']}, mask)
    W1, _, _, _, _ = _draws(Rk)

    def st(w):
        r = {k: _rmse(y, q, w) for k, q in P.items()}
        return r['F0'] - r['F'], r['F'] - r['F2']

    pt = st(np.ones(len(y)))
    B = [st(w) for w in W1]
    return dict(n=int(len(y)), gain=round(pt[0], 4), inflation=round(pt[1], 4),
                gain_ci=_ci([b[0] for b in B]), inflation_ci=_ci([b[1] for b in B]))


def paired(late, early, mask=None):
    """RMSE(early) - RMSE(late), inside the same draws: what the later report adds under this encoding."""
    y, P, Rk = _fit({'late': [late], 'early': [early]}, mask)
    W1, W2, W3, W4, W5 = _draws(Rk)

    def st(w):
        return _rmse(y, P['early'], w) - _rmse(y, P['late'], w)

    return dict(n=int(len(y)), point=round(st(np.ones(len(y))), 4),
                ci95_home=_ci([st(w) for w in W1]), ci95_two_way=_ci([st(w) for w in W2]),
                ci95_season=_ci([st(w) for w in W3]), ci95_away=_ci([st(w) for w in W4]),
                ci95_iid=_ci([st(w) for w in W5]))


def paired_difference(mask=None, late7='F7_1'):
    """Rule Y-e: (F7 late - F7 early) - (F1 late - F1 early), inside one set of draws.

    The two encodings' contrasts are each other's comparison, so the quantity the plan asks for is their
    difference in the same draw and not the overlap of two intervals.
    """
    y, P, Rk = _fit({'f7_late': [late7], 'f7_early': ['F7_6'],
                     'f1_late': ['F1_1'], 'f1_early': ['F1_6']}, mask)
    W1, W2, W3, W4, W5 = _draws(Rk)

    def st(w):
        r = {k: _rmse(y, q, w) for k, q in P.items()}
        return (r['f7_early'] - r['f7_late']) - (r['f1_early'] - r['f1_late'])

    return dict(n=int(len(y)), point=round(st(np.ones(len(y))), 4),
                ci95_home=_ci([st(w) for w in W1]), ci95_two_way=_ci([st(w) for w in W2]),
                ci95_season=_ci([st(w) for w in W3]), ci95_away=_ci([st(w) for w in W4]),
                ci95_iid=_ci([st(w) for w in W5]))


# ---------------------------------------------------------------- the archive-coverage subsample
_last_held = dict(zip(*(lambda s: (s.index, s.values))(
    listed.groupby(['game_id', 'team']).ts.max())))
_tip = dict(zip(*(lambda s: (s.index, s.values))(listed.groupby('game_id').start.max())))


def covered(hours):
    out = []
    delta = pd.Timedelta(hours=hours)
    for g, h, a in zip(G.game_id, G.home, G.away):
        t = _tip.get(g)
        if t is None:
            out.append(False)
            continue
        cut = pd.Timestamp(t) - delta
        out.append(all(_last_held.get((g, tm)) is not None and _last_held.get((g, tm)) > cut
                       for tm in (h, a)))
    return np.array(out)[fixed]


def transitions():
    """How the statuses move between the two cutoffs, for interpretation and not for inference."""
    S6, _ = statuses_at(6)
    S1, _ = statuses_at(1)
    k = ['game_id', 'team', 'pid']
    m = S6.merge(S1, on=k, how='outer', suffixes=('_6', '_1'))
    a = m.availability_status_6.fillna(NOT_LISTED)
    b = m.availability_status_1.fillna(NOT_LISTED)
    moved = m[a.values != b.values]
    am, bm = a[a.values != b.values], b[a.values != b.values]
    return dict(
        listed_rows_six_hours=int(len(S6)), listed_rows_one_hour=int(len(S1)),
        rows_that_change=int(len(moved)),
        not_listed_at_six_then_out=int(((am == NOT_LISTED) & (bm == 'Out')).sum()),
        listed_short_of_out_then_out=int((am.isin(STATUSES[1:]) & (bm == 'Out')).sum()),
        out_at_six_then_softened=int(((am == 'Out') & (bm != 'Out')).sum()),
        team_games_with_a_change=int(moved.groupby(['game_id', 'team']).ngroups))


def main() -> int:
    out = {'plan': 'docs/plans/paper2_Y_f7_at_the_cutoffs.md', 'plan_sha256': _sha,
           'seed': SEED, 'reps': REPS, 'fixed_games': int(fixed.sum()),
           'note': 'The rotation, the reference minutes, the prior efficiency and the per-status sit-out '
                   'rates are the six-hour build\'s; only the status is re-read at the later cutoff.'}

    # ---- controls (Y-a, Y-b)
    c1 = gain_and_inflation('F1_6')
    cp = paired('F1_1', 'F1_6')
    c7 = gain_and_inflation('F7_6')
    ctrl = dict(f1_six_hours=c1, f1_paired_one_hour=cp, f7_six_hours=c7, expected=CONTROL)
    ya = (c1['n'] == CONTROL['n'] and abs(c1['gain'] - CONTROL['gain_F1_6h']) < 5e-5
          and abs(c1['inflation'] - CONTROL['infl_6h']) < 5e-5
          and abs(cp['point'] - CONTROL['paired_F1']) < 5e-5
          and cp['ci95_home'] == CONTROL['paired_F1_ci'])
    yb = abs(c7['gain'] - CONTROL['strict_F7_gain']) <= 0.002
    ctrl['Y-a'] = 'pass' if ya else 'FAIL'
    ctrl['Y-b'] = 'pass' if yb else f"FAIL: F7 at six hours {c7['gain']} against {CONTROL['strict_F7_gain']}"
    out['controls'] = ctrl
    print(f"  Y-a control: {ctrl['Y-a']}  F1 6h {c1['gain']}/{c1['inflation']} on {c1['n']}, "
          f"paired {cp['point']} {cp['ci95_home']}", flush=True)
    print(f"  Y-b control: {ctrl['Y-b']}  F7 6h {c7['gain']} {c7['gain_ci']}", flush=True)
    if not ya:
        OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
        raise SystemExit('rule Y-a: the F1 path does not reproduce the published cutoff curve; '
                         'nothing is interpreted')

    # ---- the result
    cov1 = covered(1)
    out['full_sample'] = {
        'f1_paired': cp,
        'f7_paired': paired('F7_1', 'F7_6'),
        'difference': paired_difference(),
        'f7_paired_rates_reestimated': paired('F7_1_reest', 'F7_6'),
        'difference_rates_reestimated': paired_difference(late7='F7_1_reest')}
    out['archive_covered'] = {
        'n_games': int(cov1.sum()), 'share': round(float(cov1.mean()), 4),
        'f1_paired': paired('F1_1', 'F1_6', mask=cov1),
        'f7_paired': paired('F7_1', 'F7_6', mask=cov1),
        'difference': paired_difference(mask=cov1)}
    out['transitions'] = transitions()

    for where in ('full_sample', 'archive_covered'):
        d = out[where]
        print(f'\n  {where}: F1 paired {d["f1_paired"]["point"]} {d["f1_paired"]["ci95_home"]}, '
              f'F7 paired {d["f7_paired"]["point"]} {d["f7_paired"]["ci95_home"]}, '
              f'difference {d["difference"]["point"]} {d["difference"]["ci95_home"]}', flush=True)

    f7, f1 = out['full_sample']['f7_paired'], out['full_sample']['f1_paired']
    dif = out['full_sample']['difference']
    out['rule_verdicts'] = {
        'Y-c (most of waiting is hardening)':
            bool(f7['ci95_home'][0] <= 0 or f7['point'] < 0.5 * f1['point']),
        'Y-d (waiting carries news under both)':
            bool(f7['ci95_home'][0] > 0 and abs(f7['point'] - f1['point']) <= 0.02),
        'Y-e (the encodings cannot be ordered at this cutoff)':
            bool(dif['ci95_home'][0] <= 0 <= dif['ci95_home'][1])}
    print('\n  rule verdicts:', json.dumps(out['rule_verdicts']), flush=True)

    OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'\n-> {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
