"""The baseline that needs no report: what a team already knows from its previous box score.

Published unchanged below its original header. It reads the originating project's paths, so
it is not rerun here; its output is `results/paper2_lagged_baseline.json` and `results/paper2_lagged_baseline_strict.json`, which carry RMSE differences, intervals and counts and no player rows.

The original header follows.

Does the pregame report beat a baseline that needs no report at all?

Round 15, item B: the paper shows what the official injury report is worth against a model with no availability
information. A referee asked the obvious harder question. A team's previous box score is public, costs nothing
and needs no report: a player who did not play last night is a fair guess to be out tonight. If the report does
not beat that, 0.197 is not evidence that the report carries information, only that absence does.

So this builds the lagged-participation feature -- the absence value of rotation players who did not play in
their team's previous game -- on the same rotation and the same valuation as F1, and runs the same rolling
design. Three contrasts matter:

  F0 against Flag   what last night's box score alone is worth
  Flag against F1   what the official report adds on top of it, which is the number the paper is missing
  F0 against F1     the paper's headline, for reference

  python league/scripts/p6_ds_lagged_baseline.py -> results/paper2_lagged_baseline.json

Intervals are the dyadic cluster-robust ones the paper reports for these four seasons, with an iid game
bootstrap beside them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / 'data/league/paper2_ext'
OUT = ROOT / 'results/paper2_lagged_baseline.json'   # reassigned below when P2_STRICT is set
# ---------------------------------------------------------------- plan V item V2: the strict benchmark
# With P2_STRICT=1 this analysis runs on the strict six-hour sample -- the games where both teams filed at
# least six hours out, no fallback accepted -- which is the benchmark Section 2.2 defines and the sample the
# paper leads on, instead of the sample that accepts a 90-minute report. The restriction applies to every
# season read, training included. Membership comes from results/paper2_strict_sample.json so that no second
# copy of the report-selection rule can drift from the first.
import os as _os
_STRICT = _os.environ.get('P2_STRICT') == '1'
_STRICT_IDS = set()
if _STRICT:
    import json as _json
    _STRICT_IDS = set(_json.load(open(ROOT / 'results/paper2_strict_sample.json'))['game_ids'])
_SFX = '_strict' if _STRICT else ''
OUT = ROOT / f'results/paper2_lagged_baseline{_SFX}.json'

SEASONS = (2023, 2024, 2025, 2026)
SEED = 20261005

G = pd.read_csv(EXT / 'bench_games.csv', dtype={'game_id': str})
P = pd.read_csv(EXT / 'bench_ref_players.csv', dtype={'game_id': str})

# ---------------------------------------------------------------- the lagged feature
# Corrected 2026-11-03, after an external check. Until then this read `out_real` from the player's
# previous row in the rotation table, ordered by game_id, under a comment asserting that the ESPN
# game_id increases with date and that the ordering had been checked against the recorded rest days.
# Neither held: no such check existed, the id is not monotone in date (Atlanta's game of 2023-10-25
# sorted after its game of 2023-11-29, five weeks later, and read its absences from it), and a
# player's previous row in this table is not the team's previous game whenever he was not in that
# game's reference rotation. On the strict holdout sample 308 games took at least one record from a
# later game and 201 read an absence flag from one. In a paper about information timing that is the
# defect it is about, so the feature is now built the way Section 3.4 describes it: the team's
# previous game in the same season by recorded tip-off time, and that game's box score decides who
# played. The diagnostic below goes into the result file, and league/tests/test_p6_ds_lagged_order.py
# fails if either defect returns.
P['value'] = P.base_minutes * P.eff_prior

_GS = pd.read_csv(EXT / 'rebuild/data/games.csv', dtype={'game_id': str}, usecols=['game_id', 'start_utc'])
_start = pd.to_datetime(_GS.set_index('game_id').start_utc, format='mixed', utc=True)
_tg = pd.concat([G[['game_id', 'season_year', 'home']].rename(columns={'home': 'team'}),
                 G[['game_id', 'season_year', 'away']].rename(columns={'away': 'team'})], ignore_index=True)
_tg['start'] = _tg.game_id.map(_start)
_tg = _tg.sort_values(['season_year', 'team', 'start'])
_tg['prev_game'] = _tg.groupby(['season_year', 'team'])['game_id'].shift(1)
_tg['prev_start'] = _tg.groupby(['season_year', 'team'])['start'].shift(1)

# Who played in a game, from its box score: no row, a DNP flag, or no minutes. That is how out_real is
# built in the rotation table, where the two agree on every one of its 178,421 rows.
_PG = pd.read_csv(EXT / 'rebuild/data/player_games.csv', dtype={'game_id': str},
                  usecols=['game_id', 'espn_player_id', 'did_not_play', 'min'])
_mins = pd.to_numeric(_PG['min'], errors='coerce').fillna(0.0)
_pl = _PG[(~_PG.did_not_play) & (_mins > 0)]
PLAYED = set(zip(_pl.game_id, _pl.espn_player_id))

P = P.merge(_tg[['game_id', 'team', 'prev_game', 'prev_start', 'start']], on=['game_id', 'team'], how='left')
P['missed_prev'] = [bool(g) and (g, p) not in PLAYED
                    for g, p in zip(P.prev_game.fillna(''), P.espn_player_id)]
lag = (P.assign(v=np.where(P.missed_prev, P.value, 0.0))
         .groupby(['game_id', 'team'], as_index=False)['v'].sum()
         .rename(columns={'v': 'lag_value'}))

_has = P.prev_game.notna()
_gap = (P.start - P.prev_start).dt.total_seconds() / 86400.0
_atl = P[(P.team == 'ATL') & (P.game_id == '401584691')]
ORDER_CHECK = {
    'rule': "the team's previous game in the same season, by recorded tip-off time; its box score "
            'decides who played',
    'rows': int(len(P)), 'rows_with_a_previous_game': int(_has.sum()),
    'rows_flagged_absent': int(P.missed_prev.sum()),
    'rows_where_the_previous_game_is_later_in_time': int((_gap < 0).sum()),
    'max_days_back': round(float(_gap.max()), 2),
    'regression_example': {
        'game_id': '401584691', 'team': 'ATL', 'date': '2023-10-25',
        'previous_game': (None if _atl.empty or pd.isna(_atl.prev_game.iloc[0])
                          else str(_atl.prev_game.iloc[0])),
        'previous_date': (None if _atl.empty or pd.isna(_atl.prev_start.iloc[0])
                          else str(_atl.prev_start.iloc[0])[:10]),
        'is_season_opener': bool(not _atl.empty and pd.isna(_atl.prev_game.iloc[0])),
        'note': "Atlanta's first game of 2023-24, so there is no previous game and the feature is zero. "
                'The game_id ordering used until 2026-11-03 gave it one: 401584143, played 2023-11-29, '
                'five weeks later.'},
}

G = G.merge(lag.rename(columns={'team': 'home', 'lag_value': 'lag_h'}), on=['game_id', 'home'], how='left')
G = G.merge(lag.rename(columns={'team': 'away', 'lag_value': 'lag_a'}), on=['game_id', 'away'], how='left')
G[['lag_h', 'lag_a']] = G[['lag_h', 'lag_a']].fillna(0.0)
G['Flag_diff'] = G.lag_h - G.lag_a
G = G[~G.any_block_missing]
if _STRICT:
    G = G[G.game_id.isin(_STRICT_IDS)]
G = G.reset_index(drop=True)

# ---------------------------------------------------------------- the same rolling design
BASE = ['strength_diff', 'rest_diff']
X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])
VERSIONS = {'F0': [], 'Flag': ['Flag_diff'], 'F1': ['F1_diff'], 'F2': ['F2_diff'],
            # the increment the paper's sentence claims: a model that keeps the lagged feature and
            # adds the report, against one that keeps the lagged feature alone. Nested, unlike the
            # substitution contrast between two alternative models.
            'Flag+F1': ['Flag_diff', 'F1_diff']}

parts = []
for s in SEASONS:
    tr, ho = G[G.season_year < s], G[G.season_year == s].copy()
    for k, c in VERSIONS.items():
        beta = np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0]
        ho['p_' + k] = X(ho, c) @ beta
    parts.append(ho)
H = pd.concat(parts, ignore_index=True)
y = H.margin.values
SQ = {k: (y - H['p_' + k].values) ** 2 for k in VERSIONS}
teams = np.array(sorted(set(H.home) | set(H.away)))
hi_, ai_ = np.searchsorted(teams, H.home.values), np.searchsorted(teams, H.away.values)
pair_code, _ = pd.factorize(pd.Series(np.minimum(hi_, ai_) * 1000 + np.maximum(hi_, ai_)))
n, n_teams, n_pair = len(y), len(teams), pair_code.max() + 1


def rmse_diff(a, b):
    return float(np.sqrt(a.mean()) - np.sqrt(b.mean()))


def dyadic_se(a, b):
    ma, mb = a.mean(), b.mean()
    psi = (a - ma) / (2 * np.sqrt(ma)) - (b - mb) / (2 * np.sqrt(mb))
    S = np.zeros(n_teams); np.add.at(S, hi_, psi); np.add.at(S, ai_, psi)
    T = np.zeros(n_pair); np.add.at(T, pair_code, psi)
    return float(np.sqrt(max((float((S ** 2).sum()) - float((T ** 2).sum())) / n ** 2, 0.0)))


def home_ci(a, b, reps=10000):
    """The design of Section 3.1: home-team x season clusters, at the paper's replicate count."""
    sys.path.insert(0, str(ROOT))
    from src.boot_designs import cluster_keys, weight_bank
    W = weight_bank(np.random.default_rng(SEED), 'home_season_cluster', reps, cluster_keys(H))
    v = [float(np.sqrt((w * a).sum() / w.sum()) - np.sqrt((w * b).sum() / w.sum())) for w in W]
    return [round(float(np.percentile(v, q)), 4) for q in (2.5, 97.5)]


def iid_ci(a, b, reps=4000):
    rng = np.random.default_rng(SEED)
    W = np.empty((reps, n))
    for r in range(reps):
        W[r] = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
    t = W.sum(1)
    v = np.sqrt(W @ a / t) - np.sqrt(W @ b / t)
    return [round(float(np.percentile(v, q)), 4) for q in (2.5, 97.5)]


CONTRASTS = [('F0_vs_Flag', 'F0', 'Flag', "what last night's box score alone is worth"),
             ('Flag_vs_F1', 'Flag', 'F1',
              'the report instead of the lagged feature: two alternative models, not an increment'),
             ('Flag_vs_JointFlagF1', 'Flag', 'Flag+F1',
              'what the report adds to a model that keeps the lagged feature: nested, so CW applies'),
             ('F0_vs_F1', 'F0', 'F1', "the paper's headline, for reference"),
             ('F1_vs_F2', 'F1', 'F2', 'the inflation, for reference')]

out = {'n_holdout_games': int(n), 'seasons': list(SEASONS), 'previous_game_rule': ORDER_CHECK,
       'note': 'Flag is the absence value of rotation players who did not play in their team previous game, '
               'on the same rotation and valuation as F1. The previous game is the team previous game by '
               'tip-off time and its box score decides who played (previous_game_rule). The first game of a '
               'season has no previous game and contributes nothing, which understates the baseline slightly.',
       'share_games_with_any_lagged_absence': round(float((G.Flag_diff != 0).mean()), 4),
       'contrasts': {}}

print(f'{"contrast":14}{"point":>8}{"home 95%":>22}{"dyadic 95%":>22}   what it is')
for key, ka, kb, what in CONTRASTS:
    a, b = SQ[ka], SQ[kb]
    pt, se = rmse_diff(a, b), dyadic_se(a, b)
    dy = [round(pt - 1.959964 * se, 4), round(pt + 1.959964 * se, 4)]
    ii = iid_ci(a, b)
    hm = home_ci(a, b)
    out['contrasts'][key] = dict(point=round(pt, 4), ci95_home=hm, dyadic_se=round(se, 4),
                                 dyadic_ci95=dy, iid_ci95=ii, description=what)
    print(f'{key:14}{pt:8.4f}{str(hm):>22}{str(dy):>22}   {what}')

# per-season, for the record
out['per_season'] = {}
for s in SEASONS:
    m = H.season_year.values == s
    out['per_season'][str(s)] = {k: round(rmse_diff(SQ[ka][m], SQ[kb][m]), 4)
                                 for k, ka, kb, _ in CONTRASTS}

# the nested pair gets the test a nested pair admits
sys.path.insert(0, str(ROOT))
from src.forecast_tests import clark_west, diebold_mariano, mean_se          # noqa: E402
from scipy import stats as _st                                              # noqa: E402

_hs = (H.home.astype(str) + '_' + H.season_year.astype(str)).values
_d = diebold_mariano(y, H['p_Flag'].values, H['p_Flag+F1'].values)
_f = clark_west(y, H['p_Flag'].values, H['p_Flag+F1'].values)
_td = float(_d.mean() / mean_se(_d, 'cluster', cluster=_hs))
_tf = float(_f.mean() / mean_se(_f, 'cluster', cluster=_hs))
out['nested_test_report_over_lag'] = {
    'note': 'Flag against Flag+F1, home-team x season clusters. The nested pair the sentence in Section 3.4 '
            'needs; DM is undersized against the larger model here, so CW is the fair test.',
    'dm_mean': round(float(_d.mean()), 4), 'dm_t': round(_td, 2),
    'dm_p_two_sided': round(float(2 * _st.norm.sf(abs(_td))), 4),
    'cw_mean': round(float(_f.mean()), 4), 'cw_t': round(_tf, 2),
    'cw_p_one_sided': round(float(_st.norm.sf(_tf)), 4)}
print(f"\nnested: report added to a model holding the lagged feature  "
      f"DM {_d.mean():+.4f} t {_td:+.2f} p {2 * _st.norm.sf(abs(_td)):.4f}   "
      f"CW {_f.mean():+.4f} t {_tf:+.2f} p {_st.norm.sf(_tf):.4f}")

OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
print(f'\n-> {OUT.relative_to(ROOT)}')
