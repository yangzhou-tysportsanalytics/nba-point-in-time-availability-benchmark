"""The deployment gap, pooled over the four rolling holdouts.

The paper calls this "the number a buyer should ask for" and then reports it only on one season and only as a
range across a grid of settings. Round 15 asked for it pooled, which is fair: it is the one quantity in the
paper addressed directly to someone deciding whether to buy a model, and a range over a grid is not an estimate.

The gap is what a model loses between its own backtest and its deployment. One model is trained on realized
participation. Its backtest is what it scores when it is also evaluated on participation -- the number a vendor
would show. Its deployment is what it scores when it is handed the pregame report instead, which is all it can
have at tip-off. The difference is the gap.

  python league/scripts/deployment_pooled.py -> results/paper2_deployment_pooled.json

Least squares, the paper's main model, on the two feature sets the paper uses: narrow (three differences) and
wide (the eight home and away values separately). Both of the targets of Section 3.1 are reported: the dyadic
cluster-robust interval for these four seasons, and the season-level interval for an NBA season, because the
calibration found a season component in quantities of this family and the four holdouts disagree.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[2]
SEASONS = (2023, 2024, 2025, 2026)
SEED = 20261005


def resolve_bench() -> tuple[Path, str]:
    """The same script serves the main build and the released one; the file present decides which.

    P2_BENCH overrides. The released build has fewer games, excludes a few differently, and carries only the
    difference features, so it runs the narrow set alone and is reported beside the main run, not for it.
    """
    env = os.environ.get('P2_BENCH')
    if env:
        return Path(env), 'custom'
    for rel, label in ((Path('data/league/paper2_ext/bench_games.csv'), 'main'),
                       (Path('data/bench_games_public.csv'), 'public')):
        if (ROOT / rel).exists():
            return ROOT / rel, label
    raise SystemExit('no benchmark file found: set P2_BENCH')


BENCH, BUILD = resolve_bench()
OUT = ROOT / f'results/paper2_deployment_pooled{"" if BUILD == "main" else "_" + BUILD}.json'

G = pd.read_csv(BENCH, dtype={'game_id': str})
G = G[~G.any_block_missing].reset_index(drop=True)

FEATURES = {
    'narrow': dict(base=['strength_diff', 'rest_diff'], f1=['F1_diff'], f2=['F2_diff']),
    'wide': dict(base=['last10_h', 'last10_a', 'rest_h', 'rest_a'],
                 f1=['F1_h', 'F1_a'], f2=['F2_h', 'F2_a']),
}
FEATURES = {k: v for k, v in FEATURES.items()
            if set(v['base'] + v['f1'] + v['f2']) <= set(G.columns)}
if 'narrow' not in FEATURES:
    raise SystemExit(f'{BENCH.name} carries none of the feature sets this script needs')


def design(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])


rows = {}
for fs, spec in FEATURES.items():
    parts = []
    for s in SEASONS:
        tr, ho = G[G.season_year < s], G[G.season_year == s].copy()
        c2, c1 = spec['base'] + spec['f2'], spec['base'] + spec['f1']
        # one model, trained on participation
        beta = np.linalg.lstsq(design(tr, c2), tr.margin.values, rcond=None)[0]
        ho['backtest'] = design(ho, c2) @ beta      # evaluated on participation: what a vendor shows
        ho['deployed'] = design(ho, c1) @ beta      # handed the pregame report instead: what it gets
        parts.append(ho)
    rows[fs] = pd.concat(parts, ignore_index=True)

# shared structure for the intervals
H = rows['narrow']
teams = np.array(sorted(set(H.home) | set(H.away)))
hi_, ai_ = np.searchsorted(teams, H.home.values), np.searchsorted(teams, H.away.values)
pair_code, _ = pd.factorize(pd.Series(np.minimum(hi_, ai_) * 1000 + np.maximum(hi_, ai_)))
n, n_teams, n_pair = len(H), len(teams), pair_code.max() + 1
rng = np.random.default_rng(SEED)
BOOT = np.empty((4000, n))
for r in range(4000):
    BOOT[r] = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
BSUM = BOOT.sum(1)


def rmse_diff(a, b):
    return float(np.sqrt(a.mean()) - np.sqrt(b.mean()))


def dyadic_se(a, b):
    ma, mb = a.mean(), b.mean()
    psi = (a - ma) / (2 * np.sqrt(ma)) - (b - mb) / (2 * np.sqrt(mb))
    S = np.zeros(n_teams); np.add.at(S, hi_, psi); np.add.at(S, ai_, psi)
    T = np.zeros(n_pair); np.add.at(T, pair_code, psi)
    return float(np.sqrt(max((float((S ** 2).sum()) - float((T ** 2).sum())) / n ** 2, 0.0)))


out = {'n_holdout_games': int(n), 'seasons': list(SEASONS), 'build': BUILD, 'bench_file': BENCH.name, 'model': 'least squares',
       'note': 'One model, trained on realized participation. Its backtest evaluates it on participation; its '
               'deployment hands it the pregame report. The gap is deployment RMSE minus backtest RMSE, so a '
               'positive number is what the buyer loses.',
       'feature_sets': {}}

print(f'{"features":10}{"backtest":>10}{"deployed":>10}{"gap":>9}{"dyadic 95%":>22}{"iid 95%":>22}')
for fs, H2 in rows.items():
    y = H2.margin.values
    sq_back = (y - H2.backtest.values) ** 2
    sq_dep = (y - H2.deployed.values) ** 2
    gap = rmse_diff(sq_dep, sq_back)                     # positive: deployment is worse
    se = dyadic_se(sq_dep, sq_back)
    dy = [round(gap - 1.959964 * se, 4), round(gap + 1.959964 * se, 4)]
    v = np.sqrt(BOOT @ sq_dep / BSUM) - np.sqrt(BOOT @ sq_back / BSUM)
    ii = [round(float(np.percentile(v, q)), 4) for q in (2.5, 97.5)]
    per = {str(s): round(rmse_diff(sq_dep[H2.season_year.values == s], sq_back[H2.season_year.values == s]), 4)
           for s in SEASONS}

    # the other target of Section 3.1: the season is the unit, so four of them and a t_3 interval
    v4 = np.array(list(per.values()))
    m4, sd4 = float(v4.mean()), float(v4.std(ddof=1))
    half = 3.182446 * sd4 / np.sqrt(4)
    season_ci = [round(float(m4 - half), 4), round(float(m4 + half), 4)]
    # the season-to-season spread, net of the sampling noise a single season carries
    within = se * np.sqrt(len(SEASONS))
    implied = float(np.sqrt(max(sd4 ** 2 - within ** 2, 0.0)))

    out['feature_sets'][fs] = dict(backtest_rmse=round(float(np.sqrt(sq_back.mean())), 4),
                                   deployed_rmse=round(float(np.sqrt(sq_dep.mean())), 4),
                                   gap=round(gap, 4), dyadic_se=round(se, 4), dyadic_ci95=dy, iid_ci95=ii,
                                   per_season=per, season_mean=round(m4, 4), season_sd=round(sd4, 4),
                                   season_ci95_t3=season_ci, implied_season_component=round(implied, 4),
                                   within_season_se=round(float(within), 4))
    print(f'{fs:10}{np.sqrt(sq_back.mean()):10.4f}{np.sqrt(sq_dep.mean()):10.4f}{gap:9.4f}'
          f'{str(dy):>22}{str(ii):>22}')
    print(f'{"":10}by season: {per}')
    print(f'{"":10}in an NBA season: {m4:.4f} {season_ci}  (season SD {sd4:.4f}, '
          f'within-season SE {within:.4f}, implied season component {implied:.4f})')

OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
print(f'\n-> {OUT.relative_to(ROOT)}')
