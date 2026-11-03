"""The same interval computed on home-side and on away-side clusters.

Published unchanged below its original header. It reads the originating project's paths, so
it is not rerun here; its output is `results/paper2_cluster_side.json`, which carries intervals and counts and no player rows.

The original header follows.

Paper 2, plan W: which team defines the cluster, and whether the placebo can choose between the two.

Section 3.1 reports every four-season quantity under a bootstrap that resamples home-team x season clusters,
and credits the placebo calibration with the choice. Section 3.2's central hedge follows: the report's value
is resolved and the premium is not, its lower bound being zero to within the bootstrap's Monte-Carlo error.

An away-team x season bootstrap is the mirror image of that design, with the same 120 clusters and the same
claim to respecting the dependence a game creates between its two teams, and it had never been computed.
This script computes it, and then asks the question that decides what the paper may say: can the placebo
discriminate between the two sides at all? Its generating equation is symmetric in home and away, so the
answer should be no -- but that is an argument, and the paper should rest on the experiment. So the placebo
is rerun with the away design added, on the same seed and the same replicate count, and the two coverages
are compared.

Rule W-b is the branch that costs this plan its claim: if the placebo scores the two sides differently by
more than the Monte-Carlo error of a coverage estimate, then the calibration can choose, the paper's existing
justification stands, and nothing here changes it.

  python league/scripts/p6_ds_cluster_side.py  -> results/paper2_cluster_side.json
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

EXT = ROOT / 'data/league/paper2_ext'
PLAN = ROOT / 'docs/plans/paper2_W_cluster_side.md'
PLAN_SHA = '6a50361451d3b730'
OUT = ROOT / 'results/paper2_cluster_side.json'
SEASONS = (2023, 2024, 2025, 2026)
BASE = ['strength_diff', 'rest_diff']
REPS, SEEDS = 10000, (20260921, 1, 2, 3, 4)
SIDES = ('home_season_cluster', 'away_season_cluster')

sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
if not sha.startswith(PLAN_SHA):
    raise SystemExit(f'the plan has changed since the rules were fixed: {sha[:16]} != {PLAN_SHA}')

G = pd.read_csv(EXT / 'bench_games.csv', dtype={'game_id': str})
G = G[~G.any_block_missing].reset_index(drop=True)
X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])

parts = []
for s in SEASONS:
    tr, ho = G[G.season_year < s], G[G.season_year == s].copy()
    for k, c in (('F0', []), ('F1', ['F1_diff']), ('F2', ['F2_diff'])):
        ho['p_' + k] = X(ho, c) @ np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0]
    parts.append(ho)
H = pd.concat(parts, ignore_index=True)
y = H.margin.values
sq = {k: (y - H['p_' + k].values) ** 2 for k in ('F0', 'F1', 'F2')}
keys = cluster_keys(H)

out = {'plan': 'docs/plans/paper2_W_cluster_side.md', 'plan_sha256': sha,
       'n': int(len(H)), 'reps': REPS, 'seeds': list(SEEDS),
       'n_home_clusters': int(keys['n_hs']), 'n_away_clusters': int(keys['n_as']),
       'note': 'The two designs are the same two lines of src/boot_designs.py applied to the home-side and '
               'away-side cluster codes, so a difference between the readings comes from the data.'}

point = {'gain_F1': float(np.sqrt(sq['F0'].mean()) - np.sqrt(sq['F1'].mean())),
         'inflation': float(np.sqrt(sq['F1'].mean()) - np.sqrt(sq['F2'].mean()))}
out['point'] = {k: round(v, 4) for k, v in point.items()}


def read(design: str) -> dict:
    lows = {k: [] for k in point}
    highs = {k: [] for k in point}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        draws = {k: np.empty(REPS) for k in point}
        for r, w in enumerate(weight_bank(rng, design, REPS, keys)):
            tot = w.sum()
            rm = {k: np.sqrt(np.sum(w * sq[k]) / tot) for k in sq}
            draws['gain_F1'][r] = rm['F0'] - rm['F1']
            draws['inflation'][r] = rm['F1'] - rm['F2']
        for k in point:
            lows[k].append(float(np.percentile(draws[k], 2.5)))
            highs[k].append(float(np.percentile(draws[k], 97.5)))
    res = {}
    for k in point:
        res[k] = {'ci95': [round(float(np.mean(lows[k])), 4), round(float(np.mean(highs[k])), 4)],
                  'lower_range': [round(min(lows[k]), 4), round(max(lows[k]), 4)],
                  'upper_range': [round(min(highs[k]), 4), round(max(highs[k]), 4)],
                  'lower_above_zero_on_all_seeds': bool(min(lows[k]) > 0)}
    return res


out['sides'] = {s: read(s) for s in SIDES}

# ---------------------------------------------------------------- plan W item 4: the ICC by side
# The loss difference whose interval is at stake, per game, for each quantity.
PERM, PRNG = 2000, np.random.default_rng(20260921 + 55)


def icc(v: np.ndarray, code: np.ndarray) -> float:
    """Share of variance between clusters: the one-way random-effects ICC, moment form."""
    g = pd.Series(v).groupby(code)
    k = g.size().values.astype(float)
    m = g.mean().values
    gm = v.mean()
    n_, ng = len(v), len(k)
    msb = float(np.sum(k * (m - gm) ** 2) / (ng - 1))
    msw = float(np.sum((v - pd.Series(v).groupby(code).transform('mean').values) ** 2) / (n_ - ng))
    k0 = (n_ - np.sum(k ** 2) / n_) / (ng - 1)
    num = msb - msw
    return float(num / (num + k0 * msw)) if (num + k0 * msw) != 0 else float('nan')


def permuted_icc(v: np.ndarray, code: np.ndarray, season: np.ndarray) -> tuple:
    """The same statistic with the values shuffled inside each season, so clusters hold no real structure."""
    out = np.empty(PERM)
    for i in range(PERM):
        w = v.copy()
        for s in np.unique(season):
            m = season == s
            w[m] = PRNG.permutation(w[m])
        out[i] = icc(w, code)
    return float(out.mean()), float(out.std(ddof=1))


_lossdiff = {'gain_F1': sq['F0'] - sq['F1'], 'inflation': sq['F1'] - sq['F2']}
_season = H.season_year.values
out['icc'] = {'permutations': PERM,
              'note': 'One-way ICC of the per-game loss difference, by side, against a within-season '
                      'permutation reference that keeps the season structure and the cluster sizes and '
                      'destroys any real clustering.'}
for q, v in _lossdiff.items():
    out['icc'][q] = {}
    for side, code in (('home', keys['hs_code']), ('away', keys['as_code'])):
        obs = icc(np.asarray(v, dtype=float), code)
        mu, sd = permuted_icc(np.asarray(v, dtype=float), code, _season)
        out['icc'][q][side] = {'icc': round(obs, 5), 'permuted_mean': round(mu, 5),
                               'permuted_sd': round(sd, 5),
                               'z': round((obs - mu) / sd, 2) if sd else float('nan')}

# ---------------------------------------------------------------- control
INF = json.load(open(ROOT / 'results/paper2_inference.json'))
_pub = INF['designs']['home_season_cluster']
want = {'gain_F1': _pub['gain_F1']['ci95'], 'inflation': _pub['infl']['ci95']}
got = {k: out['sides']['home_season_cluster'][k]['ci95'] for k in want}
out['control'] = {'published': want, 'got': got,
                  'reproduced': all(abs(got[k][i] - want[k][i]) < 2e-3 for k in want for i in (0, 1))}
if not out['control']['reproduced']:
    print(json.dumps(out['control'], indent=1))
    raise SystemExit('the home-side reading did not reproduce the published interval; nothing written')

# ---------------------------------------------------------------- can the placebo tell the sides apart?
CAL = json.load(open(ROOT / 'results/paper2_boot_calibration.json'))
out['placebo'] = {'note': 'Run by p6_ds_boot_calibration.py with away_season_cluster added to its design '
                          'list; both scenarios, same seed and replicate count as the published run.',
                  'from_file': 'results/paper2_boot_calibration.json'}
for q in ('gain_F1', 'infl'):
    out['placebo'][q] = {}
    for scen in ('no_team', 'fitted'):
        d = CAL['quantities'][q]['calibration'][scen]['designs']
        row = {s: d[s]['coverage'] for s in ('home_season_cluster', 'away_season_cluster') if s in d}
        if len(row) == 2:
            row['difference'] = round(row['away_season_cluster'] - row['home_season_cluster'], 4)
        out['placebo'][q][scen] = row

json.dump(out, open(OUT, 'w'), indent=1)

print(f"n = {out['n']:,};  home clusters {out['n_home_clusters']}, away clusters {out['n_away_clusters']}")
print(f"point estimates: report {point['gain_F1']:.4f}, premium {point['inflation']:.4f}")
print(f"control against results/paper2_inference.json: {out['control']['reproduced']}")
for s in SIDES:
    print(f"\n{s}")
    for k in ('gain_F1', 'inflation'):
        d = out['sides'][s][k]
        print(f"  {k:10} {d['ci95']}   lower across seeds {d['lower_range']}"
              f"   above zero on all five: {d['lower_above_zero_on_all_seeds']}")
print()
for q in ('gain_F1', 'infl'):
    for scen, row in out['placebo'][q].items():
        if isinstance(row, dict) and 'away_season_cluster' in row:
            print(f"  placebo {q:8} {scen:8} home {row['home_season_cluster']:.3f}  "
                  f"away {row['away_season_cluster']:.3f}  difference {row['difference']:+.4f}")
        elif isinstance(row, dict):
            print(f"  placebo {q:8} {scen:8} away design not in the calibration file yet")
print()
for q in ('gain_F1', 'inflation'):
    for side in ('home', 'away'):
        d = out['icc'][q][side]
        print(f"  ICC {q:10} {side:5} {d['icc']:+.5f}   permuted {d['permuted_mean']:+.5f} "
              f"(sd {d['permuted_sd']:.5f})   z {d['z']:+.2f}")
print(f"\n-> {OUT.relative_to(ROOT)}")
