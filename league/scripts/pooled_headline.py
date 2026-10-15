"""The pooled rolling headline, on this repository's own rebuild.

The paper leads on a pooled rolling-origin design: each of the four holdout seasons in turn, trained on all
earlier ones. The repository's data-and-code section says that analysis has its own released script, and until
now it did not -- `rebuild.py model` runs the single 2025-26 holdout only, which is the specification an
earlier version of the paper led with. This runs the pooled design, under both intervals the paper reports.

    python league/scripts/rebuild.py build      # writes the game-level table this reads
    python league/scripts/pooled_headline.py    # -> results/paper2_pooled_headline_public.json

The figure to compare with is the one the paper prints for the public path: the report worth 0.196
(0.131-0.260) and the premium 0.074 (0.012-0.135) on 4,835 holdout games, against 0.197 and 0.059 on the
paper's own build. The two builds differ in which games they can serve, not in the rule, and the reading that
should match is the one the paper states: the report's value clear of zero, the premium not resolved above
zero under the generalising design.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'league/scripts'))
import public_inputs as pi                                                   # noqa: E402

OUT = ROOT / 'results/paper2_pooled_headline_public.json'
SEASONS = (2023, 2024, 2025, 2026)
BASE = ['strength_diff', 'rest_diff']
SEED, REPS = 20260921, 10000
VERS = {'F0': [], 'F1': ['F1_diff'], 'F2': ['F2_diff']}


def main() -> int:
    G = pi.games()
    G = G[~G.any_block_missing].reset_index(drop=True)
    X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])
    parts = []
    for s in SEASONS:
        tr, ho = G[G.season_year < s], G[G.season_year == s].copy()
        if not len(tr) or not len(ho):
            continue
        for k, c in VERS.items():
            ho['p_' + k] = X(ho, c) @ np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0]
        parts.append(ho)
    H = pd.concat(parts, ignore_index=True)
    y = H.margin.values
    sq = {k: (y - H['p_' + k].values) ** 2 for k in VERS}
    print(f'pooled holdout games: {len(H):,} ' +
          '(' + ', '.join(f'{s}: {int((H.season_year == s).sum()):,}' for s in SEASONS) + ')')

    # the two designs the paper reports: home-team x season clusters for a statement about these four
    # seasons, and the seasons resampled before the clusters inside them for the average over seasons
    code, keys = pd.factorize(H.home.astype(str) + '_' + H.season_year.astype(str))
    seasons = sorted(H.season_year.unique())
    ids = {s: np.flatnonzero(pd.Series(keys).str.endswith('_' + str(s)).values) for s in seasons}
    rng = np.random.default_rng(SEED)
    banks = {'home': [], 'season': []}
    for _ in range(REPS):
        c = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)).astype(float)
        banks['home'].append(c[code])
    for _ in range(REPS):
        c = np.zeros(len(keys))
        for s in rng.choice(seasons, len(seasons)):
            c += np.bincount(rng.choice(ids[s], len(ids[s])), minlength=len(keys)).astype(float)
        banks['season'].append(c[code])

    def stat(w, a, b):
        t = w.sum()
        return float(np.sqrt((w * sq[a]).sum() / t) - np.sqrt((w * sq[b]).sum() / t))

    res = {'build': 'public rebuild', 'n': int(len(H)), 'reps': REPS, 'seed': SEED,
           'by_season': {str(s): int((H.season_year == s).sum()) for s in SEASONS},
           'rmse': {k: round(float(np.sqrt(sq[k].mean())), 4) for k in VERS},
           'estimates': {}}
    print(f'\n{"contrast":16}{"point":>9}{"these four seasons":>24}{"averaged over seasons":>26}')
    for lab, a, b in (('gain_F1', 'F0', 'F1'), ('inflation', 'F1', 'F2'), ('gain_F2', 'F0', 'F2')):
        e = {'point': round(stat(np.ones(len(y)), a, b), 4)}
        for name, W in banks.items():
            draws = [stat(w, a, b) for w in W]
            e['ci95_' + name] = [round(float(np.percentile(draws, q)), 4) for q in (2.5, 97.5)]
        res['estimates'][lab] = e
        print(f'{lab:16}{e["point"]:9.4f}{str(e["ci95_home"]):>24}{str(e["ci95_season"]):>26}')

    OUT.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(f'\n-> {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
