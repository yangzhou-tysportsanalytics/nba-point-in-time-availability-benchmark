"""The valuation without the 240-minute rescaling, on this repository's own rebuild.

The feature values an absent player at reference minutes times prior efficiency, and reference minutes are his
mean minutes over his last 20 games rescaled so the team's rotation sums to 240. The rescaling is a modelling
choice: it makes a thin rotation's players individually more valuable. This rebuilds the feature from the raw
last-20 minutes and reruns the rolling design, so the rescaling is the only thing switched off.

    python league/scripts/rebuild.py build          # writes the player-level table this needs
    python league/scripts/valuation_robustness.py   # -> results/paper2_valuation_robustness_public.json

The history comes from the shipped module. An earlier version of this analysis in the originating project
looked each player's minutes up among the players appearing in the game being predicted, so every absent
player -- which is the whole feature -- fell back to the build's own rescaled minutes, the intervention never
reached the rows it was meant to change, and the comparison it produced was withdrawn. The rule travels with
the code for that reason.
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
from history import last_games_before_cutoff                                 # noqa: E402

OUT = ROOT / 'results/paper2_valuation_robustness_public.json'
SEASONS = (2023, 2024, 2025, 2026)
BASE = ['strength_diff', 'rest_diff']
SEED, REPS = 20260921, 10000


def main() -> int:
    R = pi.ref_players()
    G = pi.games()
    tg = pi.team_games()[['game_id', 'team', 'cutoff_utc', 'variant']].drop_duplicates(['game_id', 'team'])
    R = R.merge(tg, on=['game_id', 'team'], how='left')
    R['cutoff'] = pd.to_datetime(R.cutoff_utc, utc=True, errors='coerce')
    cut_hours = R.variant.map({'6h': 6.0, '90m': 1.5}).fillna(6.0)
    R['start'] = R.cutoff + pd.to_timedelta(cut_hours, unit='h')

    box = pi.box_scores()
    box['min'] = pd.to_numeric(box['min'], errors='coerce')
    box = box[box['min'].fillna(0) > 0].copy()
    box['espn_player_id'] = box.espn_player_id.astype(str)
    box['game_id'] = box.game_id.astype(str)
    gstart = R[['game_id', 'start']].dropna().drop_duplicates('game_id')

    R['raw20'] = last_games_before_cutoff(box, R, gstart, 'min',
                                          cutoff_hours=cut_hours.values, max_age_days=365)
    no_history = int(R.raw20.isna().sum())
    R['raw20'] = R.raw20.fillna(R.base_minutes)   # no history at all: keep the build's value, and count it

    # a check the originating project also runs: rescaling the raw minutes to 240 should return the build's
    # reference minutes, which says the rescaling really is the only thing switched off
    tot = R.groupby(['game_id', 'team']).raw20.transform('sum')
    R['rescaled'] = R.raw20 * 240.0 / tot.replace(0, np.nan)
    diff = (R.rescaled - R.base_minutes).abs()
    check = {'rows': int(len(R)), 'rows_without_history': no_history,
             'rows_differing_above_1e6': int((diff > 1e-6).sum()),
             'max_abs_diff': round(float(diff.max(skipna=True)), 6)}

    R['val_noscale'] = R.raw20 * R.eff_prior
    for ver, flag in (('F1', 'out_pre'), ('F2', 'out_real')):
        v = pd.Series(np.where(R[flag].astype(bool), R.val_noscale, 0.0)).groupby([R.game_id, R.team]).sum()
        G[f'{ver}_ns_diff'] = [v.get((g, h), 0.0) - v.get((g, a), 0.0)
                               for g, h, a in zip(G.game_id, G.home, G.away)]
    H0 = G[~G.any_block_missing].reset_index(drop=True)

    VERS = {'F0': [], 'F1': ['F1_diff'], 'F2': ['F2_diff'],
            'F1_ns': ['F1_ns_diff'], 'F2_ns': ['F2_ns_diff']}
    X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])
    parts = []
    for s in SEASONS:
        tr, ho = H0[H0.season_year < s], H0[H0.season_year == s].copy()
        for k, c in VERS.items():
            ho['p_' + k] = X(ho, c) @ np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0]
        parts.append(ho)
    H = pd.concat(parts, ignore_index=True)
    y = H.margin.values
    SQ = {k: (y - H['p_' + k].values) ** 2 for k in VERS}

    code, keys = pd.factorize(H.home.astype(str) + '_' + H.season_year.astype(str))
    rng = np.random.default_rng(SEED)
    W = [np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)).astype(float)[code]
         for _ in range(REPS)]

    def stat(w, a, b):
        t = w.sum()
        return float(np.sqrt((w * SQ[a]).sum() / t) - np.sqrt((w * SQ[b]).sum() / t))

    res = {'build': 'public rebuild', 'reps': REPS, 'rescaling_check': check, 'estimates': {}}
    print('\nrescaling check:', json.dumps(check))
    print(f'{"contrast":26}{"point":>9}{"home 95%":>22}')
    for key, a, b in (('gain_F1_noscale', 'F0', 'F1_ns'), ('inflation_noscale', 'F1_ns', 'F2_ns'),
                      ('gain_F1', 'F0', 'F1'), ('inflation', 'F1', 'F2')):
        pt = stat(np.ones(len(y)), a, b)
        draws = [stat(w, a, b) for w in W]
        ci = [round(float(np.percentile(draws, q)), 4) for q in (2.5, 97.5)]
        res['estimates'][key] = {'point': round(pt, 4), 'ci95_home': ci}
        print(f'{key:26}{pt:9.4f}{str(ci):>22}')

    OUT.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(f'\n-> {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
