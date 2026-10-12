"""The plus-minus valuation of Section 4.2, on this repository's own rebuild.

Per-minute box-score efficiency is replaced by the player's mean plus-minus per minute over his last 20 games
that had finished before the cutoff, winsorised with bounds taken from earlier seasons only. The rotation, the
reference minutes, the model, the rolling design and the bootstrap are unchanged, and the same valuation is
used for the compliant and the leaky feature.

This is the valuation that puts the premium on the other side of zero, so it is the one a reader is most
likely to want to run rather than take on trust. The CC0 box-score source carries plusMinusPoints, so it can
be run here in full.

    python league/scripts/rebuild.py build        # writes the player-level table this needs
    python league/scripts/impact_valuation.py     # -> results/paper2_impact_valuation_public.json

The history lookup is the shipped module rather than a reimplementation: an earlier version of this analysis
in the originating project looked a player's history up among the players appearing in the game being
predicted, and an absent player never does, so every absent player was valued at zero and the result had to be
withdrawn. The rule is part of what the paper claims, so it travels with the code.
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
from history import last_games_before_cutoff, season_prior_bounds            # noqa: E402

OUT = ROOT / 'results/paper2_impact_valuation_public.json'
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
    box['plus_minus'] = pd.to_numeric(box.plus_minus, errors='coerce')
    box = box[box['min'].fillna(0) > 0].copy()
    box['pm_per_min'] = box.plus_minus / box['min']
    box['espn_player_id'] = box.espn_player_id.astype(str)
    box['game_id'] = box.game_id.astype(str)
    gstart = R[['game_id', 'start']].dropna().drop_duplicates('game_id')
    box = box.merge(gstart, on='game_id', how='left')
    box = box.merge(G[['game_id', 'season_year']], on='game_id', how='left')

    # winsorise with bounds from earlier seasons only, so nothing from the holdout enters the valuation
    bounds = season_prior_bounds(box.dropna(subset=['season_year']), 'pm_per_min')
    lo = box.season_year.map({k: v[0] for k, v in bounds.items()})
    hi = box.season_year.map({k: v[1] for k, v in bounds.items()})
    box['pm_per_min'] = box.pm_per_min.clip(lo, hi)

    R['pm20'] = last_games_before_cutoff(box, R, gstart.rename(columns={'start': 'start'}), 'pm_per_min',
                                         min_periods=5, cutoff_hours=cut_hours.values, max_age_days=365)
    cover = float(R.pm20.notna().mean())
    R['pm20'] = R.pm20.fillna(0.0)   # fewer than five prior games: valued at the league-average impact, zero
    R['val_impact'] = R.base_minutes * R.pm20

    for ver, flag in (('F1', 'out_pre'), ('F2', 'out_real')):
        v = pd.Series(np.where(R[flag].astype(bool), R.val_impact, 0.0)).groupby([R.game_id, R.team]).sum()
        G[f'{ver}_pm_diff'] = [v.get((g, h), 0.0) - v.get((g, a), 0.0)
                               for g, h, a in zip(G.game_id, G.home, G.away)]
    H0 = G[~G.any_block_missing].reset_index(drop=True)

    VERS = {'F0': [], 'F1': ['F1_diff'], 'F2': ['F2_diff'],
            'F1_pm': ['F1_pm_diff'], 'F2_pm': ['F2_pm_diff']}
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

    res = {'build': 'public rebuild', 'reps': REPS,
           'diagnostics': {'players_with_history': round(cover, 4), 'games': int(len(H0)),
                           'sd_F2_pm': round(float(H0.F2_pm_diff.std()), 3),
                           'sd_F2_eff': round(float(H0.F2_diff.std()), 3)},
           'estimates': {}}
    print(f'\nplayers with five or more prior games: {cover:.1%}')
    print(f'{"contrast":26}{"point":>9}{"home 95%":>22}')
    for key, a, b in (('gain_F1_impact', 'F0', 'F1_pm'), ('inflation_impact', 'F1_pm', 'F2_pm'),
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
