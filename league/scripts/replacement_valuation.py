"""The replacement-level valuation of Section 4.2, on this repository's own rebuild.

Both of the paper's other valuations charge an absence at the absent player's whole output. A team does not
play short-handed, so the cost of an absence is the gap between the player and whoever absorbs his minutes.
This values a rotation player at

    base_minutes * (eff_prior - eff_replacement)

with the replacement level read from the same history eff_prior uses and nothing else: rotation player-games
whose own game had settled -- a game counts as known three hours after tip-off -- before this team-game's
cutoff, within the preceding 365 days. A level computed from the whole sample would carry the holdout into the
feature. The level is the minutes-weighted 20th percentile of per-minute efficiency over that window, and it
is not clipped, so a player below replacement level gets a negative value: clipping would assume in advance
that no absence can help a team.

    python league/scripts/rebuild.py build            # writes the player-level table this needs
    python league/scripts/replacement_valuation.py    # -> results/paper2_replacement_valuation_public.json

The numbers will not match the paper's to the last digit and are not meant to. This build has 6,146 games
against 6,153, excludes a few differently, and has no transaction log, so its participation feature sits a
little higher; the paper reports that comparison in its public-rebuild section. What should match is the
reading: the report's value positive and clear of zero, the premium not resolved above zero.
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

OUT = ROOT / 'results/paper2_replacement_valuation_public.json'
SEASONS = (2023, 2024, 2025, 2026)
BASE = ['strength_diff', 'rest_diff']
SEED, REPS = 20260921, 10000
SETTLE_HOURS, WINDOW_DAYS, Q = 3, 365, 0.20


def weighted_quantile(values, weights, q):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return float('nan')
    v, w = v[keep], w[keep]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, q * cum[-1], side='left')])


def levels_by_cutoff(history: pd.DataFrame, cutoffs, q=Q) -> pd.Series:
    """One replacement level per distinct cutoff; team-games share cutoffs, and the window is the only input."""
    h = history.dropna(subset=['eff_prior', 'minutes']).sort_values('start')
    end = pd.DatetimeIndex(h['start']).tz_localize(None).to_numpy()
    settle = end + np.timedelta64(SETTLE_HOURS, 'h')
    vals, mins = h['eff_prior'].to_numpy(float), h['minutes'].to_numpy(float)
    out = {}
    for cut in pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(cutoffs)))):
        c = np.datetime64(cut.tz_convert(None) if cut.tzinfo else cut)
        hi = np.searchsorted(settle, c, side='right')
        lo = np.searchsorted(end, c - np.timedelta64(WINDOW_DAYS, 'D'), side='left')
        out[cut] = weighted_quantile(vals[lo:hi], mins[lo:hi], q) if hi > lo else float('nan')
    return pd.Series(out)


def main() -> int:
    R = pi.ref_players()
    G = pi.games()
    tg = pi.team_games()[['game_id', 'team', 'cutoff_utc', 'variant']].drop_duplicates(['game_id', 'team'])
    games = G[['game_id', 'game_date_et']].copy()

    R = R.merge(tg, on=['game_id', 'team'], how='left')
    R['cutoff'] = pd.to_datetime(R.cutoff_utc, utc=True, errors='coerce')
    # the window is measured from tip-off, because a game counts as known three hours after it
    # tip-off from the cutoff and the rule that set it, exactly: six hours for the 6h variant, ninety
    # minutes for the fallback. Assuming six hours for both would put the fallback games 4.5 hours late.
    R['start'] = R.cutoff + pd.to_timedelta(R.variant.map({'6h': 6.0, '90m': 1.5}).fillna(6.0), unit='h')
    hist = R.dropna(subset=['cutoff', 'eff_prior']).rename(columns={'base_minutes': 'minutes'})
    levels = levels_by_cutoff(hist[['start', 'minutes', 'eff_prior']], R.cutoff.dropna().unique())
    R['level'] = R.cutoff.map(levels)
    R['val_rep'] = R.base_minutes * (R.eff_prior - R.level)

    diag = {'rotation_rows': int(len(R)), 'rows_without_a_level': int(R.level.isna().sum()),
            'level_mean': round(float(R.level.mean()), 5),
            'share_below_replacement': round(float((R.val_rep < 0).mean()), 4)}

    for ver, flag in (('F1', 'out_pre'), ('F2', 'out_real')):
        v = pd.Series(np.where(R[flag].astype(bool), R.val_rep, 0.0)).groupby([R.game_id, R.team]).sum()
        G[f'{ver}_rep_diff'] = [v.get((g, h), 0.0) - v.get((g, a), 0.0)
                                for g, h, a in zip(G.game_id, G.home, G.away)]
    H0 = G[~G.any_block_missing].reset_index(drop=True)
    diag['games'] = int(len(H0))
    diag['corr_F2_rep_with_existing'] = round(float(np.corrcoef(H0.F2_rep_diff, H0.F2_diff)[0, 1]), 4)

    VERS = {'F0': [], 'F1': ['F1_diff'], 'F2': ['F2_diff'],
            'F1_rep': ['F1_rep_diff'], 'F2_rep': ['F2_rep_diff']}
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

    # home-team x season clusters, the design the paper reports for a quantity stated for these four seasons
    code, keys = pd.factorize(H.home.astype(str) + '_' + H.season_year.astype(str))
    rng = np.random.default_rng(SEED)
    W = [np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)).astype(float)[code]
         for _ in range(REPS)]

    def stat(w, a, b):
        t = w.sum()
        return float(np.sqrt((w * SQ[a]).sum() / t) - np.sqrt((w * SQ[b]).sum() / t))

    res = {'build': 'public rebuild', 'diagnostics': diag, 'reps': REPS, 'estimates': {}}
    print(f'\n{"contrast":28}{"point":>9}{"home 95%":>22}')
    for key, a, b in (('report_replacement', 'F0', 'F1_rep'), ('premium_replacement', 'F1_rep', 'F2_rep'),
                      ('report_existing', 'F0', 'F1'), ('premium_existing', 'F1', 'F2')):
        pt = stat(np.ones(len(y)), a, b)
        draws = [stat(w, a, b) for w in W]
        ci = [round(float(np.percentile(draws, q)), 4) for q in (2.5, 97.5)]
        res['estimates'][key] = {'point': round(pt, 4), 'ci95_home': ci}
        print(f'{key:28}{pt:9.4f}{str(ci):>22}')

    OUT.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(f'\n-> {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
