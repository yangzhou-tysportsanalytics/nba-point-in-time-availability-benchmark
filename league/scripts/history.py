"""History windows, copied unchanged from the originating project so the two cannot drift.

The valuations released here need a player mean over his last games that had finished before a
team-game cutoff, and the winsorising bounds that use only earlier seasons. Both rules are part of
what the paper claims, so they are shipped rather than reimplemented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GAME_HOURS = 3.0   # a game is treated as finished three hours after tip-off
CUTOFF_HOURS = 6.0  # the paper's cutoff


def last_games_before_cutoff(box: pd.DataFrame, ref: pd.DataFrame, games: pd.DataFrame, value: str,
                             window: int = 20, min_periods: int = 1,
                             cutoff_hours=CUTOFF_HOURS, max_age_days: float | None = None) -> pd.Series:
    """Mean of `value` over a player's last `window` games finished before each reference row's cutoff.

    `cutoff_hours` is a number, or a Series aligned to `ref` when the cutoff differs per row (the build's
    90-minute fallback). `max_age_days` drops games older than that many days before the same cutoff, which is
    the main build's 365-day history window; None keeps every earlier game.
    """
    b = box[['game_id', 'espn_player_id', value]].dropna(subset=['espn_player_id', value]).merge(
        games[['game_id', 'start']], on='game_id', how='left')
    b = b.dropna(subset=['start'])
    b['finished'] = b['start'] + pd.Timedelta(hours=GAME_HOURS)
    b = b.sort_values(['espn_player_id', 'finished'], kind='mergesort').reset_index(drop=True)
    pid = b.espn_player_id.to_numpy()
    fin = b.finished.to_numpy().astype('datetime64[ns]').astype('int64')
    vals = b[value].to_numpy(dtype=float)
    csum = np.concatenate([[0.0], np.cumsum(vals)])

    left = ref[['game_id', 'espn_player_id']].reset_index(drop=True).merge(
        games[['game_id', 'start']], on='game_id', how='left')
    hours = pd.Series(cutoff_hours, index=left.index) if np.isscalar(cutoff_hours) \
        else pd.Series(np.asarray(cutoff_hours, dtype=float), index=left.index)
    cut = (left['start'] - pd.to_timedelta(hours, unit='h')).to_numpy().astype('datetime64[ns]').astype('int64')
    age = None if max_age_days is None else int(max_age_days * 24 * 3600 * 1_000_000_000)
    game_ns = int(GAME_HOURS * 3600 * 1_000_000_000)

    # `pid` is sorted, so each player's rows are one slice; within it the games are in time order
    lo_all = np.searchsorted(pid, left.espn_player_id.to_numpy(), side='left')
    hi_all = np.searchsorted(pid, left.espn_player_id.to_numpy(), side='right')
    out = np.full(len(left), np.nan)
    for r, (lo, hi, c) in enumerate(zip(lo_all, hi_all, cut)):
        if hi <= lo or c != c:
            continue
        i = int(np.searchsorted(fin[lo:hi], c, side='left')) - 1   # last game finished strictly before the cutoff
        if i < 0:
            continue
        # the build's floor is on the history game's start, not its end (league/src/build.py, HIST_DAYS)
        j = 0 if age is None else int(np.searchsorted(fin[lo:hi], c - age + game_ns, side='left'))
        start = max(j, i - window + 1)
        n = i - start + 1
        if n < min_periods:
            continue
        out[r] = (csum[lo + i + 1] - csum[lo + start]) / n
    return pd.Series(out, index=ref.index)


def season_prior_bounds(frame: pd.DataFrame, value: str, season_col: str = 'season_year',
                        lo: float = 0.01, hi: float = 0.99) -> dict:
    """Winsorising bounds for each season, computed only from earlier seasons (the earliest keeps its own)."""
    seasons = sorted(frame[season_col].dropna().unique())
    bounds = {}
    for s in seasons:
        prior = frame[frame[season_col] < s][value].dropna()
        if len(prior) < 1000:
            prior = frame[frame[season_col] == s][value].dropna()
        bounds[s] = (float(prior.quantile(lo)), float(prior.quantile(hi)))
    return bounds


if __name__ == '__main__':  # a check that the lookup is defined for absent players, which is the whole point
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    S = dict(dtype={'game_id': str, 'espn_player_id': str}, low_memory=False)
    EXT, DATA = ROOT / 'data/league/paper2_ext', ROOT / 'data/league/paper2_ext/rebuild/data'
    ref = pd.read_csv(EXT / 'bench_ref_players.csv', **S)
    box = pd.read_csv(DATA / 'player_games.csv', **S)
    box['min'] = pd.to_numeric(box['min'], errors='coerce')
    games = pd.read_csv(DATA / 'games.csv', dtype={'game_id': str})
    games['start'] = pd.to_datetime(games.start_utc, utc=True)
    h = last_games_before_cutoff(box[box['min'].fillna(0) > 0], ref, games, 'min')
    for label, mask in (('actually absent', ref.out_real.astype(bool)), ('Out at the cutoff', ref.out_pre.astype(bool)),
                        ('all reference rows', pd.Series(True, index=ref.index))):
        sub = h[mask.values]
        print(f'{label:22s} rows {len(sub):7,}  with history {sub.notna().sum():7,} '
              f'({100 * sub.notna().mean():5.1f}%)  mean {sub.mean():.2f}')
