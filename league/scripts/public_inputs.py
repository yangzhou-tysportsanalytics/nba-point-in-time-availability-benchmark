"""Load the rebuild's own outputs under the column names the analysis scripts use.

The released analyses were written against the originating project's tables. Everything they need is in this
repository's rebuild, under slightly different names: the player identifier is `pid` here and
`espn_player_id` there, and the reference-rotation table is written by `rebuild.py build` rather than shipped.
Rather than fork each script, they all load their inputs through here, so a reader can compare a released
script with the paper's description line by line.

What is shipped and what is not:

  data/bench_games_public.csv              shipped: game-level, no player rows
  data/status_table_public.csv             shipped: counts only
  data/out/bench_ref_players_public.csv    NOT shipped, produced by `rebuild.py build`

The last one is a player-level table and the source terms forbid redistributing one, so it is gitignored and
the reader makes it themselves. That is the only thing standing between this repository and rerunning every
player-level result in the paper, and it costs a rebuild rather than a request.

    python league/scripts/fetch_sources.py
    python league/scripts/rebuild.py parse 3600     # until it prints "done"
    python league/scripts/rebuild.py build          # writes both tables below
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# the same convention rebuild.py uses, including the override, so a reader who sets P2_PUBLIC does not have
# to set it twice
PUB = Path(os.environ.get('P2_PUBLIC', ROOT / 'data/public'))
OUT = PUB / 'out'
KAG = PUB / 'kaggle_v515'
PARTS = PUB / 'parse'
SHIPPED = ROOT / 'data'

REF = OUT / 'bench_ref_players_public.csv'
GAMES_REBUILT = OUT / 'bench_games_public.csv'
GAMES_SHIPPED = SHIPPED / 'bench_games_public.csv'
TEAM_GAMES = OUT / 'team_games_public.csv'

HOWTO = ('run `python league/scripts/rebuild.py build` first; it writes the player-level reference table, '
         'which is not shipped because the source terms forbid redistributing one')


def _need(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f'{path} is missing: {HOWTO}')
    return path


def ref_players() -> pd.DataFrame:
    """Reference-rotation player-games, named as the originating project names them."""
    r = pd.read_csv(_need(REF), dtype={'game_id': str, 'pid': str}, low_memory=False)
    return r.rename(columns={'pid': 'espn_player_id'})


def games(prefer_rebuilt: bool = True) -> pd.DataFrame:
    """Game-level benchmark. The rebuilt copy if present, else the shipped one, which is the same table."""
    p = GAMES_REBUILT if (prefer_rebuilt and GAMES_REBUILT.exists()) else _need(GAMES_SHIPPED)
    return pd.read_csv(p, dtype={'game_id': str})


def team_games() -> pd.DataFrame:
    """Per team-game cutoffs and the block-missing flag, for analyses that re-read status at other cutoffs."""
    t = pd.read_csv(_need(TEAM_GAMES), dtype={'game_id': str})
    return t.rename(columns={'cutoff': 'cutoff_utc'})


def report_rows() -> pd.DataFrame:
    """Every parsed report row, for the decomposition and the cutoff curve.

    One file per parsed batch under data/parse, as `rebuild.py parse` writes them.
    """
    parts = sorted(PARTS.glob('rows_*.csv'))
    if not parts:
        raise SystemExit(f'{PARTS} holds no rows_*.csv: run `rebuild.py parse` first')
    return pd.concat([pd.read_csv(p, low_memory=False) for p in parts], ignore_index=True)


def box_scores() -> pd.DataFrame:
    """Player box scores from the CC0 Kaggle source, including plus-minus.

    The plus-minus valuation of Section 4.2 needs `plusMinusPoints`, which this source carries, so that
    valuation is reproducible here and not only checkable from its result file.
    """
    p = _need(KAG / 'PlayerStatistics.csv')
    cols = ['personId', 'gameId', 'numMinutes', 'plusMinusPoints', 'points', 'reboundsTotal', 'assists',
            'steals', 'blocks', 'turnovers', 'fieldGoalsAttempted', 'fieldGoalsMade',
            'freeThrowsAttempted', 'freeThrowsMade']
    have = pd.read_csv(p, nrows=0).columns
    use = [c for c in cols if c in have]
    b = pd.read_csv(p, usecols=use, low_memory=False)
    return b.rename(columns={'personId': 'espn_player_id', 'gameId': 'game_id', 'numMinutes': 'min',
                             'plusMinusPoints': 'plus_minus'})


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    for name, fn in (('ref_players', ref_players), ('games', games), ('team_games', team_games)):
        try:
            d = fn()
            print(f'{name:14} {len(d):>8,} rows  {list(d.columns)[:6]}')
        except SystemExit as e:
            print(f'{name:14} unavailable: {e}')
