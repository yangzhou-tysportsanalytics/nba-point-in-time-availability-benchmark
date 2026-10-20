"""Plan V item V3: separating a sample difference from a construction difference between two data paths.

The released rebuild path and the main build produce their own `bench_games` tables and run the same rolling
design on each, so when their premiums differ -- 0.074 (0.012-0.135) against 0.059 (0.000-0.118) -- two
explanations are available at once: the tables hold different games (4,835 against 4,846), and the features
inside them are built by different code from different inputs.

Restricting both to the identifiers they share removes the first explanation. What is left is the second, and
`feature_agreement` measures it directly rather than inferring it from the estimates: on the shared games,
how often is each feature the same number, and how far apart is it when it is not.

`restrict` returns rows in the order of the identifiers given, because a row-wise comparison of two tables
that happen to be sorted differently pairs the wrong games together and would report a construction
difference that is only an ordering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: two paths that compute the same number through different float arithmetic count as agreeing
TOL = 1e-9


def shared_ids(a: pd.DataFrame, b: pd.DataFrame, col: str = 'game_id') -> list:
    """The identifiers present in both tables, sorted, each once."""
    return sorted(set(a[col]) & set(b[col]))


def restrict(frame: pd.DataFrame, ids: list, col: str = 'game_id') -> pd.DataFrame:
    """The rows for `ids`, in the order of `ids`, with identifiers the frame lacks dropped."""
    d = frame.drop_duplicates(col).set_index(col)
    keep = [i for i in ids if i in d.index]
    return d.loc[keep].reset_index()


def feature_agreement(a: pd.DataFrame, b: pd.DataFrame, cols: list,
                      id_col: str = 'game_id', tol: float = TOL) -> dict:
    """How far two paths' features differ on the games they share, column by column."""
    ids = shared_ids(a, b, id_col)
    ra, rb = restrict(a, ids, id_col), restrict(b, ids, id_col)
    out = {'n_shared': len(ids), 'shared_ids': ids, 'tol': tol}
    for c in cols:
        d = np.asarray(rb[c], dtype=float) - np.asarray(ra[c], dtype=float)
        same = np.abs(d) <= tol
        out[c] = {
            'share_identical': round(float(same.mean()), 6) if len(d) else float('nan'),
            'n_different': int((~same).sum()),
            'max_abs_difference': round(float(np.abs(d).max()), 6) if len(d) else float('nan'),
            'mean_difference': round(float(d.mean()), 6) if len(d) else float('nan'),
            'mean_abs_difference': round(float(np.abs(d).mean()), 6) if len(d) else float('nan'),
            'sd_difference': round(float(d.std(ddof=1)), 6) if len(d) > 1 else float('nan'),
        }
        # the exact-match rate of a continuous quantity computed by two codebases is near zero whatever
        # they do, so the correlation is what says how close the two constructions are
        x, y = np.asarray(ra[c], dtype=float), np.asarray(rb[c], dtype=float)
        out[c]['correlation'] = (round(float(np.corrcoef(x, y)[0, 1]), 6)
                                 if len(x) > 1 and x.std() > 0 and y.std() > 0 else float('nan'))
        out[c]['sd_a'] = round(float(x.std(ddof=1)), 4) if len(x) > 1 else float('nan')
        out[c]['sd_b'] = round(float(y.std(ddof=1)), 4) if len(y) > 1 else float('nan')
    return out


#: the two paths spell six franchises differently; the released table uses the longer form
TEAM_ALIASES = {'GS': 'GSW', 'NO': 'NOP', 'NY': 'NYK', 'SA': 'SAS', 'UTAH': 'UTA', 'WSH': 'WAS'}


def normalise_team(code) -> str:
    """One spelling per franchise, so the two paths' rows can be matched at all."""
    c = str(code).strip().upper()
    return TEAM_ALIASES.get(c, c)


def game_key(frame: pd.DataFrame, dates: dict | None = None,
             date_col: str = 'game_date_et') -> pd.Series:
    """A key the two paths share: season, date, and the two teams in one spelling.

    The released table carries NBA game ids and ours carries ESPN event ids, so the identifier columns have
    no values in common and a join on `game_id` silently matches nothing. `dates` maps a frame's own
    `game_id` to a date string, for the table that has no date column of its own.
    """
    raw = frame['game_id'].map(dates) if dates is not None else frame[date_col]
    # the two paths write the date differently -- ours ISO, the released table MM/DD/YYYY -- and both are
    # parsed rather than sliced, because a ten-character slice keeps "10/19/2021" intact and matches nothing
    d = (pd.to_datetime(pd.Series(raw, index=frame.index).astype(str), format='mixed', utc=True)
           .dt.strftime('%Y-%m-%d'))
    return (frame.season_year.astype(str) + '|' + d + '|'
            + frame.home.map(normalise_team) + '|' + frame.away.map(normalise_team))
