"""Plan V item V1: the two timestamp conventions, and a comparison that holds the sample fixed.

A report has two times. The one printed in its header is what the league says it published, and the one the
server returns in `Last-Modified` is when the file it served was last written. The paper's point-in-time rule
uses the header time; the stricter rule uses the later of the two, on the grounds that a reader could not
have had the file before the server had it.

Section 4.6's published run evaluated the strict feature on the games the *header* convention can serve. For
the team-games whose qualifying report the strict rule takes away, the team then contributed no absentees,
which reads a missing report as a healthy roster. So the movement it reported mixes two changes: a different
report selected, and health imputed where no report exists. `common_sample` exists to remove the second, and
it drops such a game rather than returning a mask that a caller could fill with zeros.

Nothing here fits a model. The functions are the sample rule, the selection rule, the disagreement count and
the paired contrast; the script that uses them is league/scripts/p6_ds_server_time.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONVENTIONS = ('header', 'server')


def server_time_of(reports: pd.DataFrame, ts: str = 'ts', lm: str = 'lm') -> pd.Series:
    """The later of the printed header time and the server's Last-Modified time, per row.

    A file with no recorded server time keeps its header time: the rule is "no earlier than the server had
    it", and an unrecorded server time is not evidence that the file was served later.
    """
    return reports[[ts, lm]].max(axis=1, skipna=True)


def selected_report(reports: pd.DataFrame, hours: float, convention: str = 'header',
                    ts: str = 'ts', lm: str = 'lm') -> dict:
    """{(game_id, team): timestamp} of the latest report available at `hours` before tip-off.

    A team-game with no report at or before the cutoff is absent from the result. It is not present with a
    null, because the caller must not be able to confuse "no report" with "a report listing nobody".
    """
    if convention not in CONVENTIONS:
        raise ValueError(f'unknown convention: {convention}')
    d = reports.copy()
    d['effective'] = server_time_of(d, ts, lm) if convention == 'server' else d[ts]
    cutoff = d['start'] - pd.Timedelta(hours=hours)
    d = d[d['effective'] <= cutoff]
    if not len(d):
        return {}
    # the report a reader would hold is the latest one available, identified by its own header time so that
    # the two conventions can be compared on which report they select
    idx = d.groupby(['game_id', 'team'])['effective'].idxmax()
    picked = d.loc[idx]
    return dict(zip(zip(picked.game_id, picked.team), picked[ts]))


def common_sample(games: pd.DataFrame, have_header: set, have_server: set) -> np.ndarray:
    """One flag per game: both teams have a qualifying report under **both** conventions.

    This is the whole point of item V1. A game in which either team is served by only one of the two
    conventions is dropped, because keeping it would require giving that team a feature value under the
    convention that cannot serve it, and the only available value is a zero that means "healthy".
    """
    both = lambda g, t: ((g, t) in have_header) and ((g, t) in have_server)
    return np.array([both(g, h) and both(g, a)
                     for g, h, a in zip(games.game_id, games.home, games.away)], dtype=bool)


def disagreement(sel_header: dict, sel_server: dict) -> dict:
    """How often the two conventions select a different report, over the team-games both can serve."""
    shared = set(sel_header) & set(sel_server)
    diff = sum(1 for k in shared if sel_header[k] != sel_server[k])
    one_only = len(set(sel_header) ^ set(sel_server))
    return {'team_games_compared': len(shared),
            'different_report': int(diff),
            'share_different': round(diff / len(shared), 4) if shared else float('nan'),
            'served_by_one_only': int(one_only),
            'served_by_header_only': int(len(set(sel_header) - set(sel_server))),
            'served_by_server_only': int(len(set(sel_server) - set(sel_header)))}


def paired_contrast(sq_a0: np.ndarray, sq_a1: np.ndarray,
                    sq_b0: np.ndarray, sq_b1: np.ndarray, W: np.ndarray) -> dict:
    """(RMSE(a0) - RMSE(a1)) - (RMSE(b0) - RMSE(b1)), with every draw reweighting the same games.

    The two conventions are compared inside each bootstrap draw. Taking the difference between two
    separately drawn intervals would carry the variance of both and would not be a contrast at all: for
    identical inputs this returns an interval of zero width, which is the test that pins it.
    """
    rmse = lambda w, sq: np.sqrt(np.sum(w * sq) / w.sum())
    point = ((np.sqrt(sq_a0.mean()) - np.sqrt(sq_a1.mean()))
             - (np.sqrt(sq_b0.mean()) - np.sqrt(sq_b1.mean())))
    draws = np.array([(rmse(w, sq_a0) - rmse(w, sq_a1)) - (rmse(w, sq_b0) - rmse(w, sq_b1)) for w in W])
    return {'point': float(point),
            'ci95': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            'draws': draws}
