"""Plan V item V4: the roster check the paper does not have.

Section 2.2 reports that the roster recalls over 99.9% of the players who actually appeared. That is the
right check for a player who is missing and it cannot detect the opposite error, a player kept on a team's
candidate list after he has left it. Such a player can carry reference minutes, count as absent in the
realized-participation feature and land in the never-listed component, without ever being on an Out list --
which is the quantity this paper is about.

The removal rule, read out of the build's own code rather than from its description: a player is on a team's
list at a cutoff when the last event at or before that cutoff puts him there. Events are box-score
appearances for a team, `join` events from the league's movement log, and `leave` events, which the build
emits for one transaction type only. A player therefore leaves a list when he is waived, when he appears or
joins elsewhere, or when 365 days pass with no event at all.

A false active cannot be observed directly, so this module measures a proxy that can: a reference-rotation
player who never appears for that team again in the rest of the season. That is not proof of an error -- a
player can miss the remainder of a season through injury -- so the quantity that matters is the comparison
between team-games near a transaction and team-games away from one, where only the first should be inflated
if the removal rule is leaking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def last_appearance(box: pd.DataFrame, pid_col: str = 'espn_player_id',
                    team_col: str = 'team', time_col: str = 'start',
                    season_col: str = 'season_year') -> dict:
    """{(season, player, team): last time he appeared for that team that season}.

    Appearance means a box-score row, played or not, because that is what the build counts as an event.
    """
    d = box[[season_col, pid_col, team_col, time_col]].dropna(subset=[pid_col])
    g = d.groupby([season_col, pid_col, team_col])[time_col].max()
    return g.to_dict()


def never_appears_again(rows: pd.DataFrame, last: dict, time_col: str = 'start') -> np.ndarray:
    """One flag per row: this player never appears for this team again after this game.

    A player with no appearance for the team at all in the season counts as never appearing again, because
    the candidate list put him there on the strength of an earlier season or a movement event.
    """
    out = np.empty(len(rows), dtype=bool)
    for i, (s, p, t, when) in enumerate(zip(rows.season_year, rows.espn_player_id, rows.team,
                                            rows[time_col])):
        lastseen = last.get((s, p, t))
        out[i] = True if lastseen is None else not (lastseen > when)
    return out


def false_active_rate(flags: np.ndarray, near: np.ndarray) -> dict:
    """The proxy's rate near a transaction and away from one, and the ratio rule V-e is written on."""
    flags, near = np.asarray(flags, dtype=bool), np.asarray(near, dtype=bool)
    n_near, n_far = int(near.sum()), int((~near).sum())
    r_near = float(flags[near].mean()) if n_near else float('nan')
    r_far = float(flags[~near].mean()) if n_far else float('nan')
    ratio = (r_near / r_far) if r_far else float('nan')
    return {'n_near_transaction': n_near, 'n_away_from_transaction': n_far,
            'rate_near': round(r_near, 6) if n_near else float('nan'),
            'rate_away': round(r_far, 6) if n_far else float('nan'),
            'ratio': round(ratio, 4) if np.isfinite(ratio) else float('nan')}
