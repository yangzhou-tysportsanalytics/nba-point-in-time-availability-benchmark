"""The bootstrap designs the paper reads its intervals under, drawn in one place.

Each design is a bank of per-game weights: a replicate reweights the holdout games and every version's RMSE
is recomputed on those weights, so the statistic is a difference of two RMSEs on the same draw. The designs
differ only in what they hold fixed.

  home_season_cluster   resample home-team x season clusters; the design for single-season quantities
  season_then_cluster   resample the seasons, then the clusters inside each drawn season
  two_way_team          resample the 30 teams once and weight a game by the product of its two draws
  iid_game              resample games

The calibration (league/scripts/p6_ds_boot_calibration.py) scores these against a known truth on the real
schedule. season_then_cluster is the one that covers at its nominal rate for a quantity pooled over the four
holdout seasons; two_way_team is about 1.5 times too wide, and the dyadic estimator is about 14% too narrow
on this schedule. Single-season quantities carry no season to resample, so they are read under
home_season_cluster, which is what season_then_cluster degenerates to when there is one season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DESIGNS = ('home_season_cluster', 'season_then_cluster', 'two_way_team', 'iid_game')


def cluster_keys(H: pd.DataFrame) -> dict:
    """The index a weight bank needs: cluster codes, team indices and which clusters belong to each season."""
    hs_code, hs_keys = pd.factorize(pd.Series(H.home.astype(str) + '_' + H.season_year.astype(str)))
    teams = np.array(sorted(set(H.home) | set(H.away)))
    seasons = tuple(sorted(H.season_year.unique()))
    suffix = pd.Series(hs_keys)
    return {
        'n': len(H), 'hs_code': hs_code, 'n_hs': len(hs_keys), 'n_teams': len(teams), 'seasons': seasons,
        'hi': np.searchsorted(teams, H.home.values), 'ai': np.searchsorted(teams, H.away.values),
        'season_cluster_ids': {s: np.flatnonzero(suffix.str.endswith('_' + str(s)).values) for s in seasons},
    }


def weight_bank(rng, design: str, reps: int, keys: dict) -> np.ndarray:
    """reps x n matrix of per-game weights for one design."""
    n, hs, n_hs = keys['n'], keys['hs_code'], keys['n_hs']
    W = np.empty((reps, n))
    for r in range(reps):
        if design == 'home_season_cluster':
            c = np.bincount(rng.integers(0, n_hs, n_hs), minlength=n_hs).astype(float)
            W[r] = c[hs]
        elif design == 'season_then_cluster':
            c = np.zeros(n_hs)
            seasons = keys['seasons']
            for s in rng.choice(seasons, len(seasons)):
                ids = keys['season_cluster_ids'][s]
                c += np.bincount(rng.choice(ids, len(ids)), minlength=n_hs).astype(float)
            W[r] = c[hs]
        elif design == 'two_way_team':
            c = np.bincount(rng.integers(0, keys['n_teams'], keys['n_teams']),
                            minlength=keys['n_teams']).astype(float)
            W[r] = c[keys['hi']] * c[keys['ai']]
        elif design == 'iid_game':
            W[r] = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        else:
            raise ValueError(f'unknown design: {design}')
    return W
