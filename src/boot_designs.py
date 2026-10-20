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

DESIGNS = ('home_season_cluster', 'away_season_cluster', 'season_then_cluster', 'two_way_team',
           'iid_game')


def cluster_keys(H: pd.DataFrame) -> dict:
    """The index a weight bank needs: cluster codes, team indices and which clusters belong to each season."""
    hs_code, hs_keys = pd.factorize(pd.Series(H.home.astype(str) + '_' + H.season_year.astype(str)))
    as_code, as_keys = pd.factorize(pd.Series(H.away.astype(str) + '_' + H.season_year.astype(str)))
    teams = np.array(sorted(set(H.home) | set(H.away)))
    seasons = tuple(sorted(H.season_year.unique()))
    suffix = pd.Series(hs_keys)
    return {
        'n': len(H), 'hs_code': hs_code, 'n_hs': len(hs_keys), 'n_teams': len(teams), 'seasons': seasons,
        'as_code': as_code, 'n_as': len(as_keys),
        'hi': np.searchsorted(teams, H.home.values), 'ai': np.searchsorted(teams, H.away.values),
        'season_cluster_ids': {s: np.flatnonzero(suffix.str.endswith('_' + str(s)).values) for s in seasons},
    }


def _pair_code(keys: dict) -> np.ndarray:
    """An id per unordered matchup, so a pair of games sharing *both* teams can be counted once."""
    hi, ai = keys['hi'], keys['ai']
    return pd.factorize(pd.Series(np.minimum(hi, ai) * (keys['n_teams'] + 1) + np.maximum(hi, ai)))[0]


def dyadic_variance(psi: np.ndarray, keys: dict) -> float:
    """The dyadic cluster-robust variance of a mean: (1/n^2) times the double sum over game pairs that
    share at least one team, the diagonal included.

    Summing psi over each team's games and squaring counts a pair that shares *both* teams twice, so those
    are subtracted once, grouped by unordered matchup.
    """
    S = np.zeros(keys['n_teams'])
    np.add.at(S, keys['hi'], psi)
    np.add.at(S, keys['ai'], psi)
    pc = _pair_code(keys)
    T = np.zeros((pc.max() + 1) if len(pc) else 1)
    np.add.at(T, pc, psi)
    return (float((S ** 2).sum()) - float((T ** 2).sum())) / keys['n'] ** 2


def dyadic_deflation(keys: dict) -> dict:
    """How much centring on the sample mean deflates `dyadic_variance` when the games are independent.

    With psi = v - vbar and v independent of variance s2, E[psi_i^2] = s2 (1 - 1/n) and E[psi_i psi_j] =
    -s2/n for i != j, so the double sum over the diagonal plus the P **ordered** pairs of distinct games
    that share a team has expectation s2 (n - 1 - P/n). Against the independent target n*s2 the ratio is

        1 - 1/n - P/n^2.

    Earlier revisions of this project used `1 - 2P/n^2` with the same ordered P, which applies the factor of
    two a second time: on the paper's schedule that reported 0.859 on the standard error where the correct
    value is 0.932, and a direct simulation gives 0.9313. The sign of the error matters for the conclusion
    it was used to support, because the placebo measures coverage of 0.860-0.873: centring explains *less*
    of that shortfall than the doubled factor implied, and the rest is real dependence.

    The exact correction under real dependence is not this factor, so it stays a diagnostic.
    """
    n, n_teams = keys['n'], keys['n_teams']
    cnt = np.zeros(n_teams)
    np.add.at(cnt, keys['hi'], 1)
    np.add.at(cnt, keys['ai'], 1)
    pc = _pair_code(keys)
    # sum_t cnt_t^2 counts every ordered pair of games sharing t, the diagonal included, and counts a pair
    # sharing both teams twice; sum over unordered matchups removes the diagonal and that excess.
    ordered_with_diagonal = float((cnt ** 2).sum() - (np.bincount(pc) ** 2).sum())
    P = ordered_with_diagonal - n          # ordered pairs of *distinct* games sharing at least one team
    v_factor = 1.0 - 1.0 / n - P / n ** 2
    # unrounded: the caller rounds for display, so a test can compare this with a simulation
    return {'ordered_pairs_sharing_a_team': int(round(P)), 'per_game': round(P / n, 1),
            'variance_factor': float(v_factor),
            'se_factor': float(np.sqrt(max(v_factor, 0.0))),
            'note': 'Centring on the sample mean makes the dyadic estimator low by about this factor when '
                    'games are independent. Compare it with the dyadic row of the placebo coverage: the '
                    'measured shortfall is larger than this, so centring is part of the story and real '
                    'dependence is the rest.'}


def weight_bank(rng, design: str, reps: int, keys: dict) -> np.ndarray:
    """reps x n matrix of per-game weights for one design."""
    n, hs, n_hs = keys['n'], keys['hs_code'], keys['n_hs']
    W = np.empty((reps, n))
    for r in range(reps):
        if design == 'home_season_cluster':
            c = np.bincount(rng.integers(0, n_hs, n_hs), minlength=n_hs).astype(float)
            W[r] = c[hs]
        elif design == 'away_season_cluster':
            # the mirror of the design above, built from the same keys by the same two lines, so that a
            # difference between the two readings can only come from the data (plan W)
            n_as, aw = keys['n_as'], keys['as_code']
            c = np.bincount(rng.integers(0, n_as, n_as), minlength=n_as).astype(float)
            W[r] = c[aw]
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
