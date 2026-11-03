"""Tests of equal predictive accuracy, and the variances they are read under.

Section 3.1 of the paper cites Diebold and Mariano (1995), Giacomini and White (2006) and Clark and West
(2007) and then uses a cluster bootstrap instead. These are the tests themselves. They matter most for the
two comparisons the paper calls nested, where the larger model pays for its extra parameter out of sample
and a plain DM test is therefore biased against it.

Sign convention throughout: a positive differential means the larger (or second) model predicts better.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def diebold_mariano(y, p_small, p_large):
    """The per-observation squared-error differential, loss(small) - loss(large)."""
    y, p_small, p_large = np.asarray(y, float), np.asarray(p_small, float), np.asarray(p_large, float)
    return (y - p_small) ** 2 - (y - p_large) ** 2


def clark_west(y, p_small, p_large):
    """Clark and West's adjusted differential, for a pair where the small model is nested in the large one.

    The adjustment adds back the squared difference of the two forecasts, which is what the larger model's
    extra estimated parameter costs it out of sample under the null of equal population accuracy.
    """
    p_small, p_large = np.asarray(p_small, float), np.asarray(p_large, float)
    return diebold_mariano(y, p_small, p_large) + (p_small - p_large) ** 2


def dyadic_variance(psi, home, away):
    """Dyadic cluster-robust variance of a mean (Aronow, Samii and Assenova 2015).

    A game belongs to two clusters at once, so the clusters overlap rather than partition the sample. The
    variance is the double sum of psi_i psi_j over games sharing at least one team, divided by n squared.
    Summing per team counts a pair that shares both teams twice, so the per-matchup sum is subtracted once.
    """
    psi = np.asarray(psi, float)
    home, away = np.asarray(home), np.asarray(away)
    n = len(psi)
    teams = np.unique(np.concatenate([home, away]))
    hi, ai = np.searchsorted(teams, home), np.searchsorted(teams, away)
    pair, _ = pd.factorize(pd.Series(np.minimum(hi, ai) * (len(teams) + 1) + np.maximum(hi, ai)))
    S = np.zeros(len(teams)); np.add.at(S, hi, psi); np.add.at(S, ai, psi)
    T = np.zeros(pair.max() + 1); np.add.at(T, pair, psi)
    return (float((S ** 2).sum()) - float((T ** 2).sum())) / n ** 2


def mean_se(v, kind='iid', home=None, away=None, cluster=None):
    """Standard error of the mean of v, under one of three readings of the dependence.

    'iid' treats games as independent, 'dyadic' as belonging to both teams' clusters, and 'cluster' as
    partitioned by the labels given (home team x season, in the paper).
    """
    v = np.asarray(v, float)
    psi = v - v.mean()
    n = len(v)
    if kind == 'iid':
        return float(np.sqrt((psi ** 2).sum()) / n)
    if kind == 'dyadic':
        return float(np.sqrt(max(dyadic_variance(psi, home, away), 0.0)))
    if kind == 'cluster':
        code, _ = pd.factorize(pd.Series(np.asarray(cluster)))
        C = np.zeros(code.max() + 1); np.add.at(C, code, psi)
        return float(np.sqrt(float((C ** 2).sum())) / n)
    raise ValueError(f'unknown variance: {kind}')
