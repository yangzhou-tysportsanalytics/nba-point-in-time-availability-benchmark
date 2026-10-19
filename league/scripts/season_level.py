"""Season-level readings: the across-season mean, and a prediction interval for one unseen season.

Plan T (docs/plans/paper2_T_prediction_interval.md). The paper reported
`mean +- t(n-1) * se(mean)` under the claim that its target was "a season we have not seen, not an average
over these four". That interval is the uncertainty of the mean. Predicting one new season carries that
season's own deviation from the mean as well, so the standard error is sqrt(n+1) times larger at the same
confidence --- a factor of 2.24 with four seasons.

Two predictive targets exist and the paper says which it reports:

  the estimate a new season would produce   var = (tau^2 + v) * (1 + 1/n)
  that season's true value                  var = tau^2 + (tau^2 + v) / n

The first is what the registered 2026--27 test will observe, so it is the one printed; the second is returned
beside it so the two are not confused. `v` is the mean within-season sampling variance when the caller can
supply it, and zero when it cannot, in which case the between-season spread still contains it and both
intervals are conservative.
"""
from __future__ import annotations

import math
from statistics import fmean

# the two-sided 97.5th percentile of t with n-1 degrees of freedom, for the n this project ever has
T_975 = {2: 12.706205, 3: 4.302653, 4: 3.182446, 5: 2.776445, 6: 2.570582, 7: 2.446912}


def season_level(per_season, within_var: float | None = None) -> dict:
    """The two readings from one estimate per season.

    `per_season` is one number per season. `within_var` is the mean sampling variance of those numbers inside
    each season, when it is known; the between-season spread then has it removed, which narrows both
    intervals and is the honest version where the input allows it.
    """
    xs = [float(x) for x in per_season]
    n = len(xs)
    out: dict = {'n_seasons': n, 'mean_of_seasons': round(fmean(xs), 6) if n else None,
                 'per_season': [round(x, 6) for x in xs],
                 'variance_model': 'between minus within' if within_var is not None
                                   else 'plain between-season spread'}
    if n < 2:
        # one season carries no between-season spread; a zero-width interval would read as perfect precision
        out.update(season_sd=None, ci95_t3=None, pi95_t3=None, pi95_t3_true_value=None,
                   season_variance_tau2=None)
        return out

    mean = fmean(xs)
    between = sum((x - mean) ** 2 for x in xs) / (n - 1)
    v = float(within_var) if within_var is not None else 0.0
    tau2 = max(between - v, 0.0)          # the season-to-season variance of the underlying value
    t = T_975.get(n)
    if t is None:
        raise ValueError(f'no t quantile recorded for {n} seasons')

    # The spread of the observed season estimates is what the mean of them, and a new one of them, vary
    # over -- so both use `between` directly. Substituting tau2 + v for it looks equivalent, and is, until
    # tau2 clamps at zero: then tau2 + v is v, which is larger than `between` precisely when there is least
    # evidence of real season-to-season variation. That inflated the report's value's published interval by
    # a factor of two from 2026-10-04 until round 19 found it. Only the reading for a new season's own
    # underlying value needs the decomposition, and there the clamp narrows, as it should.
    se_mean = math.sqrt(between / n)
    se_pred = math.sqrt(between * (1 + 1 / n))
    se_pred_true = math.sqrt(tau2 + between / n)
    out.update(
        season_sd=round(math.sqrt(between), 6),
        season_variance_tau2=round(tau2, 9),
        mean_within_season_var=round(v, 9),
        se_of_mean=round(se_mean, 6),
        ci95_t3=[round(mean - t * se_mean, 6), round(mean + t * se_mean, 6)],
        pi95_t3=[round(mean - t * se_pred, 6), round(mean + t * se_pred, 6)],
        pi95_t3_true_value=[round(mean - t * se_pred_true, 6), round(mean + t * se_pred_true, 6)],
        targets={'ci95_t3': 'the average over seasons drawn from this process',
                 'pi95_t3': 'the estimate one unseen season would produce',
                 'pi95_t3_true_value': "that season's own underlying value"},
    )
    return out
