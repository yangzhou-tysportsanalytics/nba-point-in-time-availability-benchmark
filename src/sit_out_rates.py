"""Plan BB: the rolling per-status sit-out rate F7 weights a report with.

F7 charges a listed player the share of his absence value that his status has historically cost: among
reference-rotation player-games carrying that status whose game had already finished, the share in which the
player did not play. The rate is rolling because the feature has to be buildable at the cutoff --- a rate
estimated on the whole sample would price a report with games that had not been played yet.

`p6_ds_cutoff_f7.py` computes this inline and its result is published, so this module is not a replacement
for that run; it is the same construction written out separately so that the public-path port can be checked
against the definition rather than against the main build's number, which is what plan BB's control BB-b
does. The cases it has to get right are in `tests/test_sit_out_rate_rules.py`: a game that ends exactly on
the cutoff counts, a status with no history yet is the prior and not a zero, the rows need not arrive in
order, a status the caller did not ask for does not leak into another, and an end time the build does not
know is an error rather than a dropped row.

The prior is Beta(1,1), the main build's: one sit-out and one appearance, so an unseen status is a half.
"""
from __future__ import annotations

import numpy as np

PRIOR_A, PRIOR_N = 1.0, 2.0


def rolling_rates(status, out_real, end, statuses, cutoffs, prior_a=PRIOR_A, prior_n=PRIOR_N):
    """Per cutoff and per status, the sit-out rate among that status's finished player-games.

    `status`, `out_real` and `end` are the three columns of the reference-rotation table: the status the
    player carried, whether he then did not play, and when his game ended. `statuses` fixes the columns of
    the result and `cutoffs` its rows, so the caller gets a rate for every status at every cutoff and not
    only for the status a player happened to carry --- a later report can move him to any of them.

    Returns an array of shape (len(cutoffs), len(statuses)). Selection is at or before the cutoff.
    """
    status = np.asarray(status, dtype=object)
    out_real = np.asarray(out_real, dtype=bool)
    end = np.asarray(end)
    cutoffs = np.asarray(cutoffs)
    if not (len(status) == len(out_real) == len(end)):
        raise ValueError(f'the three columns disagree: {len(status)}, {len(out_real)}, {len(end)}')
    if end.dtype.kind == 'f' and np.isnan(end).any():
        raise ValueError('a player-game with no end time cannot be placed before or after a cutoff; the '
                         'build knows when every game it keeps finished, so this is a join that failed '
                         'rather than a row to drop')
    if end.dtype.kind == 'M' and np.isnat(end).any():
        raise ValueError('a player-game with no end time cannot be placed before or after a cutoff')

    out = np.empty((len(cutoffs), len(statuses)))
    for j, s in enumerate(statuses):
        pick = status == s
        e, o = end[pick], out_real[pick]
        order = np.argsort(e, kind='stable')
        e, o = e[order], o[order]
        cum = np.concatenate([[0.0], np.cumsum(o, dtype=float)])
        k = np.searchsorted(e, cutoffs, side='right')
        out[:, j] = (cum[k] + prior_a) / (k + prior_n)
    return out
