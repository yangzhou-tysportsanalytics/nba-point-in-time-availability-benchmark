"""Plan Z: comparing two build paths on one sample, with only the availability build swapped.

The two tables carry different identifiers and share none, so they are joined on season, date and the two
teams, and the comparison is only as good as that join. Two things here can be wrong while every number they
produce still looks plausible, so both are separate from the script that runs them and are tested against
`tests/test_paired_build_rules.py`:

  * `aligned` returns index vectors into each side that point at the same games in the same order. A
    duplicate key on either side would double-count a game silently, and a key present on one side only
    must leave both arms rather than be filled; both are errors here.
  * `paired_contrast` differences the two arms' premiums **under one draw**. Resampling each arm separately
    and differencing the two results discards the pairing the whole experiment is for, and widens the
    interval by the variance the pairing removes.
"""
from __future__ import annotations

import numpy as np


def aligned(keys_a, keys_b):
    """Index vectors into each side, pointing at the games both hold, in one deterministic order.

    Returns (ia, ib, keys). `keys` is sorted, so the order does not depend on which side is passed first or
    on the order either table happens to be in.
    """
    la, lb = list(keys_a), list(keys_b)
    for side, ks in (('first', la), ('second', lb)):
        if len(set(ks)) != len(ks):
            dup = next(k for k in ks if ks.count(k) > 1)
            raise ValueError(f'duplicate key in the {side} table: {dup!r}; a join on it would double-count')
    pos_a = {k: i for i, k in enumerate(la)}
    pos_b = {k: i for i, k in enumerate(lb)}
    keys = sorted(set(pos_a) & set(pos_b))
    if not keys:
        raise ValueError('the two tables share no game: the join keys do not match')
    return [pos_a[k] for k in keys], [pos_b[k] for k in keys], keys


def paired_contrast(sq, w, arm_a, arm_b):
    """(premium of arm A) - (premium of arm B) under the weighting `w`, which both arms share.

    `sq` maps a model name to its per-game squared losses; an arm is the pair (compliant, leaky) whose RMSE
    difference is that arm's premium.
    """
    w = np.asarray(w, dtype=float)
    for name in (*arm_a, *arm_b):
        if name not in sq:
            raise KeyError(f'no squared losses for {name!r}')
        if len(sq[name]) != len(w):
            raise ValueError(f'{name!r} has {len(sq[name])} games against {len(w)} weights')
    tot = w.sum()
    rmse = lambda k: float(np.sqrt((w * np.asarray(sq[k], dtype=float)).sum() / tot))
    return (rmse(arm_a[0]) - rmse(arm_a[1])) - (rmse(arm_b[0]) - rmse(arm_b[1]))
