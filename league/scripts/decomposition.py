"""The decomposition of Section 4.6 and its put-backs, on this repository's own rebuild.

The difference between the compliant and the leaky feature decomposes exactly into three kinds of absence,
each defined on players who did not play and none of them available at the cutoff:

  declared   absent, not counted by the compliant feature, and Out on the last report held at thirty minutes
  late       absent, not Out at thirty minutes, but listed on that report
  never      absent and not on that report at all

Each is disjoint from the compliant feature's own Out set, which matters: an earlier version of this analysis
in the originating project counted a player twice when his status was withdrawn on a later report and he then
sat, and the attribution that rested on it was withdrawn.

Two readings are produced for each component. A put-back adds it into the feature at the valuation's own
weight, which is a substitution contrast and whose sign is not pinned. Entering it as its own regressor lets
the data choose the weight, which makes the comparison nested, so its population value is non-negative and
Clark and West's adjustment is read beside the plain statistic. The paper reports both and they say
different things.

Two versions of "all three" exist and this script keeps them apart, which an earlier version of it did
not. The paper's is F1 plus the three components, which differs from the leaky feature by the
player-games the compliant feature calls Out who then played -- they are in F1 and not in F2 -- so its
put-back is close to the leaky feature's and not equal to it. The identity needs that fourth term
subtracted as well, and with it the feature **is** the leaky one, so its put-back equals the
F1-against-F2 row by construction and is not a second result. Reporting only the second, as this script
did, made the release disagree with the paper on a number that looks like the same one.

    python league/scripts/rebuild.py build    # writes the player-level table this needs
    python league/scripts/decomposition.py    # -> results/paper2_decomposition_public.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'league/scripts'))
import public_inputs as pi                                                   # noqa: E402
import rebuild as rb                                                         # noqa: E402

OUT = ROOT / 'results/paper2_decomposition_public.json'
SEASONS = (2023, 2024, 2025, 2026)
BASE = ['strength_diff', 'rest_diff']
SEED, REPS = 20260921, 10000


def main() -> int:
    g, box, _ = rb.load_games_box()
    rows, blocks, _ = rb.load_reports(g)
    R = pi.ref_players()
    R['val'] = R.base_minutes * R.eff_prior
    G = pi.games()

    best30, st30 = rb.status_at(g, rows, blocks, box, pd.Timedelta(minutes=30))
    key = lambda d: set(zip(d.game_id.astype(str), d.team, d.pid.astype(str)))
    out30 = key(st30[(st30.status == 'Out') & st30.pid.notna()])
    listed30 = key(st30[st30.pid.notna()])

    k = list(zip(R.game_id.astype(str), R.team, R.espn_player_id.astype(str)))
    sat = {t for t, a in zip(k, R.out_real.astype(bool)) if a}
    f1_out = {t for t, a in zip(k, R.out_pre.astype(bool)) if a}

    # each component is disjoint from what the compliant feature already counts
    declared = {t for t in sat if t not in f1_out and t in out30}
    late = {t for t in sat if t not in out30 and t in listed30 and t not in f1_out}
    never = {t for t in sat if t not in out30 and t not in listed30 and t not in f1_out}
    counts = {'absent': len(sat), 'declared': len(declared), 'late': len(late), 'never': len(never),
              'out_at_cutoff_but_played': len(f1_out - sat)}
    print('components:', json.dumps(counts))

    def diff_for(members):
        v = pd.Series(np.where([t in members for t in k], R.val, 0.0)).groupby([R.game_id, R.team]).sum()
        return [v.get((gm, h), 0.0) - v.get((gm, a), 0.0) for gm, h, a in zip(G.game_id, G.home, G.away)]

    G['c1_diff'] = diff_for(declared)
    G['c2_diff'] = diff_for(late)
    G['c3_diff'] = diff_for(never)
    # the identity also subtracts the players the compliant feature counts as Out who then played: they are in
    # F1 and not in F2, so leaving the term out leaves a residual rather than an identity
    G['played_diff'] = diff_for(f1_out - sat)
    for n, c in (('F1pc1', ['c1_diff']), ('F1pc2', ['c2_diff']), ('F1pc3', ['c3_diff']),
                 ('F1pc123', ['c1_diff', 'c2_diff', 'c3_diff'])):
        G[n + '_diff'] = G.F1_diff + sum(G[x] for x in c)
    # the paper's "all three" is the line above: F1 and the three components, differing from the leaky
    # feature by the player-games F1 calls Out who then played. Subtracting those as well closes the
    # identity and produces the leaky feature itself, which is a check and not a put-back
    G['F1pc123id_diff'] = G.F1pc123_diff - G.played_diff

    H0 = G[~G.any_block_missing].reset_index(drop=True)
    resid = (H0.F2_diff - H0.F1pc123id_diff).abs().max()
    gap = (H0.F2_diff - H0.F1pc123_diff).abs().max()
    print(f'identity residual with the fourth term, largest over games: {resid:.6f}')
    print(f'the paper\'s all three against the leaky feature, largest gap: {gap:.6f}')

    VERS = {'F0': [], 'F1': ['F1_diff'], 'F2': ['F2_diff'],
            'F1+c1': ['F1pc1_diff'], 'F1+c2': ['F1pc2_diff'], 'F1+c3': ['F1pc3_diff'],
            'F1+all': ['F1pc123_diff'],
            'F1|c1': ['F1_diff', 'c1_diff'], 'F1|c2': ['F1_diff', 'c2_diff'],
            'F1|c3': ['F1_diff', 'c3_diff'],
            'F1|all': ['F1_diff', 'c1_diff', 'c2_diff', 'c3_diff'],
            'F1+all_identity': ['F1pc123id_diff']}
    X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])
    parts = []
    for s in SEASONS:
        tr, ho = H0[H0.season_year < s], H0[H0.season_year == s].copy()
        for n, c in VERS.items():
            ho['p_' + n] = X(ho, c) @ np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0]
        parts.append(ho)
    H = pd.concat(parts, ignore_index=True)
    y = H.margin.values
    SQ = {n: (y - H['p_' + n].values) ** 2 for n in VERS}

    code, keys = pd.factorize(H.home.astype(str) + '_' + H.season_year.astype(str))
    rng = np.random.default_rng(SEED)
    W = [np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)).astype(float)[code]
         for _ in range(REPS)]

    def est(a, b):
        f = lambda w: float(np.sqrt((w * SQ[a]).sum() / w.sum()) - np.sqrt((w * SQ[b]).sum() / w.sum()))
        draws = [f(w) for w in W]
        return {'point': round(f(np.ones(len(y))), 4),
                'ci95_home': [round(float(np.percentile(draws, q)), 4) for q in (2.5, 97.5)]}

    res = {'build': 'public rebuild', 'reps': REPS, 'component_counts': counts,
           'identity_residual_max': round(float(resid), 6),
           'paper_all_three_vs_F2_max_gap': round(float(gap), 6),
           'note_all_three':
               "fixed_weight['all three'] is the paper's definition, F1 plus the three components, "
               'which differs from the leaky feature by the player-games F1 calls Out who then '
               'played. The version that also subtracts those closes the identity and is the leaky '
               'feature itself, so its contrast against F1 equals the F1-against-F2 row by '
               'construction; it sits in the reference block and is a check, not a second put-back.',
           'fixed_weight': {}, 'free_coefficient': {}, 'reference': {}}
    print(f'\n{"contrast":24}{"fixed weight":>24}{"free coefficient":>24}')
    for lab, fx, fr in (('declared (c1)', 'F1+c1', 'F1|c1'), ('late (c2)', 'F1+c2', 'F1|c2'),
                        ('never (c3)', 'F1+c3', 'F1|c3'), ('all three', 'F1+all', 'F1|all')):
        a, b = est('F1', fx), est('F1', fr)
        res['fixed_weight'][lab] = a
        res['free_coefficient'][lab] = b
        print(f'{lab:24}{a["point"]:+8.4f} {str(a["ci95_home"]):>14}'
              f'{b["point"]:+8.4f} {str(b["ci95_home"]):>14}')
    for lab, a, b in (('F1 vs F2', 'F1', 'F2'), ('F1+c1 vs F2', 'F1+c1', 'F2'),
                      ('F1|all vs F2', 'F1|all', 'F2'),
                      ('all three, the paper definition, vs F2', 'F1+all', 'F2'),
                      ('all three with the fourth term vs F2', 'F1+all_identity', 'F2'),
                      ('all three with the fourth term vs F1', 'F1', 'F1+all_identity')):
        res['reference'][lab] = est(a, b)
        e = res['reference'][lab]
        print(f'{lab:24}{e["point"]:+8.4f} {str(e["ci95_home"]):>14}')

    OUT.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(f'\n-> {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
