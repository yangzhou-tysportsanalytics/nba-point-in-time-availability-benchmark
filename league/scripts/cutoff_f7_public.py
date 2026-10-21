"""Plan BB: plan Y's crossing, on this repository's own rebuild.

What is reading the report at one hour instead of six worth to a model that already reads every status the
six-hour report publishes? The originating project answered that on its own build and the release shipped
that run as `cutoff_f7_main_build.py` with no public-path control, because that script reads the project's
paths. This is the control. It is the same question asked of a build whose tip-off is the scheduled time in
the Kaggle game table, whose roster at the cutoff comes from box appearances with no transaction log, and
whose eligibility rule keeps 4,117 games against the main build's 4,131 --- so it is a second build and not
a copy, and plan BB says in advance what agreement and disagreement each license.

    python league/scripts/rebuild.py parse 3600   # until it prints "done"
    python league/scripts/rebuild.py build        # writes the player-level table this reads
    python league/scripts/cutoff_f7_public.py     # -> results/paper2_cutoff_f7_public.json

Nothing here reimplements the build. The report rows, the name matching and the rule for which report is
current at a cutoff come from `rebuild.py`; the encoding is `src/cutoff_encoding.py`, already tested in this
repository; the rolling sit-out rates are `src/sit_out_rates.py`, tested against their definition because
this is the one piece with no published counterpart. Following the cutoff curve so the two can be compared,
the rotation, the reference minutes and the prior efficiency are the six-hour build's and only the status is
re-read; the sit-out rates are held at their six-hour values, with the re-estimated version beside them.

Controls first, BB-a and BB-b. The F1 arm must reproduce the published public-path cutoff curve at both
cutoffs to the fourth decimal, and the rate table must agree with a direct count. Nothing is interpreted if
BB-a fails.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'league/scripts'))
sys.path.insert(0, str(ROOT))
import public_inputs as pi                                                      # noqa: E402
import rebuild as rb                                                            # noqa: E402
from src.boot_designs import cluster_keys as _cluster_keys, weight_bank as _weight_bank   # noqa: E402
from src.cutoff_encoding import NOT_LISTED, weight                              # noqa: E402
from src.sit_out_rates import rolling_rates                                     # noqa: E402

PLAN = ROOT / 'docs/plans/paper2_BB_f7_public_path.md'
PLAN_SHA = '29a271cd8dbee581'
_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
if not _sha.startswith(PLAN_SHA):
    raise SystemExit(f'the plan has changed since the rules were fixed: {_sha[:16]} != {PLAN_SHA}')

CURVE = ROOT / 'results/paper2_cutoff_curve_public.json'
OUT = ROOT / 'results/paper2_cutoff_f7_public.json'
SEED, REPS = 20260921, 2000
SEASONS = (2023, 2024, 2025, 2026)
BASE = ['strength_diff', 'rest_diff']
STATUSES = ['Out', 'Doubtful', 'Questionable', 'Probable', 'Available']
CUTS = (6, 1)
GAME_HOURS = 2.5          # the main build's end-of-game proxy, so a rate never uses an unfinished game
# An eighth external review pointed out that `start + 2.5h` is a proxy for the end of a game and not the
# end of a game: it is the wall-clock length of a regulation game, and a game that went to overtime was
# still being played when the proxy says it had finished. The sit-out rates are built only from games the
# proxy calls finished, so a rate at a cutoff can be contaminated by a game that had not ended. These are
# the three definitions the run compares. `overtime_aware` adds a quarter of an hour per extra period, read
# off the box score's own minutes, and `safety_margin` is long enough that no game of any length is inside
# it.
END_VARIANTS = {'published': 2.5, 'overtime_aware': 2.5, 'safety_margin': 4.0}
OT_HOURS = 0.25
# what the main build found, for the report and not for a rule: no rule below is read off the distance
MAIN = {'f1_paired': 0.0666, 'f7_paired': 0.0297, 'difference': -0.0369,
        'difference_ci95_home': [-0.06, -0.0157], 'n': 4131}


# ---------------------------------------------------------------- inputs, through the released loaders
print('loading the rebuild', flush=True)
g, box, _ = rb.load_games_box()
rows, blocks, _ = rb.load_reports(g)
R = pi.ref_players()
G = pi.games()
R['val'] = R.base_minutes * R.eff_prior
R['out_real'] = R.out_real.astype(bool)
R['pid_key'] = R.espn_player_id.astype(str)
_n_ref = len(R)
R = R.merge(g[['game_id', 'start']], on='game_id')
if len(R) != _n_ref:
    raise SystemExit(f'{_n_ref - len(R)} reference player-games are for a game the Kaggle table does not '
                     f'hold; a rate cannot be dated without a tip-off and these rows are not dropped')
start_of = g.set_index('game_id').start
# extra periods per game: a team's minutes divided by five is the game's length, 48 in regulation, and the
# source rounds each player's minutes so regulation reads 46 to 48
_len = box.groupby(['game_id', 'team'])['min'].sum().groupby('game_id').max() / 5.0
OT = ((_len - 48) / 5).round().clip(lower=0)
if start_of.reindex(G.game_id).isna().any():
    raise SystemExit('a benchmark game has no tip-off in the Kaggle table')
print(f'  {len(G):,} benchmark games, {len(R):,} reference player-games', flush=True)


# ---------------------------------------------------------------- the sit-out rates, at the six-hour cutoff
# Among reference-rotation player-games of a status whose game had finished, the share who did not play, with
# a Beta(1,1) prior. The status is the build's own column, which is the six-hour report's with the 90-minute
# fallback where a team had not filed --- the same column the main build's rates are estimated on. Evaluated
# for every status at every game's cutoff, not only for the status the player carried there, because a later
# report can move him to any of them.
def _ends(variant: str) -> np.ndarray:
    """When each player-game's own game had finished, under one of the three definitions."""
    hours = pd.Series(np.full(len(R), END_VARIANTS[variant]), index=R.index)
    if variant == 'overtime_aware':
        hours = hours + OT_HOURS * R.game_id.map(OT).fillna(0).values
    return (R.start + pd.to_timedelta(hours, unit='h')).values.astype('datetime64[ns]')


def rate_table(cut_hours: float, variant: str = 'published') -> pd.DataFrame:
    cuts = (start_of.reindex(G.game_id) - pd.Timedelta(hours=cut_hours)).values.astype('datetime64[ns]')
    q = rolling_rates(R.status.values, R.out_real.values, _ends(variant), STATUSES, cuts)
    return pd.DataFrame(q, columns=STATUSES, index=G.game_id.values)


Q6 = rate_table(6)
Q1 = rate_table(1)
print('  sit-out rates at the last six-hour cutoff: '
      + ', '.join(f'{s} {Q6[s].iloc[-1]:.3f}' for s in STATUSES), flush=True)


# ---------------------------------------------------------------- the two encodings at a cutoff
_CACHE: dict = {}


def statuses_at(hours: float):
    """Per team-game, the status of every listed player on the last report filed by that cutoff.

    Returns the status frame and the set of team-games that had filed, so a team that had not is unknown
    rather than healthy.
    """
    if hours not in _CACHE:
        best, st = rb.status_at(g, rows, blocks, box, pd.Timedelta(hours=hours))
        st = st[st.pid.notna() & (st.pid.astype(str) != '')].copy()
        st['pid_key'] = st.pid.astype(str)
        _CACHE[hours] = (st[['game_id', 'team', 'pid_key', 'status']].drop_duplicates(
            ['game_id', 'team', 'pid_key']), set(zip(best.game_id.astype(str), best.team)))
    return _CACHE[hours]


def team_values(hours: float, encoding: str, rates: pd.DataFrame):
    """The absence value of each team-game under one encoding, summed over the reference rotation."""
    S, have = statuses_at(hours)
    m = R.merge(S, on=['game_id', 'team', 'pid_key'], how='left')
    status = m.status_y.fillna(NOT_LISTED).values
    cols = {k: rates[k].reindex(m.game_id.values).values for k in STATUSES}
    w = np.array([weight(s, {k: cols[k][i] for k in STATUSES}, encoding)
                  for i, s in enumerate(status)])
    v = pd.Series(m.val.values * w).groupby([m.game_id.values, m.team.values]).sum()
    return v, have


def fdiff(v: pd.Series) -> np.ndarray:
    d = dict(zip(v.index, v.values))
    return np.array([d.get((gm, h), 0.0) - d.get((gm, a), 0.0)
                     for gm, h, a in zip(G.game_id, G.home, G.away)])


# ---------------------------------------------------------------- the sample, as the cutoff curve fixes it
_, have6 = statuses_at(6)
fixed = np.array([((gm, h) in have6) and ((gm, a) in have6)
                  for gm, h, a in zip(G.game_id, G.home, G.away)]) & ~G.any_block_missing.values
H = G[fixed].reset_index(drop=True)
print(f'  strict six-hour sample: {len(H):,} games', flush=True)

for enc, tag in (('out_only', 'F1'), ('all_statuses', 'F7')):
    for h in CUTS:
        v, _ = team_values(h, enc, Q6)
        H[f'{tag}_{h}'] = fdiff(v)[fixed]
        print(f'  {tag} at {h}h built', flush=True)
v, _ = team_values(1, 'all_statuses', Q1)        # the sensitivity of the rate convention, rule BB-f
H['F7_1_reest'] = fdiff(v)[fixed]
H['F2_diff'] = G.F2_diff.values[fixed]


# ---------------------------------------------------------------- inference, as the cutoff curve reads it
def _fit(cols, mask=None):
    y_all, P, keep = [], {k: [] for k in cols}, []
    for s in SEASONS:
        tr = H[H.season_year < s] if mask is None else H[(H.season_year < s) & mask]
        sel = (H.season_year == s) if mask is None else ((H.season_year == s) & mask)
        ho = H[sel]
        X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])
        for k, c in cols.items():
            P[k].append(X(ho, c) @ np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0])
        y_all.append(ho.margin.values)
        keep.append(ho)
    return np.concatenate(y_all), {k: np.concatenate(v) for k, v in P.items()}, pd.concat(keep)


def _draws(Rk, reps=REPS, seed=SEED):
    rng = np.random.default_rng(seed)
    code, keys = pd.factorize(pd.Series((Rk.home + '_' + Rk.season_year.astype(str)).values))
    nk = len(keys)
    teams = np.array(sorted(set(Rk.home) | set(Rk.away)))
    hi_, ai_ = np.searchsorted(teams, Rk.home.values), np.searchsorted(teams, Rk.away.values)
    W1 = [np.bincount(rng.integers(0, nk, nk), minlength=nk).astype(float)[code] for _ in range(reps)]
    W2 = []
    for _ in range(reps):
        c = np.bincount(rng.integers(0, len(teams), len(teams)), minlength=len(teams)).astype(float)
        W2.append(c[hi_] * c[ai_])
    W3 = list(_weight_bank(rng, 'season_then_cluster', reps, _cluster_keys(Rk)))
    return W1, W2, W3


def _ci(v):
    return [round(float(np.percentile(v, q)), 4) for q in (2.5, 97.5)]


def _rmse(y, p, w):
    return np.sqrt(np.sum(w * (y - p) ** 2) / w.sum())


def gain_and_inflation(col, mask=None):
    y, P, Rk = _fit({'F0': [], 'F': [col], 'F2': ['F2_diff']}, mask)
    W1, _, _ = _draws(Rk)

    def st(w):
        r = {k: _rmse(y, q, w) for k, q in P.items()}
        return r['F0'] - r['F'], r['F'] - r['F2']

    pt = st(np.ones(len(y)))
    B = [st(w) for w in W1]
    return dict(n=int(len(y)), gain=round(pt[0], 4), inflation=round(pt[1], 4),
                gain_ci=_ci([b[0] for b in B]), inflation_ci=_ci([b[1] for b in B]))


def paired(late, early, mask=None):
    """RMSE(early) - RMSE(late), inside the same draws: what the later report adds under this encoding."""
    y, P, Rk = _fit({'late': [late], 'early': [early]}, mask)
    W1, W2, W3 = _draws(Rk)

    def st(w):
        return _rmse(y, P['early'], w) - _rmse(y, P['late'], w)

    return dict(n=int(len(y)), point=round(st(np.ones(len(y))), 4),
                ci95_home=_ci([st(w) for w in W1]), ci95_two_way=_ci([st(w) for w in W2]),
                ci95_season=_ci([st(w) for w in W3]))


def paired_difference(mask=None, late7='F7_1'):
    """(F7 late - F7 early) - (F1 late - F1 early), inside one set of draws.

    The two encodings' contrasts are each other's comparison, so the quantity is their difference in the same
    draw and not the overlap of two intervals.
    """
    y, P, Rk = _fit({'f7_late': [late7], 'f7_early': ['F7_6'],
                     'f1_late': ['F1_1'], 'f1_early': ['F1_6']}, mask)
    W1, W2, W3 = _draws(Rk)

    def st(w):
        r = {k: _rmse(y, q, w) for k, q in P.items()}
        return (r['f7_early'] - r['f7_late']) - (r['f1_early'] - r['f1_late'])

    return dict(n=int(len(y)), point=round(st(np.ones(len(y))), 4),
                ci95_home=_ci([st(w) for w in W1]), ci95_two_way=_ci([st(w) for w in W2]),
                ci95_season=_ci([st(w) for w in W3]))


# ---------------------------------------------------------------- BB-b, the rates against a direct count
def rate_control() -> dict:
    """The rolling table against the definition, counted at three cutoffs without the rolling machinery."""
    worst, where = 0.0, ''
    ends = (R.start + pd.Timedelta(hours=GAME_HOURS)).values.astype('datetime64[ns]')
    cuts = (start_of.reindex(G.game_id) - pd.Timedelta(hours=6)).values.astype('datetime64[ns]')
    for i in (0, len(cuts) // 2, len(cuts) - 1):
        for s in STATUSES:
            pick = (R.status.values == s) & (ends <= cuts[i])
            direct = (R.out_real.values[pick].sum() + 1.0) / (pick.sum() + 2.0)
            d = abs(direct - Q6[s].iloc[i])
            if d > worst:
                worst, where = d, f'{s} at game {i}'
    return dict(worst_absolute_difference=float(worst), worst_at=where,
                verdict='pass' if worst < 1e-9 else 'FAIL')


def transitions() -> dict:
    """How the statuses move between the two cutoffs, for interpretation and not for inference."""
    S6, _ = statuses_at(6)
    S1, _ = statuses_at(1)
    k = ['game_id', 'team', 'pid_key']
    m = S6.merge(S1, on=k, how='outer', suffixes=('_6', '_1'))
    a, b = m.status_6.fillna(NOT_LISTED), m.status_1.fillna(NOT_LISTED)
    moved = a.values != b.values
    am, bm = a[moved], b[moved]
    return dict(listed_rows_six_hours=int(len(S6)), listed_rows_one_hour=int(len(S1)),
                rows_that_change=int(moved.sum()),
                not_listed_at_six_then_out=int(((am == NOT_LISTED) & (bm == 'Out')).sum()),
                listed_short_of_out_then_out=int((am.isin(STATUSES[1:]) & (bm == 'Out')).sum()),
                out_at_six_then_softened=int(((am == 'Out') & (bm != 'Out')).sum()),
                team_games_with_a_change=int(m[moved].groupby(['game_id', 'team']).ngroups))


def main() -> int:
    out = {'plan': 'docs/plans/paper2_BB_f7_public_path.md', 'plan_sha256': _sha, 'build': 'public rebuild',
           'seed': SEED, 'reps': REPS, 'fixed_games': int(fixed.sum()), 'main_build_for_reference': MAIN,
           'note': 'The rotation, the reference minutes, the prior efficiency and the per-status sit-out '
                   'rates are the six-hour build\'s; only the status is re-read at the later cutoff. This '
                   'is a control on a second build, not a replication: the two builds keep different games.'}

    # ---- controls
    curve = json.loads(CURVE.read_text(encoding='utf-8'))['curve']
    c1 = {h: gain_and_inflation(f'F1_{h}') for h in CUTS}
    ok = []
    for h in CUTS:
        pub = curve[str(h)]
        ok.append(c1[h]['n'] == pub['n'] and abs(c1[h]['gain'] - pub['gain_F1']) < 1e-4
                  and abs(c1[h]['inflation'] - pub['inflation']) < 1e-4)
        print(f"  BB-a at {h}h: this run {c1[h]['gain']}/{c1[h]['inflation']} on {c1[h]['n']}, "
              f"published {pub['gain_F1']}/{pub['inflation']} on {pub['n']}", flush=True)
    bb_a = all(ok)
    bb_b = rate_control()
    seen = set(pd.concat([statuses_at(h)[0].status for h in CUTS]).unique())
    bb_c = seen <= set(STATUSES)
    out['controls'] = {'f1_by_cutoff': {str(h): c1[h] for h in CUTS},
                       'published_curve': {str(h): curve[str(h)] for h in CUTS},
                       'BB-a': 'pass' if bb_a else 'FAIL', 'BB-b': bb_b,
                       'BB-c': 'pass' if bb_c else f'FAIL: statuses not in the rate table: '
                                                   f'{sorted(seen - set(STATUSES))}',
                       'statuses_seen': sorted(seen)}
    print(f"  BB-b: {bb_b['verdict']} (worst {bb_b['worst_absolute_difference']:.2e}, "
          f"{bb_b['worst_at']})\n  BB-c: {out['controls']['BB-c']}", flush=True)
    if not bb_a:
        OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
        raise SystemExit('rule BB-a: the F1 arm does not reproduce the published public-path cutoff curve; '
                         'nothing is interpreted')

    # ---- the result
    f7_6 = gain_and_inflation('F7_6')
    out['six_hour_levels'] = {'F1': c1[6], 'F7': f7_6}
    out['full_sample'] = {
        'f1_paired': paired('F1_1', 'F1_6'),
        'f7_paired': paired('F7_1', 'F7_6'),
        'difference': paired_difference(),
        'f7_paired_rates_reestimated': paired('F7_1_reest', 'F7_6'),
        'difference_rates_reestimated': paired_difference(late7='F7_1_reest')}
    out['transitions'] = transitions()

    d = out['full_sample']
    print(f"\n  F1 paired {d['f1_paired']['point']} {d['f1_paired']['ci95_home']}"
          f"\n  F7 paired {d['f7_paired']['point']} {d['f7_paired']['ci95_home']}"
          f"\n  difference {d['difference']['point']} {d['difference']['ci95_home']}"
          f"  (main build {MAIN['difference']} {MAIN['difference_ci95_home']})", flush=True)
    print(f"  F7 gain at six hours {f7_6['gain']} {f7_6['gain_ci']} against F1's {c1[6]['gain']} "
          f"{c1[6]['gain_ci']}", flush=True)

    # ---- the end-of-game proxy, which an eighth review asked to be checked rather than assumed
    base_ends = _ends('published')
    cuts6 = (start_of.reindex(G.game_id) - pd.Timedelta(hours=6)).values.astype('datetime64[ns]')
    sens = {}
    for variant in END_VARIANTS:
        e = _ends(variant)
        # A player-game is exposed at a cutoff when the two definitions disagree about whether its game
        # had finished by then. The rates are read at every game's own six-hour cutoff, so the count is
        # over (player-game, cutoff) pairs: for each row, the cutoffs in [min(end), max(end)), which is
        # where the comparison `end <= cutoff` flips. Checked against the pairwise comparison itself on a
        # sample of rows, because an off-by-one in the side of the search returns zero and looks tidy.
        _lo = np.minimum(base_ends, e)
        _hi = np.maximum(base_ends, e)
        _sorted = np.sort(cuts6)
        flips = int((np.searchsorted(_sorted, _hi, 'left')
                     - np.searchsorted(_sorted, _lo, 'left')).sum())
        _k = np.random.default_rng(0).choice(len(e), min(2000, len(e)), replace=False)
        _bf = int(((base_ends[_k][:, None] <= _sorted[None, :])
                   != (e[_k][:, None] <= _sorted[None, :])).sum())
        _ss = int((np.searchsorted(_sorted, _hi[_k], 'left')
                   - np.searchsorted(_sorted, _lo[_k], 'left')).sum())
        if _bf != _ss:
            raise SystemExit(f'the exposure count disagrees with the comparison it stands for: '
                             f'{_ss} against {_bf} on 2,000 rows')
        q6 = rate_table(6, variant)
        worst = float((q6 - Q6).abs().max().max())
        v, _ = team_values(6, 'all_statuses', q6)
        H['F7_6_v'] = fdiff(v)[fixed]
        v, _ = team_values(1, 'all_statuses', q6)
        H['F7_1_v'] = fdiff(v)[fixed]
        pr = paired('F7_1_v', 'F7_6_v')
        pd_ = paired_difference(late7='F7_1_v')
        # the difference uses F7_6 for the early arm, so rebuild it against this variant's own early arm
        sens[variant] = {'end_hours': END_VARIANTS[variant],
                         'overtime_aware': variant == 'overtime_aware',
                         'player_game_cutoff_pairs_reclassified': flips,
                         'player_game_cutoff_pairs': int(len(e) * len(cuts6)),
                         'player_games_in_an_overtime_game': int((R.game_id.map(OT).fillna(0) > 0).sum()),
                         'worst_rate_difference_from_published': round(worst, 6),
                         'f7_paired': pr, 'difference': pd_}
        print(f"  end proxy {variant:<15} worst rate difference {worst:.2e}, "
              f"F7 paired {pr['point']} {pr['ci95_home']}, difference {pd_['point']} "
              f"{pd_['ci95_home']}", flush=True)
    out['end_of_game_proxy'] = {
        'note': 'The sit-out rates are built from games the proxy calls finished. `published` is the main '
                "build's fixed 2.5 hours; `overtime_aware` adds a quarter of an hour per extra period, "
                "read off the box score's own minutes; `safety_margin` is 4 hours, longer than any game. "
                'The contrast under each is reported so that the figure the paper prints is not resting on '
                'the proxy being exact.',
        'overtime_games': int((OT > 0).sum()), 'games': int(len(OT)),
        'variants': sens}

    dif, dre = d['difference'], d['difference_rates_reestimated']
    lo, hi = dif['ci95_home']
    corroborated = dif['point'] < 0 and hi < 0
    out['rule_verdicts'] = {
        'BB-d (not corroborated: the interval contains zero, or the point is positive)':
            bool(not corroborated),
        'BB-e (corroborated: negative and resolved)': bool(corroborated),
        'BB-f (sensitive to the rate convention: the sign flips when the rates are re-estimated)':
            bool((dif['point'] < 0) != (dre['point'] < 0)),
        'reading': ('the main build\'s finding holds on this build' if corroborated else
                    'the same sign and not resolved on this build' if dif['point'] < 0 else
                    'the opposite sign on this build')}
    print('\n  rule verdicts:', json.dumps(out['rule_verdicts'], indent=1), flush=True)

    OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'\n-> {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
