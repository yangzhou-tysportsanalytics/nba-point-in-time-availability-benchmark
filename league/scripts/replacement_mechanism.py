"""The two mechanism tests of Section 4.6 under all three valuations, on this repository's own rebuild.

The paper narrows where the premium comes from with two tests: putting each component of the difference
between the two availability features back into the compliant one, and reading the compliant feature at a
later cutoff. Both were first computed under the valuation that charges an absence at the player's whole prior
efficiency, which is the valuation that makes the premium largest. A review asked the obvious question --
whether the two tests say the same thing under the other two valuations -- and the answer is that one of them
does and one does not, so the paper's claim about sizes is restricted to the two efficiency-based valuations
while the claim about direction is kept in general form.

Nothing here reimplements the build. The report rows, the name matching and the rule for which report is
current at a cutoff come from rebuild.py called with a different cutoff; the replacement level comes from
replacement_valuation.py's own estimator; the plus-minus history comes from history.py, which is the
originating project's module shipped unchanged.

    python league/scripts/rebuild.py build            # writes the player-level table this needs
    python league/scripts/replacement_mechanism.py    # -> results/paper2_replacement_mechanism_public.json

The first valuation is carried through the same code as a control. Where results/paper2_cutoff_curve_public.json
already exists, its curve must reproduce here, because the two scripts differ only in the loop around them; a
disagreement is this script's bug and not a finding, and it is printed as such.
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
from history import last_games_before_cutoff, season_prior_bounds            # noqa: E402
from replacement_valuation import levels_by_cutoff                           # noqa: E402

OUT = ROOT / 'results/paper2_replacement_mechanism_public.json'
CONTROL = ROOT / 'results/paper2_cutoff_curve_public.json'
CUTS_H = [12, 9, 6, 4, 3, 2, 1.5, 1, 0.5]
SEASONS = (2023, 2024, 2025, 2026)
BASE = ['strength_diff', 'rest_diff']
SEED, REPS = 20260921, 2000
VALS = ('base', 'rep', 'imp')
VAL_NAME = {'base': 'minutes x prior efficiency', 'rep': 'minutes x (efficiency - replacement)',
            'imp': 'minutes x plus-minus per minute'}


def valuations(R: pd.DataFrame, G: pd.DataFrame) -> dict:
    """The three per-player values, each built the way the published script for it builds it."""
    tg = pi.team_games()[['game_id', 'team', 'cutoff_utc', 'variant']].drop_duplicates(['game_id', 'team'])
    R = R.merge(tg, on=['game_id', 'team'], how='left')
    R['cutoff'] = pd.to_datetime(R.cutoff_utc, utc=True, errors='coerce')
    cut_hours = R.variant.map({'6h': 6.0, '90m': 1.5}).fillna(6.0)
    # tip-off from the cutoff and the rule that set it: assuming six hours for the fallback games would put
    # them four and a half hours late
    R['start'] = R.cutoff + pd.to_timedelta(cut_hours, unit='h')

    R['val_base'] = R.base_minutes * R.eff_prior

    hist = R.dropna(subset=['cutoff', 'eff_prior']).rename(columns={'base_minutes': 'minutes'})
    levels = levels_by_cutoff(hist[['start', 'minutes', 'eff_prior']], R.cutoff.dropna().unique())
    R['level'] = R.cutoff.map(levels)
    # not clipped: a below-replacement player keeps his negative value, because clipping would assume that no
    # absence can help a team. A missing level stays missing rather than reading as a costless absence.
    R['val_rep'] = R.base_minutes * (R.eff_prior - R.level)

    box = pi.box_scores()
    box['min'] = pd.to_numeric(box['min'], errors='coerce')
    box['plus_minus'] = pd.to_numeric(box.plus_minus, errors='coerce')
    box = box[box['min'].fillna(0) > 0].copy()
    box['pm_per_min'] = box.plus_minus / box['min']
    box['espn_player_id'] = box.espn_player_id.astype(str)
    box['game_id'] = box.game_id.astype(str)
    gstart = R[['game_id', 'start']].dropna().drop_duplicates('game_id')
    box = box.merge(gstart, on='game_id', how='left').merge(G[['game_id', 'season_year']], on='game_id',
                                                            how='left')
    bounds = season_prior_bounds(box.dropna(subset=['season_year']), 'pm_per_min')
    box['pm_per_min'] = box.pm_per_min.clip(box.season_year.map({k: v[0] for k, v in bounds.items()}),
                                            box.season_year.map({k: v[1] for k, v in bounds.items()}))
    R['pm20'] = last_games_before_cutoff(box, R, gstart, 'pm_per_min', min_periods=5,
                                         cutoff_hours=cut_hours.values, max_age_days=365)
    diag = {'rotation_rows': int(len(R)), 'rows_without_a_level': int(R.level.isna().sum()),
            'level_mean': round(float(R.level.mean()), 5),
            'share_below_replacement': round(float((R.val_rep < 0).mean()), 4),
            'pm20_coverage': round(float(R.pm20.notna().mean()), 4)}
    R['pm20'] = R.pm20.fillna(0.0)   # fewer than five prior games: the league-average impact, zero
    R['val_imp'] = R.base_minutes * R.pm20
    return {'R': R, 'diagnostics': diag}


def main() -> int:
    g, box, _ = rb.load_games_box()
    rows, blocks, _ = rb.load_reports(g)
    G = pi.games()
    built = valuations(pi.ref_players(), G)
    R, diag = built['R'], built['diagnostics']
    print('diagnostics:', json.dumps(diag), flush=True)

    k = list(zip(R.game_id.astype(str), R.team, R.espn_player_id.astype(str)))
    key = lambda d: set(zip(d.game_id.astype(str), d.team, d.pid.astype(str)))

    def diff_for(members: set, val: str) -> list:
        v = pd.Series(np.where([t in members for t in k], R[val], 0.0)).groupby([R.game_id, R.team]).sum()
        return [v.get((gm, h), 0.0) - v.get((gm, a), 0.0) for gm, h, a in zip(G.game_id, G.home, G.away)]

    def diff_flag(flag: str, val: str) -> list:
        v = pd.Series(np.where(R[flag].astype(bool), R[val], 0.0)).groupby([R.game_id, R.team]).sum()
        return [v.get((gm, h), 0.0) - v.get((gm, a), 0.0) for gm, h, a in zip(G.game_id, G.home, G.away)]

    for v in VALS:
        G[f'F1_{v}'] = diff_flag('out_pre', f'val_{v}')
        G[f'F2_{v}'] = diff_flag('out_real', f'val_{v}')

    filed = {}
    for h in CUTS_H:
        best, st = rb.status_at(g, rows, blocks, box, pd.Timedelta(hours=h))
        out = key(st[(st.status == 'Out') & st.pid.notna()])
        for v in VALS:
            G[f'F1c{h}_{v}'] = diff_for(out, f'val_{v}')
        have = set(zip(best.game_id.astype(str), best.team))
        filed[h] = np.array([((gm, hm) in have) and ((gm, aw) in have)
                             for gm, hm, aw in zip(G.game_id, G.home, G.away)])
        if h == 0.5:
            out30, listed30 = out, key(st[st.pid.notna()])
        print(f'{h:>5} h: features built, both teams filed in {filed[h].sum():,} games', flush=True)

    # the three components, each disjoint from the Out set the compliant feature already counts
    sat = {t for t, a in zip(k, R.out_real.astype(bool)) if a}
    f1_out = {t for t, a in zip(k, R.out_pre.astype(bool)) if a}
    comp = {'c1': {t for t in sat if t not in f1_out and t in out30},
            'c2': {t for t in sat if t not in f1_out and t not in out30 and t in listed30},
            'c3': {t for t in sat if t not in f1_out and t not in out30 and t not in listed30}}
    assert set().union(*comp.values()) == sat - f1_out, 'the components do not partition the absences'
    counts = {c: len(s) for c, s in comp.items()}
    print('components:', json.dumps(counts), flush=True)
    for v in VALS:
        for c, members in comp.items():
            G[f'{c}_{v}'] = diff_for(members, f'val_{v}')
        G[f'F1p1_{v}'] = G[f'F1_{v}'] + G[f'c1_{v}']
        G[f'F1p2_{v}'] = G[f'F1_{v}'] + G[f'c2_{v}']
        G[f'F1p3_{v}'] = G[f'F1_{v}'] + G[f'c3_{v}']

    def fit(sample: pd.DataFrame, vers: dict) -> pd.DataFrame:
        parts = []
        for s in SEASONS:
            tr, ho = sample[sample.season_year < s], sample[sample.season_year == s].copy()
            if not len(tr) or not len(ho):
                continue
            for n, c in vers.items():
                X = np.column_stack([np.ones(len(tr))] + [tr[x].values for x in BASE + c])
                Xh = np.column_stack([np.ones(len(ho))] + [ho[x].values for x in BASE + c])
                ho['p_' + n] = Xh @ np.linalg.lstsq(X, tr.margin.values, rcond=None)[0]
            parts.append(ho)
        return pd.concat(parts, ignore_index=True)

    def bank(H: pd.DataFrame) -> list:
        """Home-team x season clusters, the design the paper reports for these four seasons."""
        code, keys = pd.factorize(H.home.astype(str) + '_' + H.season_year.astype(str))
        rng = np.random.default_rng(SEED)
        return [np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)).astype(float)[code]
                for _ in range(REPS)]

    def contrast(H: pd.DataFrame, W: list, a: str, b: str) -> dict:
        y = H.margin.values
        sq = {n: (y - H['p_' + n].values) ** 2 for n in (a, b)}
        f = lambda w: float(np.sqrt((w * sq[a]).sum() / w.sum()) - np.sqrt((w * sq[b]).sum() / w.sum()))
        draws = [f(w) for w in W]
        return {'point': round(f(np.ones(len(y))), 4),
                'ci95_home': [round(float(np.percentile(draws, q)), 4) for q in (2.5, 97.5)],
                'n': int(len(y))}

    res = {'build': 'public rebuild', 'reps': REPS, 'seed': SEED, 'valuations': VAL_NAME,
           'diagnostics': diag, 'component_counts': counts, 'curve': {}, 'paired': {}, 'putbacks': {}}

    # ------------------------------------------------------------------ the curve and the paired contrasts
    fixed = filed[6] & ~G.any_block_missing.values
    HC = G[fixed].reset_index(drop=True)
    res['strict_six_hour_games'] = int(fixed.sum())
    print(f'\nstrict six-hour sample: {fixed.sum():,} games')
    for v in VALS:
        vers = {'F0': [], 'F2': [f'F2_{v}'], **{f'F1_{h}': [f'F1c{h}_{v}'] for h in CUTS_H}}
        Hv = fit(HC, vers)
        W = bank(Hv)
        res['curve'][v] = {str(h): {'gain_F1': contrast(Hv, W, 'F0', f'F1_{h}'),
                                    'premium': contrast(Hv, W, f'F1_{h}', 'F2')} for h in CUTS_H}
        res['paired'][v] = {'6h_minus_30min': contrast(Hv, W, 'F1_6', 'F1_0.5'),
                            '6h_minus_1h': contrast(Hv, W, 'F1_6', 'F1_1')}
        print(f'\n{v:5} {VAL_NAME[v]}')
        for h in CUTS_H:
            e, p = res['curve'][v][str(h)]['gain_F1'], res['curve'][v][str(h)]['premium']
            print(f'   {h:>4}h  gain {e["point"]:+.4f} {str(e["ci95_home"]):>20}'
                  f'   premium {p["point"]:+.4f} {str(p["ci95_home"]):>20}')
        for lab, e in res['paired'][v].items():
            print(f'   paired {lab:14} {e["point"]:+.4f} {e["ci95_home"]}')

    # ------------------------------------------------------------------ the put-backs
    HP = G[~G.any_block_missing].reset_index(drop=True)
    for v in VALS:
        vers = {'F0': [], 'F1': [f'F1_{v}'], 'F2': [f'F2_{v}'],
                'F1p1': [f'F1p1_{v}'], 'F1p2': [f'F1p2_{v}'], 'F1p3': [f'F1p3_{v}'],
                'free_c1': [f'F1_{v}', f'c1_{v}'], 'free_c2': [f'F1_{v}', f'c2_{v}'],
                'free_c3': [f'F1_{v}', f'c3_{v}']}
        Hv = fit(HP, vers)
        W = bank(Hv)
        res['putbacks'][v] = {
            'fixed_weight': {n: contrast(Hv, W, 'F1', n) for n in ('F1p1', 'F1p2', 'F1p3')},
            'free_coefficient': {n: contrast(Hv, W, 'F1', n) for n in ('free_c1', 'free_c2', 'free_c3')},
            'premium': contrast(Hv, W, 'F1', 'F2'), 'report_value': contrast(Hv, W, 'F0', 'F1')}
        e = res['putbacks'][v]
        print(f'\n{v:5} put-backs against F1 ({e["premium"]["n"]:,} games)')
        for n in ('F1p1', 'F1p2', 'F1p3'):
            f = n.replace('F1p', 'free_c')
            print(f'   {n:8} fixed {e["fixed_weight"][n]["point"]:+.4f} '
                  f'{str(e["fixed_weight"][n]["ci95_home"]):>20}   free '
                  f'{e["free_coefficient"][f]["point"]:+.4f} {str(e["free_coefficient"][f]["ci95_home"]):>20}')
        print(f'   premium  {e["premium"]["point"]:+.4f} {e["premium"]["ci95_home"]}'
              f'   report {e["report_value"]["point"]:+.4f} {e["report_value"]["ci95_home"]}')

    # ------------------------------------------------------------------ the control
    if CONTROL.exists():
        pub = json.loads(CONTROL.read_text(encoding='utf-8'))['curve']
        rowsc = []
        for h in CUTS_H:
            for lab, mine, theirs in (('gain', res['curve']['base'][str(h)]['gain_F1']['point'],
                                       pub[str(h)]['gain_F1']),
                                      ('premium', res['curve']['base'][str(h)]['premium']['point'],
                                       pub[str(h)]['inflation'])):
                rowsc.append({'what': f'{h}h {lab}', 'cutoff_curve_script': theirs, 'here': mine,
                              'difference': round(mine - theirs, 6)})
        bad = [r['what'] for r in rowsc if abs(r['difference']) > 5e-4]
        res['control_against_cutoff_curve'] = {'comparisons': rowsc, 'disagreements': bad,
                                               'note': 'the two scripts differ only in the loop around the '
                                                       'same build, so a disagreement is this script\'s bug'}
        print('\ncontrol against cutoff_curve.py, first valuation:')
        for r in rowsc:
            print(f'   {r["what"]:14} there {r["cutoff_curve_script"]:+.4f}  here {r["here"]:+.4f}  '
                  f'{"OK" if abs(r["difference"]) <= 5e-4 else "DIFFERS"}')
        if bad:
            print(f'\nthe control fails on {bad}; read nothing else from this run')
    else:
        print(f'\n{CONTROL.name} is not here, so the control was not run; '
              f'run league/scripts/cutoff_curve.py first')

    OUT.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(f'\n-> {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
