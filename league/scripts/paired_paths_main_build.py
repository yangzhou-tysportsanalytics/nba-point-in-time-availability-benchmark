"""Plan Z as it was run: the two build paths on one sample, with only the availability build swapped.

This is the main build's script, published unchanged below its original header. It needs both
builds' game-level tables and this repository ships one of them, so it reads the originating
project's paths and does not run here; `results/paper2_paired_paths.json` is its result.

The two pieces that can be wrong quietly -- aligning two tables that share no identifier, and
forming a contrast that uses one draw for both arms -- are in `src/paired_build.py` in the main
project with ten tests written before them.

The original header follows.

Plan Z (docs/plans/paper2_Z_paired_build_paths.md): the two build paths on one sample, with only the
availability build swapped.

  python league/scripts/p6_ds_paired_paths.py  -> results/paper2_paired_paths.json

Section 2.4 compares the main build's premium with the released path's by restricting each to the game
identifiers they share and reading each on its own remaining sample: 4,839 games against 4,835, with 4,833
held by both. The difference between the two point estimates is then set beside the Monte-Carlo error of a
bound, which is a comparison of magnitudes and not a test of the difference, and the section says so and
names this run as the one that would settle it.

Here the sample is fixed to the games both paths keep, the outcome and the two controls are the main
build's for both arms, the rolling split is the same, and the only thing that differs between the arms is
the pair (F1_diff, F2_diff). The premiums are then differenced **inside the same bootstrap draw**, which is
what pairing means and what two separately drawn intervals cannot give.

The controls themselves differ slightly between the paths -- 129 of 6,146 games on the strength control, 14
on rest -- so the run is repeated with the released path's controls as a sensitivity, and the paper reports
both if they disagree by more than the Monte-Carlo error of a bound.
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
sys.path.insert(0, str(ROOT))
from src.boot_designs import cluster_keys, weight_bank            # noqa: E402
from src.paired_build import aligned, paired_contrast             # noqa: E402
from src.path_compare import game_key                             # noqa: E402

PLAN = ROOT / 'docs/plans/paper2_Z_paired_build_paths.md'
PLAN_SHA = '4caa7484a7e10ac8'
_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
if not _sha.startswith(PLAN_SHA):
    raise SystemExit(f'the plan has changed since the rules were fixed: {_sha[:16]} != {PLAN_SHA}')

OUT = ROOT / 'results/paper2_paired_paths.json'
MAIN = ROOT / 'data/league/paper2_ext/bench_games.csv'
PUBLIC = ROOT / 'data/public/out/bench_games_public.csv'
SEED, REPS = 20260921, 10000
SEASONS = (2023, 2024, 2025, 2026)
BASE = ['strength_diff', 'rest_diff']
MC = 0.005                      # Section 3.1's Monte-Carlo error of a bound, for rule Z-f
CONTROL = {'gain_F1': 0.1966, 'inflation': 0.0588}      # the published pooled headline, for rule Z-a

ci = lambda v: [round(float(np.percentile(v, q)), 4) for q in (2.5, 97.5)]
X = lambda d, cols: np.column_stack([np.ones(len(d))] + [d[c].values for c in cols])


def fit_pooled(G: pd.DataFrame, specs: dict, base: list) -> tuple[np.ndarray, dict, pd.DataFrame]:
    """Rolling-origin predictions for every spec on one sample, one split and one outcome."""
    ys, P, keep, by_origin = [], {k: [] for k in specs}, [], {}
    for s in SEASONS:
        tr, ho = G[G.season_year < s], G[G.season_year == s]
        # both arms are fitted on the same rows at every origin, which is the point of the design and is
        # recorded rather than asserted: a common test set does not imply a common training set
        by_origin[str(s)] = {'train': int(len(tr)), 'test': int(len(ho))}
        for k, cols in specs.items():
            P[k].append(X(ho, base + cols) @ np.linalg.lstsq(X(tr, base + cols), tr.margin.values,
                                                             rcond=None)[0])
        ys.append(ho.margin.values)
        keep.append(ho)
    out = pd.concat(keep)
    out.attrs['by_origin'] = by_origin
    return np.concatenate(ys), {k: np.concatenate(v) for k, v in P.items()}, out


def read_build(path: Path, dates: dict | None) -> pd.DataFrame:
    g = pd.read_csv(path, dtype={'game_id': str})
    g['match_key'] = game_key(g, dates=dates)
    if g.match_key.duplicated().any():
        raise SystemExit(f'the match key is not unique in {path.name}; nothing written')
    return g


def main() -> int:
    _g = pd.read_csv(ROOT / 'data/league/paper2_ext/rebuild/data/games.csv', dtype={'game_id': str})
    _et = pd.to_datetime(_g.start_utc, utc=True).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    Gm = read_build(MAIN, dict(zip(_g.game_id, _et)))
    Gp = read_build(PUBLIC, None)

    out = {'plan': 'docs/plans/paper2_Z_paired_build_paths.md', 'plan_sha256': _sha,
           'seed': SEED, 'reps': REPS,
           'note': 'One sample, one outcome, one split and one pair of controls; the arms differ only in '
                   'which build the availability features come from, and the premiums are differenced '
                   'inside the same bootstrap draw.'}

    # ---------------------------------------------------------------- Z-a: the plumbing
    Gm_own = Gm[~Gm.any_block_missing].reset_index(drop=True)
    y0, P0, H0 = fit_pooled(Gm_own, {'F0': [], 'F1': ['F1_diff'], 'F2': ['F2_diff']}, BASE)
    sq0 = {k: (y0 - v) ** 2 for k, v in P0.items()}
    w1 = np.ones(len(y0))
    r = lambda sq, k, w: float(np.sqrt((w * sq[k]).sum() / w.sum()))
    ctrl = {'n': int(len(y0)),
            'gain_F1': round(r(sq0, 'F0', w1) - r(sq0, 'F1', w1), 4),
            'inflation': round(r(sq0, 'F1', w1) - r(sq0, 'F2', w1), 4),
            'expected': CONTROL}
    ok = all(abs(ctrl[k] - CONTROL[k]) < 5e-5 for k in CONTROL)
    ctrl['Z-a'] = 'pass' if ok else 'FAIL'
    out['control'] = ctrl
    print(f"  Z-a control: {ctrl['Z-a']}  gain {ctrl['gain_F1']} premium {ctrl['inflation']} "
          f"on {ctrl['n']} games", flush=True)
    if not ok:
        OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
        raise SystemExit('rule Z-a: the main build does not reproduce its published headline')

    # ---------------------------------------------------------------- the common sample
    km = Gm[~Gm.any_block_missing].reset_index(drop=True)
    kp = Gp[~Gp.any_block_missing].reset_index(drop=True)
    ia, ib, keys = aligned(km.match_key.tolist(), kp.match_key.tolist())
    A, B = km.iloc[ia].reset_index(drop=True), kp.iloc[ib].reset_index(drop=True)
    # Z-b: the two arms must be the same games in the same order with the same outcome
    assert len(A) == len(B) == len(keys), (len(A), len(B), len(keys))
    assert (A.match_key.values == B.match_key.values).all(), 'the two arms are not aligned'
    assert np.allclose(A.margin.values, B.margin.values), 'the two paths disagree about the outcome'

    G = A[['match_key', 'season_year', 'home', 'away', 'margin', 'strength_diff', 'rest_diff']].copy()
    G['mF1'], G['mF2'] = A.F1_diff.values, A.F2_diff.values
    G['pF1'], G['pF2'] = B.F1_diff.values, B.F2_diff.values
    G['pstrength'], G['prest'] = B.strength_diff.values, B.rest_diff.values
    out['sample'] = {
        'n_common_games': int(len(G)),
        'kept_by_main_only': int(len(km) - len(G)), 'kept_by_public_only': int(len(kp) - len(G)),
        'games_with_a_different_F1': int((G.mF1.values != G.pF1.values).sum()),
        'games_with_a_different_F2': int((G.mF2.values != G.pF2.values).sum()),
        'games_with_a_different_strength_control': int((G.strength_diff.values != G.pstrength.values).sum())}
    print(f"  common sample: {out['sample']['n_common_games']:,} games, "
          f"{out['sample']['kept_by_main_only']} ours only, "
          f"{out['sample']['kept_by_public_only']} theirs only", flush=True)

    # ---------------------------------------------------------------- the paired run
    def run(base: list, label: str) -> dict:
        specs = {'F0': [], 'mF1': ['mF1'], 'mF2': ['mF2'], 'pF1': ['pF1'], 'pF2': ['pF2']}
        y, P, H = fit_pooled(G, specs, base)
        sq = {k: (y - v) ** 2 for k, v in P.items()}
        banks = {d: list(weight_bank(np.random.default_rng(SEED), d, REPS, cluster_keys(H)))
                 for d in ('home_season_cluster', 'two_way_team', 'season_then_cluster')}
        one = np.ones(len(y))

        def arm(w, a, b):
            return r(sq, a, w) - r(sq, b, w)

        res = {'label': label, 'n': int(len(y)), 'by_origin': H.attrs.get('by_origin', {})}
        for name, a, b in (('main_gain', 'F0', 'mF1'), ('public_gain', 'F0', 'pF1'),
                           ('main_premium', 'mF1', 'mF2'), ('public_premium', 'pF1', 'pF2')):
            res[name] = {'point': round(arm(one, a, b), 4),
                         **{f'ci95_{d.split("_")[0]}': ci([arm(w, a, b) for w in banks[d]])
                            for d in banks}}
        for name, aa, bb in (('premium_main_minus_public', ('mF1', 'mF2'), ('pF1', 'pF2')),
                             ('gain_main_minus_public', ('F0', 'mF1'), ('F0', 'pF1'))):
            res[name] = {'point': round(paired_contrast(sq, one, aa, bb), 4),
                         **{f'ci95_{d.split("_")[0]}': ci([paired_contrast(sq, w, aa, bb) for w in banks[d]])
                            for d in banks}}
        return res

    out['main_controls'] = run(BASE, "one sample, the main build's controls")
    out['public_controls'] = run(['pstrength', 'prest'], "one sample, the released path's controls")

    for key in ('main_controls', 'public_controls'):
        d = out[key]
        print(f"\n  {d['label']} ({d['n']:,} holdout games)", flush=True)
        for name in ('main_premium', 'public_premium', 'premium_main_minus_public',
                     'main_gain', 'public_gain', 'gain_main_minus_public'):
            print(f"    {name:28} {d[name]['point']:+.4f}  {d[name]['ci95_home']}", flush=True)

    pm = out['main_controls']['premium_main_minus_public']
    gm = out['main_controls']['gain_main_minus_public']
    swap = abs(pm['point'] - out['public_controls']['premium_main_minus_public']['point'])
    out['rule_verdicts'] = {
        'Z-c (the two paths cannot be ordered)': bool(pm['ci95_home'][0] <= 0 <= pm['ci95_home'][1]),
        'Z-d (the constructions differ on a fixed sample)': bool(not (pm['ci95_home'][0] <= 0 <= pm['ci95_home'][1])),
        'Z-e (the gain does not simply reproduce)': bool(not (gm['ci95_home'][0] <= 0 <= gm['ci95_home'][1])),
        'Z-f (the controls matter)': bool(swap > MC),
        'controls_swap_moves_the_difference_by': round(swap, 4)}
    print('\n  rule verdicts:', json.dumps(out['rule_verdicts']), flush=True)

    OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'\n-> {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
