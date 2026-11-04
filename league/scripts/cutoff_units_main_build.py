"""The benchmark comparisons restated in decision units: log loss and correct winners per 100 games.

Published unchanged below its original header. It reads the originating project's paths, so
it is not rerun here; its output is `results/paper2_cutoff_units.json`, which carries point estimates and intervals and no player rows.

The original header follows.

Plan CC (docs/plans/paper2_CC_cutoff_decision_units.md): the paired cutoff contrast in decision units.

  python league/scripts/../docs/paper/ssac27_submission/cutoff_units.py
  -> docs/paper/ssac27_submission/results/paper2_cutoff_units.json

Plan Y measured what waiting from six hours before tip-off to one hour is worth in points of margin RMSE.
RMSE is not a unit anyone decides in, and the SSAC paper's audience decides in wins, so this restates the
same contrast in log loss, Brier score and correct winners per 100 games.

The build is not reimplemented: `p6_ds_cutoff_f7.py` is imported, which runs its own build and its own
plan-hash check, so the sample, the encodings, the sit-out rates, the rolling origins and the bootstrap
bank are plan Y's by construction rather than by resemblance. Its writes sit behind `__main__`, so
importing it computes without touching `results/paper2_cutoff_f7.json`.

Probabilities come from each version's own training residual SD at each origin, which is how
`p6_ds_pooled_secondary.py` builds the manuscript's decision units. Nothing here uses the outcome to
select a report: both cutoffs are read from reports published before their own cutoff.

Output goes outside `results/` on purpose: the manuscript is frozen, this number does not belong to it,
and the self-audit ties the acknowledgement of its thirteen not-done claims to a hash of the result-file
inventory. The SSAC document's number check reads this file as its third pool.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CF7 = ROOT / 'league/scripts/p6_ds_cutoff_f7.py'
PLAN = ROOT / 'docs/plans/paper2_CC_cutoff_decision_units.md'
OUT = HERE / 'results/paper2_cutoff_units.json'

# Plan CC's controls, read off the committed result files. Nothing is interpreted unless all of them hold.
# The one-hour paired contrast read 0.0666 [0.0305, 0.1031] before the late window of Section 4 was
# collected. It moved on the same 4,131 games, which are now read at the last report published before the
# cutoff rather than the last one held; the pre-collection values stand in the frozen
# results/paper2_cutoff_curve.json. This file belongs to the conference paper alone and was replaced
# rather than duplicated, so the control is restated against what the script now writes. The other three
# did not move: paired_F7 is the rounded form and holds inside its own tolerance.
CONTROL = {'paired_F1': 0.0655, 'paired_F1_ci': [0.0291, 0.1018], 'paired_F7': 0.030,
           'f7_gain_6h': 0.2198, 'n': 4131}
TOL = {'paired_F1': 0.0001, 'paired_F1_ci': 0.0001, 'paired_F7': 0.002, 'f7_gain_6h': 0.002}


def _wavg(x, w):
    return float(np.sum(w * x) / np.sum(w))


def units_metrics(y, pred, sigma, w):
    """RMSE, log loss, Brier and accuracy under weights `w`.

    `sigma` is per game: the training residual SD of *this* version at the origin that game was held out
    at. Passing another version's SD changes log loss and Brier, which is why it is an argument rather
    than a constant.
    """
    from scipy.stats import norm
    win = (y > 0).astype(float)
    p = norm.cdf(pred / sigma).clip(1e-6, 1 - 1e-6)
    return {'rmse': float(np.sqrt(_wavg((y - pred) ** 2, w))),
            'logloss': -_wavg(win * np.log(p) + (1 - win) * np.log(1 - p), w),
            'brier': _wavg((p - win) ** 2, w),
            'acc': _wavg(((pred > 0) == (win == 1)).astype(float), w)}


def check_controls(seen: dict) -> list[str]:
    """Which of plan CC's controls fail, with what was seen. Empty means every control held."""
    bad = []
    if int(seen.get('n', -1)) != CONTROL['n']:
        bad.append(f"n is {seen.get('n')}, not {CONTROL['n']}")
    for key in ('paired_F1', 'paired_F7', 'f7_gain_6h'):
        got = seen.get(key)
        if got is None or abs(float(got) - CONTROL[key]) > TOL[key]:
            bad.append(f'{key} is {got}, not {CONTROL[key]} within {TOL[key]}')
    ci = seen.get('paired_F1_ci') or [None, None]
    for got, want in zip(ci, CONTROL['paired_F1_ci']):
        if got is None or abs(float(got) - want) > TOL['paired_F1_ci']:
            bad.append(f'paired_F1_ci is {ci}, not {CONTROL["paired_F1_ci"]}')
            break
    return bad


def _ci(v, nd=5):
    return [round(float(np.percentile(v, q)), nd) for q in (2.5, 97.5)]


def _build():
    """Import plan Y's script, which builds the sample on import and writes nothing."""
    spec = importlib.util.spec_from_file_location('p6_ds_cutoff_f7', CF7)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_with_sigma(cf7, cols):
    """Rolling-origin predictions plus, per game, the training residual SD of the version that made them."""
    H, BASE = cf7.H, cf7.BASE
    y, keep = [], []
    P = {k: [] for k in cols}
    S = {k: [] for k in cols}
    for s in cf7.SEASONS:
        tr, ho = H[H.season_year < s], H[H.season_year == s]
        X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])
        for k, c in cols.items():
            b = np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0]
            P[k].append(X(ho, c) @ b)
            sd = float(np.std(tr.margin.values - X(tr, c) @ b, ddof=len(b)))
            S[k].append(np.full(len(ho), sd))
        y.append(ho.margin.values)
        keep.append(ho)
    import pandas as pd
    return (np.concatenate(y), {k: np.concatenate(v) for k, v in P.items()},
            {k: np.concatenate(v) for k, v in S.items()}, pd.concat(keep))


def paired_units(cf7, late, early):
    """units(late) - units(early) inside the same draws, for every metric.

    A difference of two intervals is not the interval of a difference, so each replicate computes the
    contrast and the percentiles are taken over those.
    """
    y, P, S, Rk = fit_with_sigma(cf7, {'late': [late], 'early': [early]})
    W1, W2, W3, W4, W5 = cf7._draws(Rk)

    def st(w):
        a = units_metrics(y, P['early'], S['early'], w)
        b = units_metrics(y, P['late'], S['late'], w)
        return {'rmse': a['rmse'] - b['rmse'],          # lower is better: early minus late
                'logloss': a['logloss'] - b['logloss'],
                'brier': a['brier'] - b['brier'],
                'acc_per100': 100 * (b['acc'] - a['acc'])}   # higher is better: late minus early

    one = np.ones(len(y))
    pt = st(one)
    banks = {'home': W1, 'two_way': W2, 'season': W3, 'away': W4, 'iid': W5}
    B = {name: [st(w) for w in bank] for name, bank in banks.items()}
    out = {'n': int(len(y))}
    for m in ('rmse', 'logloss', 'brier', 'acc_per100'):
        nd = 2 if m == 'acc_per100' else (4 if m == 'rmse' else 5)
        out[m] = {'point': round(pt[m], nd)}
        for name, reps in B.items():
            out[m][f'ci95_{name}'] = _ci([r[m] for r in reps], nd)
    return out


def gain_units(cf7, col):
    """F0 -> this version at six hours, in the same units: the control for the F7 gain."""
    y, P, S, Rk = fit_with_sigma(cf7, {'F0': [], 'F': [col]})
    W1 = cf7._draws(Rk)[0]

    def st(w):
        a = units_metrics(y, P['F0'], S['F0'], w)
        b = units_metrics(y, P['F'], S['F'], w)
        return {'rmse': a['rmse'] - b['rmse'], 'logloss': a['logloss'] - b['logloss'],
                'brier': a['brier'] - b['brier'], 'acc_per100': 100 * (b['acc'] - a['acc'])}

    pt = st(np.ones(len(y)))
    B = [st(w) for w in W1]
    out = {'n': int(len(y))}
    for m in ('rmse', 'logloss', 'brier', 'acc_per100'):
        nd = 2 if m == 'acc_per100' else (4 if m == 'rmse' else 5)
        out[m] = {'point': round(pt[m], nd), 'ci95_home': _ci([b[m] for b in B], nd)}
    return out


def main(argv=None) -> int:
    import hashlib
    plan_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()[:16]
    print(f'plan CC {plan_sha}; building plan Y’s sample…', flush=True)
    cf7 = _build()

    # ---- controls first, run through plan Y's own functions rather than through copies of them
    c_f1 = cf7.paired('F1_1', 'F1_6')
    c_f7 = cf7.paired('F7_1', 'F7_6')
    c_g7 = cf7.gain_and_inflation('F7_6')
    seen = {'n': c_f1['n'], 'paired_F1': c_f1['point'], 'paired_F1_ci': c_f1['ci95_home'],
            'paired_F7': c_f7['point'], 'f7_gain_6h': c_g7['gain']}
    bad = check_controls(seen)
    print('  controls: ' + ('all hold' if not bad else '; '.join(bad)), flush=True)
    if bad:
        print('plan CC: controls failed, nothing written', file=sys.stderr)
        return 1

    # ---- the result
    res = {'plan': 'CC', 'plan_sha256_16': plan_sha, 'built_from': str(CF7.relative_to(ROOT)),
           'seed': cf7.SEED, 'replicates': cf7.REPS, 'sample': 'strict benchmark, both teams filed at 6h',
           'controls_seen': seen,
           'f1_paired_units': paired_units(cf7, 'F1_1', 'F1_6'),
           'f7_paired_units': paired_units(cf7, 'F7_1', 'F7_6'),
           'f7_paired_units_rates_reestimated': paired_units(cf7, 'F7_1_reest', 'F7_6'),
           'f1_gain_6h_units': gain_units(cf7, 'F1_6'),
           'f7_gain_6h_units': gain_units(cf7, 'F7_6')}

    for name in ('f1_paired_units', 'f7_paired_units'):
        d = res[name]
        print(f'\n  {name}: RMSE {d["rmse"]["point"]} {d["rmse"]["ci95_home"]}; '
              f'log loss {d["logloss"]["point"]} {d["logloss"]["ci95_home"]}; '
              f'winners/100 {d["acc_per100"]["point"]} {d["acc_per100"]["ci95_home"]}', flush=True)

    # The plan's conditional: the SSAC paper may claim the contrast in winner accuracy only if it is
    # resolved above zero there under the reported design. Recorded, not decided by prose later.
    res['verdicts'] = {
        enc: {'resolved_in_winner_accuracy':
              bool(res[f'{enc}_paired_units']['acc_per100']['ci95_home'][0] > 0),
              'resolved_in_log_loss':
              bool(res[f'{enc}_paired_units']['logloss']['ci95_home'][0] > 0)}
        for enc in ('f1', 'f7')}
    print('\n  verdicts: ' + json.dumps(res['verdicts']), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    assert 'results' not in OUT.relative_to(ROOT).parts[:2], 'plan CC writes outside the audited results/'
    OUT.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(f'\n-> {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
