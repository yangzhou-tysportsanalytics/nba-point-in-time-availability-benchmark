"""Plan AA section 5 as it was run: what the returned roster audit says.

Published unchanged below its original header. It reads the returned worksheet, which is a player-level
table and is not redistributed, so this script is not rerun here; its output is
`results/paper2_roster_audit_result.json`, which carries rates, counts and the corrected build and no
player rows.

Two things in it are worth reading rather than taking on trust. The first is the weighting: the sampler drew
its strata in sequence and excluded what it had already drawn, so the three frames overlap and weighting
each stratum by its own frame counts the population twice over --- the header below gives the number. The
estimates are post-stratified on the definitional cells instead, and the control is that the weighted
estimate of the flagged absence value comes back within 2% of the population total. The second is the rule
that fired: AA-a, the corrected premium moves by less than the Monte-Carlo error of a bound, so the audit is
reported as a bound on this error and the headline does not move.

Import names: the originating project's layout is `src/roster_precision.py` (shipped here as
`src/roster_precision.py`), `league/scripts/p6_ds_roster_precision.py` (here
`league/scripts/roster_precision_main_build.py`) and `league/scripts/p6_ds_roster_audit_sample.py` (here
`league/scripts/roster_audit_sample_main_build.py`). The import and `importlib` paths below are the
project's, not this repository's, which is one reason these three are published to be read rather than
run.

The original header follows.

Plan AA section 5: what the returned roster audit says, under the rules fixed in section 6.

  python league/scripts/p6_ds_roster_audit_result.py  -> results/paper2_roster_audit_result.json

The audit asks, of 150 stints drawn at seed 20261020, whether a player the participation feature counts as
absent was still on that team. This script reads the returned verdicts and does what the plan says: the
error rate by stints, by rows and by absence value, within each stratum and reweighted to the flagged
population; the same with `cannot_confirm` counted both ways, as a range; the control stratum against the
flagged ones; and the corrected build, which is a new result file beside the old and never a rewrite of it.

The population is rebuilt through `p6_ds_roster_audit_sample.population()`, the construction that drew the
sample, so the weights are not a second definition of it.

**How the sample is weighted, and why not the obvious way.** The sampler drew the strata in sequence and
excluded what it had already drawn: stratum B was drawn from the flagged stints *left after A*, and C from
what was left after B. So the three frames overlap --- a top-decile stint not drawn into A is still in B's
frame --- and weighting each stratum by its own frame counts the population twice over. It does: that
weighting puts the flagged absence value at 221,693 against the 110,754 the sample file records, a factor
of two. The estimates below are therefore post-stratified on the definitional cells, which do partition the
879 flagged stints: high value, then near a transaction, then the rest. As drawn, the 150 fall 54 / 64 / 2
into those cells rather than 50 / 40 / 30, and that weighting puts the flagged absence value at 108,503
against 110,754, within 2%. It is control 2 below.

**What the realized draw cannot say much about.** Two of the 150 land in the cell of flagged stints that are
neither high value nor near a transaction, because after A and B were removed the remainder was still mostly
near-transaction stints. That cell holds 86 stints and 11,724 of the flagged absence value, and its rate
here rests on two judgements. The intervals carry it; the text should say it.

What the rates are and are not: the proxy that selected the flagged stints reads the rest of the season, so
it is a sampling device, not a roster rule, and these rates bound one error in one direction. A player
wrongly removed from a list is invisible here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PLAN = ROOT / 'docs/plans/paper2_AA_roster_audit.md'
PLAN_SHA = 'b2accd0a0fb42060'
_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
if not _sha.startswith(PLAN_SHA):
    raise SystemExit(f'the plan has changed since the rules were fixed: {_sha[:16]} != {PLAN_SHA}')

FILLED = ROOT / 'docs/paper/audit/roster_audit_filled.csv'
SAMPLE = ROOT / 'results/paper2_roster_audit_sample.json'
OUT = ROOT / 'results/paper2_roster_audit_result.json'
SEED, REPS = 20261021, 10000
FLAGGED = ['A_high_value', 'B_near_transaction', 'C_flagged_random']
CONTROL = 'D_control'
# The measures, all three on the same row set. The worksheet's own `rows_counted_absent` and
# `absence_value_carried` are whole-stint totals over every absence in that (player, team, season), which
# is the right question to put to an auditor and the wrong numerator for an error rate: a player who left
# in February has genuine absences in December, and the denominator the paper quotes --- the absence value
# of rows where the proxy can be read --- is a different row set again. So the measures below are rebuilt
# per stint from the player-level table, restricted to the readable rows, with the artefact part being the
# rows at or after the confirmed departure.
MEASURES = {'stints': None, 'rows': 'abs_rows_readable', 'value': 'abs_value_readable'}
ARTEFACT = {'stints': None, 'rows': 'art_rows_readable', 'value': 'art_value_readable'}
VALUE_TOLERANCE = 0.10          # how far the weighted estimate of the flagged value may sit from the total


def _sampler():
    spec = importlib.util.spec_from_file_location(
        '_sampler', ROOT / 'league/scripts/p6_ds_roster_audit_sample.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sid(d: pd.DataFrame) -> list[str]:
    return [f'{p}_{t}_{int(s)}' for p, t, s in zip(d.espn_player_id, d.team, d.season_year)]


def cells(stint: pd.DataFrame, ctrl: pd.DataFrame) -> tuple[dict[str, str], dict[str, int]]:
    """The definitional partition of the flagged stints, plus the controls, as stint_id -> cell."""
    cut = stint.absence_value.quantile(0.9)
    lab = ['A_high_value' if v >= cut else ('B_near_transaction' if nt else 'C_flagged_random')
           for v, nt in zip(stint.absence_value, stint.near_transaction)]
    of = dict(zip(_sid(stint), lab))
    of.update({s: CONTROL for s in _sid(ctrl)})
    N = {k: lab.count(k) for k in FLAGGED}
    N[CONTROL] = int(len(ctrl))
    return of, N


def rate(w: pd.DataFrame, weights: dict[str, float], measure: str | None, which: list[str],
         errors: np.ndarray, artefact: str | None = None) -> float:
    """A post-stratified share on one row set: the artefact part over the whole of it.

    For stints the numerator counts the stints in error. For rows and value the numerator is the part of
    the stint the confirmed departure accounts for, not the whole stint, and the denominator is the same
    rows; `errors` selects which stints contribute a numerator at all, so the two readings of
    `cannot_confirm` differ in whether an unconfirmable stint contributes its whole readable self.
    """
    num = den = 0.0
    for c in which:
        pick = (w.cell == c).values
        if measure is None:
            m = np.ones(int(pick.sum()))
            a = m
        else:
            m = w[measure].values[pick].astype(float)
            # a stint selected as an error by a reading that has no departure date for it (an
            # unconfirmable one) contributes all of its readable absence, which is the reading
            a = np.where(np.isnan(w[artefact].values[pick].astype(float)), m,
                         w[artefact].values[pick].astype(float))
        num += weights[c] * float(a[errors[pick]].sum())
        den += weights[c] * float(m.sum())
    return float('nan') if den == 0 else num / den


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """The one-sided upper confidence limit for a binomial rate, which 0 of n needs and a bootstrap of it
    cannot give: resampling zero errors returns zero every time."""
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(1 - alpha, k + 1, n - k))


def per_stint_measures(w: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    """Rebuild each sampled stint's absence rows and value, and the part a confirmed departure accounts for.

    Readable means the rows the paper's denominator is built from, those with ten team-games still to come.
    The artefact part is the absence rows at or after `left_date`; it is NaN where there is no date, which
    is how the unconfirmable reading knows to take the whole stint instead.
    """
    R = R.copy()
    R['sid'] = [f'{p}_{t}_{int(s)}' for p, t, s in zip(R.espn_player_id, R.team, R.season_year)]
    R = R[R.sid.isin(set(w.stint_id)) & R.out_real.values]
    readable = (R.remaining_games >= 10).values
    left = {r.stint_id: pd.Timestamp(r.left_date).tz_localize('UTC')
            for r in w.itertuples() if r.verdict == 'had_left'}
    after = np.array([s in left and ts >= left[s] for s, ts in zip(R.sid, R.start)])

    R = R.assign(_rd=readable, _af=after)
    R['_v'] = R.value.values
    R['_rdv'] = R._v * R._rd
    R['_arv'] = R._v * (R._rd & R._af)
    R['_arr'] = (R._rd & R._af).astype(int)
    R['_rdr'] = R._rd.astype(int)
    out = R.groupby('sid').agg(abs_rows_all=('_v', 'size'), abs_value_all=('_v', 'sum'),
                               abs_rows_readable=('_rdr', 'sum'), abs_value_readable=('_rdv', 'sum'),
                               art_rows_readable=('_arr', 'sum'), art_value_readable=('_arv', 'sum'))
    m = w.merge(out, left_on='stint_id', right_index=True, how='left')
    for c in ('abs_rows_all', 'abs_value_all', 'abs_rows_readable', 'abs_value_readable'):
        m[c] = m[c].fillna(0.0)
    # no date means no artefact part; NaN is the signal, not zero
    for c in ('art_rows_readable', 'art_value_readable'):
        m[c] = np.where(m.verdict.values == 'had_left', m[c].fillna(0.0).values, np.nan)
    return m


EXT = ROOT / 'data/league/paper2_ext'
PUBLISHED = {'gain': 0.197, 'inflation': 0.059}      # the pooled figures the paper prints, to be reproduced
MC_MAX = 0.006                                       # the Monte-Carlo error of a reported bound, nMCmax
HOLDOUTS = (2023, 2024, 2025, 2026)
BASE = ['strength_diff', 'rest_diff']


def _pooled(G: pd.DataFrame, f2: str, reps: int = REPS, seed: int = 20260921) -> dict:
    """The pooled rolling headline on one participation column, under the design the paper reports."""
    X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])
    ys, P, keep = [], {k: [] for k in ('F0', 'F1', 'F2')}, []
    for s in HOLDOUTS:
        tr, ho = G[G.season_year < s], G[G.season_year == s]
        for k, c in (('F0', []), ('F1', ['F1_diff']), ('F2', [f2])):
            P[k].append(X(ho, c) @ np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0])
        ys.append(ho.margin.values)
        keep.append(ho)
    y, H = np.concatenate(ys), pd.concat(keep)
    sq = {k: (y - np.concatenate(v)) ** 2 for k, v in P.items()}
    code, keys = pd.factorize(pd.Series((H.home + '_' + H.season_year.astype(str)).values))
    rng = np.random.default_rng(seed)
    W = [np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)).astype(float)[code]
         for _ in range(reps)]

    def st(w_, a, b):
        t = w_.sum()
        return float(np.sqrt((w_ * sq[a]).sum() / t) - np.sqrt((w_ * sq[b]).sum() / t))

    o = {'n': int(len(y))}
    for lab, a, b in (('gain', 'F0', 'F1'), ('inflation', 'F1', 'F2')):
        o[lab] = round(st(np.ones(len(y)), a, b), 4)
        d = [st(w_, a, b) for w_ in W]
        o[lab + '_ci'] = [round(float(np.percentile(d, q)), 4) for q in (2.5, 97.5)]
    return o


def corrected_build(w: pd.DataFrame) -> dict:
    """The participation feature with the confirmed departures removed from the cutoff onward.

    Only the stints this audit judged `had_left` are corrected, 26 of the 879 flagged, so this is the
    confirmed part of the error and not an estimate of all of it: the value-weighted rate above is what
    bounds the rest. The feature is rebuilt from the player-level table rather than edited in place, and
    the control is that rebuilding it without any correction reproduces the published figures.
    """
    R = pd.read_csv(EXT / 'bench_ref_players.csv', dtype={'game_id': str, 'espn_player_id': str})
    G = pd.read_csv(EXT / 'bench_games.csv', dtype={'game_id': str})
    games = pd.read_csv(EXT / 'rebuild/data/games.csv', dtype={'game_id': str})
    games['start'] = pd.to_datetime(games.start_utc, utc=True)
    R = R.merge(games[['game_id', 'start']], on='game_id', how='left')
    R['val'] = R.base_minutes * R.eff_prior
    R['out_real'] = R.out_real.astype(bool)

    def f2_diff(flag: np.ndarray) -> pd.Series:
        v = pd.Series(R.val.values * flag).groupby([R.game_id.values, R.team.values]).sum()
        d = dict(zip(v.index, v.values))
        return pd.Series([d.get((g, h), 0.0) - d.get((g, a), 0.0)
                          for g, h, a in zip(G.game_id, G.home, G.away)], index=G.index)

    G['F2_rebuilt'] = f2_diff(R.out_real.values)
    worst = float((G.F2_rebuilt - G.F2_diff).abs().max())

    left = {r.stint_id: pd.Timestamp(r.left_date).tz_localize('UTC')
            for r in w.itertuples() if r.verdict == 'had_left'}
    sid = [f'{p}_{t}_{int(s)}' for p, t, s in zip(R.espn_player_id, R.team, R.season_year)]
    gone = np.array([s in left and ts >= left[s] for s, ts in zip(sid, R.start)])
    G['F2_corrected'] = f2_diff(R.out_real.values & ~gone)

    B = G[~G.any_block_missing].reset_index(drop=True)
    pub = _pooled(B, 'F2_diff')
    reb = _pooled(B, 'F2_rebuilt')
    cor = _pooled(B, 'F2_corrected')
    control_ok = (worst < 1e-6
                  and abs(pub['gain'] - PUBLISHED['gain']) < 5e-4
                  and abs(pub['inflation'] - PUBLISHED['inflation']) < 5e-4)
    moved = abs(cor['inflation'] - pub['inflation'])
    o = {
        'note': 'Only the 26 stints judged had_left are corrected, out of 879 flagged, so this is the '
                'confirmed part of the error. The value-weighted rate above bounds the rest. The strict '
                'headline and the never-listed put-back re-read the report series in two other scripts '
                'and are not rerun here.',
        'corrected_stints': len(left),
        'rows_removed_from_the_absence_set': int((gone & R.out_real.values).sum()),
        'absence_value_removed': round(float(R.val.values[gone & R.out_real.values].sum()), 2),
        'control_rebuilt_feature_matches': {'worst_absolute_difference': worst,
                                            'published': PUBLISHED, 'rebuilt': pub,
                                            'verdict': 'pass' if control_ok else 'FAIL'},
        'pooled_published': pub, 'pooled_corrected': cor,
        'premium_moves_by': round(moved, 4), 'gain_moves_by': round(abs(cor['gain'] - pub['gain']), 4),
        'MC_max': MC_MAX,
        'AA-a (within the Monte-Carlo error of a bound: the headline does not move)': bool(moved <= MC_MAX),
        'AA-b (larger than that: both are reported)': bool(MC_MAX < moved <= 0.5 * pub['inflation']),
        'AA-c (larger than half the premium: roster precision is first order)':
            bool(moved > 0.5 * pub['inflation']),
    }
    print(f"\n  corrected build: {o['corrected_stints']} stints, "
          f"{o['rows_removed_from_the_absence_set']} absent rows removed, "
          f"{o['absence_value_removed']:,.0f} of absence value")
    print(f"    control: rebuilt feature matches the shipped one to {worst:.2e}; rebuilt pooled "
          f"{pub['gain']}/{pub['inflation']} against the published {PUBLISHED['gain']}/"
          f"{PUBLISHED['inflation']} -> {o['control_rebuilt_feature_matches']['verdict']}")
    print(f"    corrected pooled {cor['gain']} {cor['gain_ci']} / {cor['inflation']} {cor['inflation_ci']}")
    print(f"    the premium moves by {moved:.4f} against an MC error of {MC_MAX}")
    for k, v in o.items():
        if k.startswith('AA-') and v:
            print(f'    {k}')
    return o


def main() -> int:
    w = pd.read_csv(FILLED, dtype={'espn_player_id': str})
    meta = json.loads(SAMPLE.read_text(encoding='utf-8'))
    if w.verdict.isna().any() or (w.verdict == '').any():
        raise SystemExit('the worksheet is not fully judged; nothing written')

    print('rebuilding the population through the sampler', flush=True)
    _R, stint, ctrl, den_readable, _name = _sampler().population()
    cell_of, N = cells(stint, ctrl)

    # ---- control 1: every drawn stint is in the population it was drawn from
    unknown = [s for s in w.stint_id if s not in cell_of]
    w['cell'] = [cell_of.get(s, '') for s in w.stint_id]
    w = per_stint_measures(w, _R)
    n = {k: int((w.cell == k).sum()) for k in N}
    weights = {k: N[k] / n[k] for k in N if n[k]}

    # ---- control 2: the weighted estimate of the flagged absence value against the population total
    est_value = sum(weights[c] * float(w.absence_value_carried[w.cell == c].sum()) for c in FLAGGED)
    true_value = float(meta['population']['absence_value_flagged'])
    ratio = est_value / true_value
    crosstab = {k: w.cell[w.stratum == k].value_counts().to_dict() for k in N}
    controls = {
        'stints_not_in_the_population': unknown,
        'drawn_label_against_definitional_cell': crosstab,
        'population_by_cell': N, 'sample_by_cell': n,
        'weights': {k: round(v, 4) for k, v in weights.items()},
        'flagged_absence_value_weighted_estimate': round(est_value, 2),
        'flagged_absence_value_population': round(true_value, 2),
        'ratio': round(ratio, 4),
        'verdict': 'pass' if (not unknown and abs(ratio - 1) <= VALUE_TOLERANCE) else 'FAIL',
    }
    print(f'  population by cell {N}')
    print(f'  sample by cell     {n}   weights ' + ', '.join(f'{k} {v:.2f}' for k, v in weights.items()))
    print(f'  flagged absence value: weighted estimate {est_value:,.0f} against {true_value:,.0f} '
          f'(ratio {ratio:.3f}, tolerance {VALUE_TOLERANCE})')
    if controls['verdict'] == 'FAIL':
        OUT.write_text(json.dumps({'controls': controls}, indent=1, ensure_ascii=False), encoding='utf-8')
        raise SystemExit('the controls do not pass; nothing is interpreted')

    share_cannot = float((w.verdict == 'cannot_confirm').mean())
    out = {'plan': 'docs/plans/paper2_AA_roster_audit.md', 'plan_sha256': _sha,
           'seed': SEED, 'reps': REPS, 'filled': FILLED.name,
           'verdicts': w.verdict.value_counts().to_dict(),
           'verdicts_by_drawn_stratum': {k: w.verdict[w.stratum == k].value_counts().to_dict() for k in N},
           'verdicts_by_cell': {k: w.verdict[w.cell == k].value_counts().to_dict() for k in N},
           'cannot_confirm_share': round(share_cannot, 4),
           'AA-e (inconclusive: more than a third cannot be confirmed)': bool(share_cannot > 1 / 3),
           'controls': controls}
    print(f"  cannot_confirm {share_cannot:.1%} of the sample; AA-e "
          f"{'FIRES' if share_cannot > 1 / 3 else 'does not fire'}", flush=True)

    # ---- the rates, both readings of cannot_confirm, bootstrapped over stints within cell
    rng = np.random.default_rng(SEED)
    idx_by = {k: np.flatnonzero((w.cell == k).values) for k in N}
    draws = [w.iloc[np.concatenate([rng.choice(v, size=len(v), replace=True)
                                    for v in idx_by.values() if len(v)])].reset_index(drop=True)
             for _ in range(REPS)]

    readings = {'cannot_confirm_not_an_error': lambda d: (d.verdict == 'had_left').values,
                'cannot_confirm_an_error': lambda d: d.verdict.isin(['had_left', 'cannot_confirm']).values}
    res: dict = {}
    for name, err in readings.items():
        block: dict = {}
        for mname, mcol in MEASURES.items():
            acol = ARTEFACT[mname]
            point = rate(w, weights, mcol, FLAGGED, err(w), acol)
            boot = [rate(d, weights, mcol, FLAGGED, err(d), acol) for d in draws]
            cpoint = rate(w, weights, mcol, [CONTROL], err(w), acol)
            cboot = [rate(d, weights, mcol, [CONTROL], err(d), acol) for d in draws]
            diff = [a - b for a, b in zip(cboot, boot)]
            by = {}
            for c in N:
                pick = (w.cell == c).values
                if mcol is None:
                    m = a = np.ones(int(pick.sum()))
                else:
                    m = w[mcol].values[pick].astype(float)
                    a = np.where(np.isnan(w[acol].values[pick].astype(float)), m,
                                 w[acol].values[pick].astype(float))
                e = err(w)[pick]
                by[c] = {'rate': round(float(a[e].sum() / m.sum()), 4) if m.sum() else None,
                         'n_stints': int(pick.sum())}
            block[mname] = {
                'flagged': round(point, 4),
                'flagged_ci': [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)],
                'control': round(cpoint, 4),
                'control_ci': [round(float(np.percentile(cboot, q)), 4) for q in (2.5, 97.5)],
                'control_minus_flagged': round(cpoint - point, 4),
                'control_minus_flagged_ci': [round(float(np.percentile(diff, q)), 4) for q in (2.5, 97.5)],
                'by_cell': by}
        # the quantity the paper cares about: the share of the readable absence value that is an artefact
        def _art(d):
            v = np.where(np.isnan(d.art_value_readable.values.astype(float)),
                         d.abs_value_readable.values.astype(float),
                         d.art_value_readable.values.astype(float))
            return sum(weights[c] * float(v[((d.cell == c) & err(d)).values].sum()) for c in FLAGGED)

        art = _art(w)
        art_boot = [_art(d) for d in draws]
        block['artefact_share_of_readable_absence_value'] = {
            'point': round(art / den_readable, 4),
            'ci': [round(float(np.percentile(art_boot, q)) / den_readable, 4) for q in (2.5, 97.5)],
            'absence_value': round(art, 2), 'readable_denominator': round(den_readable, 2),
            'note': 'The flagged stints carry '
                    f"{meta['population']['share_of_readable']:.1%} of this denominator; this is the part "
                    'of it the audit judges to be a roster artefact.'}
        res[name] = block
    out['rates'] = res

    for name, block in res.items():
        print(f'\n  {name}:')
        for mname in MEASURES:
            b = block[mname]
            print(f'    by {mname:<7} flagged {b["flagged"]:.3f} {b["flagged_ci"]}, '
                  f'control {b["control"]:.3f} {b["control_ci"]}, '
                  f'difference {b["control_minus_flagged"]:.3f} {b["control_minus_flagged_ci"]}')
            print('      by cell: ' + ', '.join(
                f'{k.split("_")[0]} {v["rate"]} (n={v["n_stints"]})' for k, v in b['by_cell'].items()))
        a = block['artefact_share_of_readable_absence_value']
        print(f'    artefact share of the readable absence value {a["point"]:.4f} {a["ci"]}')

    # ---- section 5.4: the corrected build, as a new result beside the old (AA-f)
    out['corrected_build'] = corrected_build(w)

    # the control stratum found no departure, so its bootstrap interval is [0, 0] by construction; the
    # quantity a reader needs is the one-sided limit that 0 of n supports
    k_ctrl = int(((w.cell == CONTROL) & (w.verdict == 'had_left')).sum())
    n_ctrl = int((w.cell == CONTROL).sum())
    out['control_stratum_binomial'] = {
        'had_left': k_ctrl, 'of': n_ctrl,
        'one_sided_95_upper_on_the_stint_rate': round(clopper_pearson_upper(k_ctrl, n_ctrl), 4),
        'note': 'A bootstrap of a stratum with no errors resamples no errors, so the [0, 0] interval '
                'above is an artefact of the resampling and not a statement that the unflagged rate is '
                'zero. This limit is for the stint rate under independent sampling; it is not the '
                'value-weighted quantity the paper reports, and it is the reason the audit is not read as '
                'an upper bound on the error in the unflagged population.'}
    print(f"\n  control stratum: {k_ctrl} of {n_ctrl} had left; one-sided 95% upper on the stint rate "
          f"{clopper_pearson_upper(k_ctrl, n_ctrl):.3f}")

    key = res['cannot_confirm_not_an_error']['value']
    lo, hi = key['control_minus_flagged_ci']
    out['AA-d (the proxy does not select errors: the control is not separable)'] = bool(lo <= 0 <= hi)
    print(f"\n  AA-d: control minus flagged by value {key['control_minus_flagged']} "
          f"{key['control_minus_flagged_ci']} -> "
          f"{'FIRES, the proxy does not select errors' if lo <= 0 <= hi else 'does not fire'}")

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'\n-> {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
