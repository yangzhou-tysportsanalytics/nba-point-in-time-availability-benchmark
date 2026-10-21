"""Plan AA section 5.4 as it was run: the strict headline and the never-listed put-back on the corrected build.

Published unchanged below its original header. It reads the originating project's paths --- the whole report
series and the returned worksheet, which is a player-level table this repository does not hold --- so it is
not rerun here; its output is `results/paper2_roster_audit_reruns.json`, which carries the two controls and
the two comparisons and no player rows.

Both controls reproduce the published figures exactly: the uncorrected strict headline at 0.1852 and 0.0502
on 4,131 games, and the uncorrected never-listed put-back at -0.0332. The never-listed component is where
428 of the 429 corrected absence rows sat, which is where a roster artefact should sit, and the put-back
still moves by 0.0004.

Import names: the originating project's layout is `league/scripts/p6_ds_roster_audit_reruns.py`, and the
module it imports is `src/boot_designs.py`, shipped here under that name.

The original header follows.

Plan AA section 5.4, the two reruns the first result script did not do.

  python league/scripts/p6_ds_roster_audit_reruns.py -> results/paper2_roster_audit_reruns.json

Section 5.4 asks for the pooled headline, the strict headline and the never-listed put-back on the
corrected build. The first was done in `p6_ds_roster_audit_result.py`, which has the game-level table it
needs; the other two re-read the report series --- the strict sample is the games where both teams had
filed six hours out, and the never-listed component is the absences of rotation players who were on no
report at all thirty minutes before tip --- so they are here, built the way
`p6_ds_cutoff_curve.py` and `p6_ds_mechanism4.py` build them rather than from a second definition.

The never-listed put-back is the one of the three that bears on the roster rule most directly: a player
kept on a list after he left is absent, is on no report, and lands in exactly that component. If correcting
the confirmed departures moves anything, it should move this.

Controls first, and nothing is interpreted if either fails: the uncorrected strict headline must reproduce
the published 0.1852 and 0.0502 on 4,131 games, and the uncorrected put-back must reproduce the published
-0.0332. Both are read off the committed result files before this script existed.

The correction is plan AA's: the absence rows of a stint judged `had_left`, from the day the departure took
effect, leave the absence set. F1 is untouched, because a player the report called Out is counted by the
report and not by hindsight, and the plan corrects the participation feature.
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
from src.boot_designs import cluster_keys as _cluster_keys, weight_bank as _weight_bank  # noqa: E402

PLAN = ROOT / 'docs/plans/paper2_AA_roster_audit.md'
PLAN_SHA = 'b2accd0a0fb42060'
_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
if not _sha.startswith(PLAN_SHA):
    raise SystemExit(f'the plan has changed since the rules were fixed: {_sha[:16]} != {PLAN_SHA}')

EXT = ROOT / 'data/league/paper2_ext'
DATA = EXT / 'rebuild/data'
FS = EXT / 'full_series'
FILLED = ROOT / 'docs/paper/audit/roster_audit_filled.csv'
OUT = ROOT / 'results/paper2_roster_audit_reruns.json'
SEED, REPS = 20260921, 2000
BASE = ['strength_diff', 'rest_diff']
CONTROL = {'strict_gain': 0.1852, 'strict_infl': 0.0502, 'strict_n': 4131, 'putback_never': -0.0332}

# ---------------------------------------------------------------- inputs, as mechanism4 loads them
print('loading the report series', flush=True)
G = pd.read_csv(EXT / 'bench_games.csv', dtype={'game_id': str})
R = pd.read_csv(EXT / 'bench_ref_players.csv', dtype={'game_id': str, 'espn_player_id': str})
games = pd.read_csv(DATA / 'games.csv', dtype={'game_id': str})
games['start'] = pd.to_datetime(games.start_utc, utc=True)
rows = pd.read_csv(FS / 'injury_rows.csv', dtype={'game_id': str})
blocks = pd.read_csv(FS / 'injury_blocks.csv', dtype={'game_id': str})
rows['ts'] = pd.to_datetime(rows.report_timestamp_et, utc=True)
blocks['ts'] = pd.to_datetime(blocks.report_timestamp_et, utc=True)
st = pd.concat([pd.read_csv(DATA / f, dtype={'game_id': str, 'espn_player_id': str})
                for f in ('pregame_status.csv', 'pregame_status_90m.csv')])
st = st.dropna(subset=['espn_player_id']).merge(games[['game_id', 'season_year']], on='game_id')
name_id = (st.drop_duplicates(['team', 'season_year', 'player_name'])
             .set_index(['team', 'season_year', 'player_name']).espn_player_id)
rows = rows.merge(games[['game_id', 'season_year', 'start']], on='game_id')
rows['pid'] = [name_id.get((t, s, n)) for t, s, n in zip(rows.team, rows.season_year, rows.player_name)]
blocks = blocks.merge(games[['game_id', 'start']], on='game_id')
listed = blocks[blocks.state == 'listed']
R = R.merge(games[['game_id', 'start']], on='game_id', how='left')
R['val'] = (R.base_minutes * R.eff_prior).values
R['out_real'] = R.out_real.astype(bool)
R['out_pre'] = R.out_pre.astype(bool)
print(f'  {len(rows):,} report rows, {len(R):,} reference player-games', flush=True)


def selected(hours: float):
    b = listed[listed.ts <= listed.start - pd.Timedelta(hours=hours)]
    best = b.sort_values('ts').groupby(['game_id', 'team']).ts.max()
    rr = rows.copy()
    rr['sel'] = [best.get((g, t)) for g, t in zip(rr.game_id, rr.team)]
    on = rr[(rr.ts == rr.sel) & rr.pid.notna()]
    o = on[on.availability_status == 'Out']
    return best, set(zip(o.game_id, o.team, o.pid)), set(zip(on.game_id, on.team, on.pid))


def team_value(keep: set) -> pd.Series:
    flag = np.array([k in keep for k in zip(R.game_id, R.team, R.espn_player_id)])
    return pd.Series(np.where(flag, R.val, 0.0)).groupby([R.game_id, R.team]).sum()


def gdiff(v: pd.Series) -> np.ndarray:
    d = dict(zip(v.index, v.values))
    return np.array([d.get((g, h), 0.0) - d.get((g, a), 0.0)
                     for g, h, a in zip(G.game_id, G.home, G.away)])


best6, out6, _ = selected(6)
_, out30, listed30 = selected(0.5)
have6 = set(best6.index)
G['both_filed_6h'] = [((g, h) in have6) and ((g, a) in have6)
                      for g, h, a in zip(G.game_id, G.home, G.away)]

# ---------------------------------------------------------------- the correction, and the two absence sets
keys = list(zip(R.game_id, R.team, R.espn_player_id))
f1_out = {k for k, o in zip(keys, R.out_pre.values) if o}
sat = {k for k, o in zip(keys, R.out_real.values) if o}

w = pd.read_csv(FILLED, dtype={'espn_player_id': str})
left = {r.stint_id: pd.Timestamp(r.left_date).tz_localize('UTC')
        for r in w.itertuples() if r.verdict == 'had_left'}
sid = [f'{p}_{t}_{int(s)}' for p, t, s in zip(R.espn_player_id, R.team, R.season_year)]
gone = np.array([s in left and ts >= left[s] for s, ts in zip(sid, R.start)])
removed = {k for k, g_, o in zip(keys, gone, R.out_real.values) if g_ and o}
sat_c = sat - removed
print(f'  the correction removes {len(removed):,} absences of {len(sat):,}', flush=True)


def never_listed_of(absences: set) -> set:
    return {k for k in absences if k not in out30 and k not in listed30 and k not in f1_out}


def f2_of(absences: set) -> np.ndarray:
    return gdiff(team_value(absences))


G['F2_rebuilt'] = f2_of(sat)
G['F2_corrected'] = f2_of(sat_c)
G['never_diff'] = f2_of(never_listed_of(sat))
G['never_diff_corrected'] = f2_of(never_listed_of(sat_c))
G['F1pc3_diff'] = G.F1_diff + G.never_diff
G['F1pc3_corrected'] = G.F1_diff + G.never_diff_corrected
worst_f2 = float((G.F2_rebuilt - G.F2_diff).abs().max())
print(f'  rebuilt participation feature matches the shipped one to {worst_f2:.2e}', flush=True)
print(f'  never-listed absences: {len(never_listed_of(sat)):,} before the correction, '
      f'{len(never_listed_of(sat_c)):,} after', flush=True)

X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])


def fit(sample: pd.DataFrame, cols: dict[str, list[str]]):
    parts = []
    for s in (2023, 2024, 2025, 2026):
        tr, ho = sample[sample.season_year < s], sample[sample.season_year == s].copy()
        for k, c in cols.items():
            ho['p_' + k] = X(ho, c) @ np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0]
        parts.append(ho)
    H = pd.concat(parts, ignore_index=True)
    rng = np.random.default_rng(SEED)
    code, kk = pd.factorize(pd.Series((H.home + '_' + H.season_year.astype(str)).values))
    W = [np.bincount(rng.integers(0, len(kk), len(kk)), minlength=len(kk)).astype(float)[code]
         for _ in range(REPS)]
    return H, W


def _rmse(y, p, w_):
    return float(np.sqrt(np.sum(w_ * (y - p) ** 2) / w_.sum()))


def headline(sample: pd.DataFrame, f2: str) -> dict:
    H, W = fit(sample, {'F0': [], 'F1': ['F1_diff'], 'F2': [f2]})
    y = H.margin.values

    def st(w_):
        r = {k: _rmse(y, H['p_' + k].values, w_) for k in ('F0', 'F1', 'F2')}
        return r['F0'] - r['F1'], r['F1'] - r['F2']

    pt = st(np.ones(len(y)))
    B = [st(w_) for w_ in W]
    return {'n': int(len(y)), 'gain': round(pt[0], 4), 'inflation': round(pt[1], 4),
            'gain_ci': [round(float(np.percentile([b[0] for b in B], q)), 4) for q in (2.5, 97.5)],
            'inflation_ci': [round(float(np.percentile([b[1] for b in B], q)), 4) for q in (2.5, 97.5)]}


def putback(sample: pd.DataFrame, col: str) -> dict:
    """RMSE(F1) - RMSE(F1 + never-listed), the contrast the paper prints as the never-listed put-back."""
    H, W = fit(sample, {'F1': ['F1_diff'], 'P': [col]})
    y = H.margin.values

    def st(w_):
        return _rmse(y, H['p_F1'].values, w_) - _rmse(y, H['p_P'].values, w_)

    B = [st(w_) for w_ in W]
    return {'n': int(len(y)), 'point': round(st(np.ones(len(y))), 4),
            'ci95_home': [round(float(np.percentile(B, q)), 4) for q in (2.5, 97.5)]}


def main() -> int:
    strict = G[G.both_filed_6h & ~G.any_block_missing].reset_index(drop=True)
    fallback = G[~G.any_block_missing].reset_index(drop=True)

    s_pub = headline(strict, 'F2_rebuilt')
    p_pub = putback(fallback, 'F1pc3_diff')
    ok_s = (s_pub['n'] == CONTROL['strict_n']
            and abs(s_pub['gain'] - CONTROL['strict_gain']) < 5e-4
            and abs(s_pub['inflation'] - CONTROL['strict_infl']) < 5e-4)
    ok_p = abs(p_pub['point'] - CONTROL['putback_never']) < 5e-4
    out = {'plan': 'docs/plans/paper2_AA_roster_audit.md', 'plan_sha256': _sha, 'seed': SEED, 'reps': REPS,
           'note': 'Plan AA section 5.4, the two reruns that re-read the report series. The correction is '
                   'of the confirmed departures only, so these are the confirmed part of the error and '
                   'not an estimate of all of it.',
           'correction': {'stints': len(left), 'absences_removed': len(removed),
                          'absences_total': len(sat),
                          'never_listed_before': len(never_listed_of(sat)),
                          'never_listed_after': len(never_listed_of(sat_c))},
           'controls': {'strict_rebuilt': s_pub, 'putback_rebuilt': p_pub, 'expected': CONTROL,
                        'participation_feature_worst_difference': worst_f2,
                        'verdict': 'pass' if (ok_s and ok_p and worst_f2 < 1e-6) else 'FAIL'}}
    print(f"\n  control strict  {s_pub['gain']}/{s_pub['inflation']} on {s_pub['n']} against "
          f"{CONTROL['strict_gain']}/{CONTROL['strict_infl']} on {CONTROL['strict_n']}")
    print(f"  control putback {p_pub['point']} against {CONTROL['putback_never']}")
    if out['controls']['verdict'] == 'FAIL':
        OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
        raise SystemExit('the controls do not reproduce the published figures; nothing is interpreted')

    s_cor = headline(strict, 'F2_corrected')
    p_cor = putback(fallback, 'F1pc3_corrected')
    out['strict_headline'] = {'published': s_pub, 'corrected': s_cor,
                              'premium_moves_by': round(abs(s_cor['inflation'] - s_pub['inflation']), 4),
                              'gain_moves_by': round(abs(s_cor['gain'] - s_pub['gain']), 4)}
    out['never_listed_putback'] = {'published': p_pub, 'corrected': p_cor,
                                   'moves_by': round(abs(p_cor['point'] - p_pub['point']), 4)}
    print(f"\n  strict headline  {s_pub['gain']}/{s_pub['inflation']} -> "
          f"{s_cor['gain']}/{s_cor['inflation']} {s_cor['inflation_ci']}  "
          f"(premium moves {out['strict_headline']['premium_moves_by']})")
    print(f"  never-listed put-back  {p_pub['point']} {p_pub['ci95_home']} -> "
          f"{p_cor['point']} {p_cor['ci95_home']}  (moves {out['never_listed_putback']['moves_by']})")

    OUT.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(f'\n-> {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
