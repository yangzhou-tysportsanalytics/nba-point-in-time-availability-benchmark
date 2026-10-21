"""Plan AA as it was run: drawing the sample for the human roster audit of the roster rule.

Published unchanged below its original header so the sampling rule can be read line by line. It reads the
originating project's paths and is not rerun here: the sample it draws is a player-level table, 150 named
players with a team, a season and the absence value each carries, and the source terms forbid
redistributing one. `results/paper2_roster_audit_sample.json` is the run's own output and carries no player
rows --- the population counts, the strata, the seed and the share of the absence value at stake.

What the sampling rule has to get right, and what a reader can check in it without the table: the unit is a
stint, one (player, team, season), because a departure is one fact that every remaining game inherits; the
proxy that selects the flagged stints reads the rest of the season and so may never become a roster rule,
only a sampling device; the control stratum is drawn from stints the proxy does *not* flag, which is what
makes a non-zero rate among the flagged ones readable at all; and the worksheet carries no indication of
what any answer is expected to be.

Import names: the originating project's layout is `src/roster_precision.py` (shipped here as
`src/roster_precision.py`), `league/scripts/p6_ds_roster_precision.py` (here
`league/scripts/roster_precision_main_build.py`) and `league/scripts/p6_ds_roster_audit_sample.py` (here
`league/scripts/roster_audit_sample_main_build.py`). The import and `importlib` paths below are the
project's, not this repository's, which is one reason these three are published to be read rather than
run.

The original header follows.

Plan AA (docs/plans/paper2_AA_roster_audit.md): draw the sample for the human roster audit.

  python league/scripts/p6_ds_roster_audit_sample.py
      -> docs/paper/audit/roster_audit_worksheet.csv
         docs/paper/audit/roster_audit_README.md
         results/paper2_roster_audit_sample.json

The question the existing audits do not answer is whether a player counted absent by the participation
feature was still on that team at the cutoff. Recall above 99.9% measures the opposite error, and the proxy
for this one -- a reference-rotation player who never appears for that team again in the season -- cannot
tell a departure from a season-ending injury. Only a person reading the transaction record can.

The unit is a stint, one (player, team, season), not a row: a player who has left carries one error that
every remaining game of the season inherits, so judging rows would spend the auditor's time re-judging the
same fact. The verdict is mapped back onto every row of the stint when the rates are computed, which is what
makes a value-weighted rate possible.

The proxy may not become a roster rule: it reads the rest of the season, so a build using it would not be
point-in-time. It is a sampling device here and a post-hoc sensitivity, nothing else.

Everything in the sampling rule is fixed in the plan: the four strata, their sizes, the seed, and the fields
the auditor fills. The analysis that reads the returned worksheet is fixed there too, before any row is
judged.
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
from src.roster_precision import last_appearance, never_appears_again       # noqa: E402

PLAN = ROOT / 'docs/plans/paper2_AA_roster_audit.md'
PLAN_SHA = 'b2accd0a0fb42060'
_sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
if not _sha.startswith(PLAN_SHA):
    raise SystemExit(f'the plan has changed since the rules were fixed: {_sha[:16]} != {PLAN_SHA}')

EXT = ROOT / 'data/league/paper2_ext'
DATA = EXT / 'rebuild/data'
AUD = ROOT / 'docs/paper/audit'
OUT_CSV = AUD / 'roster_audit_worksheet.csv'
OUT_MD = AUD / 'roster_audit_README.md'
OUT_JSON = ROOT / 'results/paper2_roster_audit_sample.json'

SEED = 20261020
NEAR_DAYS = 7
HAD_CHANCES = 10
SIZES = {'A_high_value': 50, 'B_near_transaction': 40, 'C_flagged_random': 30, 'D_control': 30}


def population():
    """The rotation rows, the flagged stints, the control stints and the readable denominator.

    Split out of `main` so that the script reading the returned worksheet can reweight the sample to the
    population by this construction rather than a second copy of it. Nothing here draws or writes; `main`
    does both, and re-running it reproduces the worksheet because the seed is fixed in the plan.
    """
    S_ = dict(dtype={'game_id': str, 'espn_player_id': str})
    R = pd.read_csv(EXT / 'bench_ref_players.csv', **S_)
    G = pd.read_csv(EXT / 'bench_games.csv', dtype={'game_id': str})
    games = pd.read_csv(DATA / 'games.csv', dtype={'game_id': str})
    games['start'] = pd.to_datetime(games.start_utc, utc=True)
    box = pd.read_csv(DATA / 'player_games.csv', **S_).merge(
        games[['game_id', 'season_year', 'start']], on='game_id', how='left')

    R = R.merge(games[['game_id', 'start']], on='game_id', how='left')
    R = R[R.game_id.isin(set(G.game_id[~G.any_block_missing]))].reset_index(drop=True)
    R['out_real'] = R.out_real.astype(bool)
    R['value'] = R.base_minutes * R.eff_prior

    # the proxy and the boundary rule, as results/paper2_roster_precision.json computes them
    R['never_again'] = never_appears_again(R, last_appearance(box))
    _tg = (games[games.game_id.isin(set(G.game_id))][['game_id', 'season_year', 'start']]
           .merge(G[['game_id', 'home', 'away']], on='game_id'))
    _long = pd.concat([_tg.assign(team=_tg.home), _tg.assign(team=_tg.away)]).sort_values('start')
    _long['remaining'] = _long.groupby(['season_year', 'team']).cumcount(ascending=False)
    _rem = dict(zip(zip(_long.game_id, _long.team), _long.remaining))
    R['remaining_games'] = [_rem.get((g, t), 0) for g, t in zip(R.game_id, R.team)]
    R['flagged'] = R.never_again.values & (R.remaining_games >= HAD_CHANCES).values

    # transactions involving the team, within a week, as the same result file defines them
    import importlib.util
    spec = importlib.util.spec_from_file_location('_rp', ROOT / 'league/scripts/p6_ds_roster_precision.py')
    near = None
    try:                                            # reuse the movement log through the existing script
        _m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_m)                 # it writes its own result file; harmless and idempotent
        near = None
    except SystemExit:
        pass
    _rp = json.loads((ROOT / 'results/paper2_roster_precision.json').read_text(encoding='utf-8'))
    # the near-transaction flag is not stored per row, so it is recomputed here from the same log
    mv_path = Path(_rp['removal_rule']['movement_log'])
    if not mv_path.exists():
        raise SystemExit(f'the movement log is not at {mv_path}; nothing written')
    mv = pd.read_csv(mv_path, dtype=str, encoding='utf-8-sig')
    mv['d'] = pd.to_datetime(mv.transaction_date, errors='coerce', utc=True)
    sys.path.insert(0, str(EXT / 'rebuild/code'))
    from league.src.build import team as team_code                           # noqa: E402
    tx = [(team_code(tid), when)
          for c in ('destination_team_id', 'origin_team_id') if c in mv.columns
          for tid, when in zip(mv.dropna(subset=[c, 'd'])[c], mv.dropna(subset=[c, 'd']).d)]
    by_team: dict[str, np.ndarray] = {}
    for t, w in tx:
        if t:
            by_team.setdefault(t, []).append(np.datetime64(w))
    by_team = {t: np.sort(np.array(v)) for t, v in by_team.items()}
    win = np.timedelta64(NEAR_DAYS, 'D')

    def near_tx(team, when) -> bool:
        ts = by_team.get(team)
        if ts is None or not len(ts):
            return False
        w = np.datetime64(when)
        i = np.searchsorted(ts, w)
        return any(0 <= j < len(ts) and abs(ts[j] - w) <= win for j in (i - 1, i))

    # ---------------------------------------------------------------- stints
    name = (box.dropna(subset=['player_name'])
               .drop_duplicates('espn_player_id').set_index('espn_player_id').player_name)
    key = ['espn_player_id', 'team', 'season_year']
    F = R[R.flagged]
    stint = F.groupby(key).agg(flagged_rows=('game_id', 'size'),
                               first_flagged=('start', 'min'), last_flagged=('start', 'max'),
                               value_flagged=('value', 'sum')).reset_index()
    _abs = R[R.out_real]
    allr = R.groupby(key).agg(rows=('game_id', 'size'),
                              absent_rows=('out_real', 'sum'),
                              season_value=('value', 'sum')).reset_index()
    # the period the auditor is being asked about, and the value at stake, for every stint and not only the
    # flagged ones: a control row with no dates and no value is a question with nothing attached to it
    _ar = _abs.groupby(key).agg(first_absent=('start', 'min'), last_absent=('start', 'max'),
                                absence_value=('value', 'sum')).reset_index()
    allr = allr.merge(_ar, on=key, how='left')
    stint = stint.merge(allr, on=key, how='left')
    stint['near_transaction'] = [near_tx(t, w) for t, w in zip(stint.team, stint.first_flagged)]

    _readable = (R.remaining_games >= HAD_CHANCES).values & R.out_real.values
    _den = float(R.value[_readable].sum())

    ctrl = allr[~allr.set_index(key).index.isin(stint.set_index(key).index)]
    ctrl = ctrl[ctrl.absent_rows > 0].copy()
    return R, stint, ctrl, _den, name


def main() -> int:
    R, stint, ctrl, _den, name = population()

    rng = np.random.default_rng(SEED)
    take = lambda d, n: d.iloc[rng.choice(len(d), size=min(n, len(d)), replace=False)] if len(d) else d
    cut = stint.absence_value.quantile(0.9)   # absence_value comes from the merge with allr above
    A = take(stint[stint.absence_value >= cut], SIZES['A_high_value'])
    rest = stint[~stint.index.isin(A.index)]
    B = take(rest[rest.near_transaction], SIZES['B_near_transaction'])
    rest = rest[~rest.index.isin(B.index)]
    C = take(rest, SIZES['C_flagged_random'])
    D = take(ctrl, SIZES['D_control'])

    rows = []
    for lab, d in (('A_high_value', A), ('B_near_transaction', B), ('C_flagged_random', C), ('D_control', D)):
        for _, r in d.iterrows():
            rows.append({
                'stint_id': f"{r.espn_player_id}_{r.team}_{int(r.season_year)}",
                'stratum': lab,
                'player': name.get(r.espn_player_id, ''),
                'espn_player_id': r.espn_player_id,
                'team': r.team,
                'season': int(r.season_year),
                'first_absent_game': str(r.get('first_absent', ''))[:10],
                'last_absent_game': str(r.get('last_absent', ''))[:10],
                'first_flagged_game': str(r.get('first_flagged', ''))[:10],
                'last_flagged_game': str(r.get('last_flagged', ''))[:10],
                'flagged_rows': int(r.get('flagged_rows', 0) or 0),
                'rows_in_stint': int(r.rows),
                'rows_counted_absent': int(r.absent_rows),
                'absence_value_carried': round(float(r.get('absence_value', 0.0) or 0.0), 2),
                'near_transaction': bool(r.get('near_transaction', False)),
                'verdict': '', 'left_date': '', 'source': '', 'note': ''})
    W = pd.DataFrame(rows)
    AUD.mkdir(parents=True, exist_ok=True)
    W.to_csv(OUT_CSV, index=False, encoding='utf-8')

    meta = {'plan': 'docs/plans/paper2_AA_roster_audit.md', 'plan_sha256': _sha, 'seed': SEED,
            'near_days': NEAR_DAYS, 'min_remaining_games': HAD_CHANCES,
            'population': {'rotation_rows': int(len(R)), 'flagged_rows': int(R.flagged.sum()),
                           'flagged_stints': int(len(stint)), 'control_stints': int(len(ctrl)),
                           'absence_value_total': round(float(R.value[R.out_real].sum()), 2),
                           'absence_value_readable': round(float(_den), 2),
                           'absence_value_flagged': round(float(stint.absence_value.sum()), 2),
                           'share_of_readable': round(float(stint.absence_value.sum() / _den), 4),
                           'share_of_all_rows': round(
                               float(stint.absence_value.sum() / R.value[R.out_real].sum()), 4),
                           'note': 'share_of_readable is the figure the paper quotes: the denominator is '
                                   'the absence value of rows where the proxy can be read, those with ten '
                                   'team-games still to come. share_of_all_rows divides by every row.'},
            'strata': {k: int((W.stratum == k).sum()) for k in SIZES},
            'top_decile_cut_absence_value': round(float(cut), 2),
            'worksheet': 'docs/paper/audit/roster_audit_worksheet.csv',
            'note': 'Stints, not rows: one judgement per (player, team, season), mapped back onto its rows '
                    'when the rates are computed. The proxy reads the rest of the season and is a sampling '
                    'device and a post-hoc sensitivity only, never a roster rule.'}
    OUT_JSON.write_text(json.dumps(meta, indent=1), encoding='utf-8')

    share = meta['population']['share_of_readable']
    OUT_MD.write_text(f"""# Roster audit worksheet

Plan: `docs/plans/paper2_AA_roster_audit.md` (SHA-256 `{_sha[:16]}`). Sample drawn at seed {SEED}.

## The question

For each row below: **was this player still on this team at the time of these games?**

The participation feature counts him as absent. If he had already left the team, that absence is an artefact
of the roster rule rather than a fact about the game, and it inflates the quantity this paper measures. The
flagged stints carry {share:.1%} of the absence value in that feature among the rows where the proxy can be
read, which is the figure the paper quotes, and {meta['population']['share_of_all_rows']:.1%} of all of it.
Either way the answer matters at the size of the paper's own result.

## What to fill, per row

| column | what to put |
|---|---|
| `verdict` | one of `still_on_team`, `had_left`, `other`, `cannot_confirm` |
| `left_date` | for `had_left` only: the date he ceased to be on that team, `YYYY-MM-DD` |
| `source` | a URL or citation for the judgement |
| `note` | free text; **required** for `other` and `cannot_confirm` |

`cannot_confirm` is a real answer. Please do not record it as `still_on_team`: the analysis counts the
unconfirmable both ways and reports the range, and that only works if they are marked.

`other` is for cases that are neither: a two-way contract moving him to the G League, a suspension, a
player who returned later under a new contract, and anything else you can describe.

## What not to do

Do not use the fact that a player never appears again as evidence that he left. That is the proxy this
audit exists to test, and using it would make the result circular. Judge from the transaction record.

## The strata, and why they are not equally interesting

| stratum | what it is |
|---|---|
| `A_high_value` | flagged stints carrying the most absence value; an error here costs the most |
| `B_near_transaction` | flagged stints whose first flagged game is within {NEAR_DAYS} days of a transaction involving the team |
| `C_flagged_random` | the remaining flagged stints, at random |
| `D_control` | **not flagged**: the player does appear again. These should almost all be `still_on_team`, and they are what makes the others readable |

Stratum D is not filler. If its error rate turns out to be the same as the flagged strata's, the proxy does
not select errors at all, and the paper has to say so.

## Counts

{len(W)} stints: {', '.join(f'{k} {v}' for k, v in meta['strata'].items())}.
They stand for {int(W.flagged_rows.sum()):,} flagged player-games out of {meta['population']['flagged_rows']:,} in the benchmark.
""", encoding='utf-8')

    print(f'  flagged rows {meta["population"]["flagged_rows"]:,} in {meta["population"]["flagged_stints"]:,} '
          f'stints; control stints {meta["population"]["control_stints"]:,}')
    print(f"  flagged share of the participation feature's absence value: {share:.1%} of the rows "
          f"where the proxy can be read, {meta['population']['share_of_all_rows']:.1%} of every row")
    print(f'  worksheet: {len(W)} stints  ' + ', '.join(f'{k} {v}' for k, v in meta['strata'].items()))
    print(f'\n-> {OUT_CSV.relative_to(ROOT)}\n-> {OUT_MD.relative_to(ROOT)}\n-> {OUT_JSON.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
