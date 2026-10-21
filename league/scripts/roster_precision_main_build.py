"""Plan V item V4 as it was run: the roster rule's removal side, and the proxy for the error it can make.

Published unchanged below its original header. It reads the originating project's paths and the league's
player-movement feed as that project fetched it, so it is not rerun here; its output is
`results/paper2_roster_precision.json`, which carries counts and rates and no player rows, with the feed
named by its source rather than by a path on the authors' machine. Plan AA's sampler reads this script to
reuse its proxy and its removal rule rather than restating them.

The original header follows.

Paper 2, plan V item V4: whether the roster keeps players who have left the team.

Section 2.2 reports roster recall of over 99.9% against the players who actually appeared. That measures the
error of leaving a player out and is blind to the opposite one. A player kept on a team's candidate list
after he has left it carries reference minutes, counts as absent in the realized-participation feature, and
can land in the never-listed component of Section 4.7 without ever appearing on an Out list -- which is the
quantity this paper measures.

The removal rule, read out of the build's own code (rebuild/code/league/src/build.py, `team_at` and
`roster`): a player is on a team's list at a cutoff when the last event at or before that cutoff puts him
there. Events are box-score appearances for a team, `join` events from the league's movement log, and
`leave` events, which `movement_events` emits for the transaction type `Waive` and for no other. So a player
leaves a list when he is waived, when he appears or joins elsewhere, or when 365 days pass with no event.
Two gaps follow from that and are reported here rather than inferred: a transaction with no destination team
that is not a waiver emits nothing at all, and a player whose name cannot be matched to an identifier has no
movement events of any kind.

A false active cannot be observed directly, so the proxy is a reference-rotation player who never appears
for that team again in the rest of the season. That is not an error on its own -- a player can miss the rest
of a season through injury -- so what rule V-e is written on is the comparison between team-games near a
transaction and team-games away from one.

  python league/scripts/p6_ds_roster_precision.py  -> results/paper2_roster_precision.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.roster_precision import false_active_rate, last_appearance, never_appears_again  # noqa: E402

EXT = ROOT / 'data/league/paper2_ext'
DATA = EXT / 'rebuild/data'
RAW = EXT / 'rebuild/raw'
PLAN = ROOT / 'docs/plans/paper2_V_server_time_and_strict.md'
PLAN_SHA = 'eb870b9cd8ec4296'
OUT = ROOT / 'results/paper2_roster_precision.json'
NEAR_DAYS = 7           # plan V: team-games within a week of a transaction affecting either team
RATIO_TRIGGER = 2.0     # plan V rule V-e

sha = hashlib.sha256(PLAN.read_bytes()).hexdigest()
if not sha.startswith(PLAN_SHA):
    raise SystemExit(f'the plan has changed since the rules were fixed: {sha[:16]} != {PLAN_SHA}')

S_ = dict(dtype={'game_id': str, 'espn_player_id': str})
R = pd.read_csv(EXT / 'bench_ref_players.csv', **S_)
G = pd.read_csv(EXT / 'bench_games.csv', dtype={'game_id': str})
games = pd.read_csv(DATA / 'games.csv', dtype={'game_id': str})
games['start'] = pd.to_datetime(games.start_utc, utc=True)
box = pd.read_csv(DATA / 'player_games.csv', **S_).merge(
    games[['game_id', 'season_year', 'start']], on='game_id', how='left')

R = R.merge(games[['game_id', 'start']], on='game_id', how='left')
R = R[R.game_id.isin(set(G.game_id[~G.any_block_missing]))].reset_index(drop=True)

# ---------------------------------------------------------------- the removal rule, as implemented
code = (EXT / 'rebuild/code/league/src/build.py').read_text(encoding='utf-8')
rule = {
    'source': 'data/league/paper2_ext/rebuild/code/league/src/build.py',
    'membership': "team_at(pid, cutoff) returns the team of the last event at or before the cutoff, None if "
                  "that event is a 'leave' or is more than HIST_DAYS old; roster() keeps a candidate when "
                  "team_at equals the team, or when the report lists him and team_at is None",
    'history_window_days': 365 if 'HIST_DAYS = pd.Timedelta(days=365)' in code else None,
    'leave_events_emitted_for': ['Waive'] if "transaction_type == 'Waive'" in code else [],
    'join_events_emitted_for': 'any transaction with a destination team',
    'gaps': ["a transaction that is neither a waiver nor has a destination team emits no event, so the "
             "player stays on his previous team's list until an appearance elsewhere or 365 days pass",
             "a player whose name cannot be matched to an identifier is skipped, so none of his movements "
             "register at all",
             "a transaction is stamped at 12:00Z on its date, so one announced later in the day is treated "
             "as known at a cutoff six hours before an evening tip"],
}

# The movement log lives outside this repository: the rebuild snapshot's path module points at a Paper_A
# checkpoint it does not carry. Resolve through the snapshot, then fall back, and record which was used,
# because a log that cannot be redistributed is part of what the released path cannot reproduce.
_mv_candidates = []
try:
    sys.path.insert(0, str(EXT / 'rebuild/code'))
    from league.src.paths import RAW as _SNAP_RAW                              # noqa: E402
    _mv_candidates.append(_SNAP_RAW / 'nba_player_movements_raw.csv')
except Exception:
    pass
_mv_candidates += [RAW / 'nba_player_movements_raw.csv',
                   Path('D:/Paper_A/data/checkpoints/phase1_2019-07-01_2026-08-28/v2_source/raw/'
                        'nba_player_movements_raw.csv'),
                   Path('D:/Paper_A/data/raw/nba_player_movements_raw.csv')]
_mv_path = next((p for p in _mv_candidates if p.exists()), None)
if _mv_path is None:
    raise SystemExit('the movement log was not found at any known location; nothing written')
mv = pd.read_csv(_mv_path, dtype=str, encoding='utf-8-sig')
mv['d'] = pd.to_datetime(mv.transaction_date, errors='coerce', utc=True)
rule['movement_log'] = str(_mv_path)
rule['movement_log_inside_this_repository'] = str(_mv_path).startswith(str(ROOT))
rule['movement_rows'] = int(len(mv))
rule['movement_types'] = mv.transaction_type.value_counts().to_dict()

# ---------------------------------------------------------------- the proxy
last = last_appearance(box)
R['never_again'] = never_appears_again(R, last)

# team-games within NEAR_DAYS of a transaction whose destination or origin is that team
tcols = [c for c in ('destination_team_id', 'origin_team_id') if c in mv.columns]
sys.path.insert(0, str(EXT / 'rebuild/code'))
try:
    from league.src.build import team as team_code
except Exception:
    team_code = None
tx = []
for c in tcols:
    sub = mv.dropna(subset=[c, 'd'])
    for tid, when in zip(sub[c], sub.d):
        code_ = team_code(tid) if team_code else None
        if code_:
            tx.append((code_, when))
TX = pd.DataFrame(tx, columns=['team', 'when'])
by_team = {t: np.sort(d.when.values) for t, d in TX.groupby('team')}
win = np.timedelta64(NEAR_DAYS, 'D')


def near_transaction(team: str, when) -> bool:
    ts = by_team.get(team)
    if ts is None or not len(ts):
        return False
    w = np.datetime64(when)
    i = np.searchsorted(ts, w)
    for j in (i - 1, i):
        if 0 <= j < len(ts) and abs(ts[j] - w) <= win:
            return True
    return False


R['near_tx'] = [near_transaction(t, w) for t, w in zip(R.team, R.start)]

# A team's last game makes the proxy vacuous: nobody appears again after it. Count each team-game's
# remaining games in its own season and read the proxy where the player had chances left.
_tg = (games[games.game_id.isin(set(G.game_id))][['game_id', 'season_year', 'start']]
       .merge(G[['game_id', 'home', 'away']], on='game_id'))
_long = pd.concat([_tg.assign(team=_tg.home), _tg.assign(team=_tg.away)])
_long = _long.sort_values('start')
_long['remaining'] = _long.groupby(['season_year', 'team']).cumcount(ascending=False)
_rem = dict(zip(zip(_long.game_id, _long.team), _long.remaining))
R['remaining_games'] = [_rem.get((g, t), 0) for g, t in zip(R.game_id, R.team)]
_HAD_CHANCES = 10
_ok = (R.remaining_games >= _HAD_CHANCES).values

rates_all = false_active_rate(R.never_again.values, R.near_tx.values)
rates = false_active_rate(R.never_again.values[_ok], R.near_tx.values[_ok])

out = {'plan': 'docs/plans/paper2_V_server_time_and_strict.md', 'plan_sha256': sha,
       'removal_rule': rule,
       'note': 'The proxy is a reference-rotation player who never appears for that team again in the '
               'season. It is not an error on its own, so the quantity rule V-e reads is the ratio between '
               'team-games near a transaction and team-games away from one.',
       'rotation_rows': int(len(R)),
       'transaction_window_days': NEAR_DAYS,
       'teams_with_transactions': int(len(by_team)),
       'boundary': {'note': 'The proxy is vacuously true in a team\'s last game, so the rule reads the '
                            'rows with games still to come.',
                    'min_remaining_games': _HAD_CHANCES,
                    'rows_with_chances_left': int(_ok.sum())},
       'proxy': rates,
       'proxy_all_rows': rates_all,
       'overall_rate': round(float(R.never_again.mean()), 6),
       'overall_rate_with_chances_left': round(float(R.never_again.values[_ok].mean()), 6)}

# how much absence value and how many never-listed absences the flagged rows carry
R['val'] = R.base_minutes * R.eff_prior
flagged = R[R.never_again & _ok]
out['flagged'] = {
    'rows': int(len(flagged)),
    'share_of_rotation_rows': round(float(len(flagged) / int(_ok.sum())), 6),
    'share_absent_in_the_end': round(float(flagged.out_real.astype(bool).mean()), 4),
    'share_out_at_the_cutoff': round(float(flagged.out_pre.astype(bool).mean()), 4),
    'absence_value_share_of_F2': round(
        float(flagged.loc[flagged.out_real.astype(bool), 'val'].sum()
              / R.loc[_ok & R.out_real.astype(bool), 'val'].sum()), 6),
    'never_listed_rows': int(((~flagged.out_pre.astype(bool)) & flagged.out_real.astype(bool)).sum()),
}

# ---------------------------------------------------------------- rule V-e
ratio = rates['ratio']
fires = bool(np.isfinite(ratio) and ratio > RATIO_TRIGGER)
out['rule_verdicts'] = {
    f'V-e: the proxy near a transaction is no more than {RATIO_TRIGGER:g} times the baseline': bool(not fires)}
out['rule_V_e'] = (
    'The proxy is more than twice as common near a transaction, so the never-listed component carries this '
    'as a stated contamination and Section 4.7 says so.' if fires else
    'The proxy is not concentrated near transactions, so the removal rule is not visibly leaking; Table 1 '
    'states the rule and this is reported as a negative result.')

json.dump(out, open(OUT, 'w'), indent=1, default=str)

print('removal rule, as implemented:')
print(f"  leave events for: {rule['leave_events_emitted_for']}; history window "
      f"{rule['history_window_days']} days; {rule['movement_rows']:,} movement rows")
for g_ in rule['gaps']:
    print(f"  gap: {g_}")
print(f"\nproxy over {out['rotation_rows']:,} reference-rotation rows "
      f"({out['teams_with_transactions']} teams with transactions):")
print(f"  never appears again for the team: {100 * out['overall_rate']:.2f}% over all rows, "
      f"{100 * out['overall_rate_with_chances_left']:.2f}% where {_HAD_CHANCES}+ team-games remain")
print(f"  (the raw rate counts a team's last game, where the proxy is vacuous: ratio there "
      f"{rates_all['ratio']:.3f} on {rates_all['n_near_transaction'] + rates_all['n_away_from_transaction']:,} rows)")
print(f"  near a transaction ({rates['n_near_transaction']:,} rows): {100 * rates['rate_near']:.2f}%")
print(f"  away from one    ({rates['n_away_from_transaction']:,} rows): {100 * rates['rate_away']:.2f}%")
print(f"  ratio: {rates['ratio']:.3f}")
f = out['flagged']
print(f"\nthe {f['rows']:,} flagged rows are {100 * f['share_of_rotation_rows']:.2f}% of the rotation; "
      f"{100 * f['share_absent_in_the_end']:.1f}% were absent in the end, "
      f"{100 * f['share_out_at_the_cutoff']:.1f}% were Out at the cutoff")
print(f"  they carry {100 * f['absence_value_share_of_F2']:.2f}% of the realized-participation feature's "
      f"absence value, and {f['never_listed_rows']:,} never-listed absences")
print()
for k, v in out['rule_verdicts'].items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print(f"\n-> {OUT.relative_to(ROOT)}")
