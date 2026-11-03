"""Where the exposure concentrates: the fifty games whose late Out news differs most between the teams.

Published unchanged below its original header. It reads the originating project's paths, so
it is not rerun here; its output is `results/paper2_example_aggregate.json`, which carries group means and counts and no player rows.

The original header follows.

Paper 2, plan N amendment 1: the aggregate that goes beside the worked example.

Over the lined holdout games of 2025-26, take the 50 ranked highest on the absence value newly ruled Out between
the six-hour cutoff and thirty minutes, and compare them with the rest on: the gap between the compliant and the
leaky prediction, each model's absolute error, and how far the point spread moved from open to close.

The ranking quantity is the **absolute home-minus-away difference** of that value, not its total: two teams that
each lose the same value rank near zero, because the models read the difference. Amended 2026-11-03: the file now
carries two blocks. `all_priced` is the original 977 games. In 32 of its top 50 a team had not filed at six hours,
where the early Out set is empty and the team's first filing counts entirely as newly ruled Out -- which is first
disclosure, not a change after a report. `strict` restricts training and holdout to the games where both teams had
filed at six hours, the rule of Section 2.2, and is what the figure shows.

  python league/scripts/p6_ds_example_aggregate.py  -> results/paper2_example_aggregate.json
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / 'data/league/paper2_ext'; DATA = EXT / 'rebuild/data'; FS = EXT / 'full_series'
G = pd.read_csv(EXT / 'bench_games.csv', dtype={'game_id': str})
R = pd.read_csv(EXT / 'bench_ref_players.csv', dtype={'game_id': str, 'espn_player_id': str})
L = pd.read_csv(EXT / 'lines_open.csv', dtype={'game_id': str})[['game_id', 'open_spread_home', 'close_spread_home']]
games = pd.read_csv(DATA / 'games.csv', dtype={'game_id': str}); games['start'] = pd.to_datetime(games.start_utc, utc=True)
rows = pd.read_csv(FS / 'injury_rows.csv', dtype={'game_id': str}); blocks = pd.read_csv(FS / 'injury_blocks.csv', dtype={'game_id': str})
rows['ts'] = pd.to_datetime(rows.report_timestamp_et, utc=True); blocks['ts'] = pd.to_datetime(blocks.report_timestamp_et, utc=True)
st = pd.concat([pd.read_csv(DATA / f, dtype={'game_id': str, 'espn_player_id': str}) for f in ('pregame_status.csv', 'pregame_status_90m.csv')])
st = st.dropna(subset=['espn_player_id']).merge(games[['game_id', 'season_year']], on='game_id')
name_id = st.drop_duplicates(['team', 'season_year', 'player_name']).set_index(['team', 'season_year', 'player_name']).espn_player_id
rows = rows.merge(games[['game_id', 'season_year', 'start']], on='game_id')
rows['pid'] = [name_id.get((t, s, n)) for t, s, n in zip(rows.team, rows.season_year, rows.player_name)]
blocks = blocks.merge(games[['game_id', 'start']], on='game_id')
listed = blocks[blocks.state == 'listed']
R['val'] = (R.base_minutes * R.eff_prior).values
G = G[~G.any_block_missing].merge(L, on='game_id', how='left')
# One source for the six-hour rule, as everywhere else in the paper.
STRICT_IDS = set(json.load(open(ROOT / 'results/paper2_strict_sample.json'))['game_ids'])
G['both_filed_6h'] = G.game_id.isin(STRICT_IDS)


def out_at(hours):
    b = listed[listed.ts <= listed.start - pd.Timedelta(hours=hours)]
    best = b.sort_values('ts').groupby(['game_id', 'team']).ts.max()
    rr = rows.copy()
    rr['sel'] = [best.get((g, t)) for g, t in zip(rr.game_id, rr.team)]
    on = rr[(rr.ts == rr.sel) & rr.pid.notna()]
    o = on[on.availability_status == 'Out']
    return set(zip(o.game_id, o.team, o.pid))


late = out_at(0.5) - out_at(6)
val = {}
for gid, team, pid, v in zip(R.game_id, R.team, R.espn_player_id, R.val):
    if (gid, team, pid) in late:
        val[(gid, team)] = val.get((gid, team), 0.0) + v
G['late_out_value'] = [abs(val.get((g, h), 0.0) - val.get((g, a), 0.0)) for g, h, a in zip(G.game_id, G.home, G.away)]

BASE = ['strength_diff', 'rest_diff']
X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])
tr = G[G.season_year.isin((2024, 2025))]
ho = G[(G.season_year == 2026) & G.open_spread_home.notna() & G.close_spread_home.notna()].copy()
for k, c in (('F1', ['F1_diff']), ('F2', ['F2_diff'])):
    ho['p_' + k] = X(ho, c) @ np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0]
ho['line_move'] = (-ho.close_spread_home) - (-ho.open_spread_home)
ho = ho.sort_values('late_out_value', ascending=False).reset_index(drop=True)
top = ho.index < 50
unfiled_in_top = int((~ho.loc[top, 'both_filed_6h']).sum())

# the strict block: both teams filed at six hours, in training and in the holdout, then ranked again
trS, hoS = tr[tr.both_filed_6h], ho[ho.both_filed_6h].copy()
for k, c in (('F1', ['F1_diff']), ('F2', ['F2_diff'])):
    hoS['p_' + k] = X(hoS, c) @ np.linalg.lstsq(X(trS, c), trS.margin.values, rcond=None)[0]
hoS = hoS.sort_values('late_out_value', ascending=False).reset_index(drop=True)
topS = hoS.index < 50


def block(d):
    return dict(n=int(len(d)),
                mean_late_out_value=round(float(d.late_out_value.mean()), 2),
                mean_abs_pred_gap=round(float((d.p_F1 - d.p_F2).abs().mean()), 2),
                mae_compliant=round(float((d.margin - d.p_F1).abs().mean()), 2),
                mae_leaky=round(float((d.margin - d.p_F2).abs().mean()), 2),
                mean_abs_line_move=round(float(d.line_move.abs().mean()), 2),
                median_abs_line_move=round(float(d.line_move.abs().median()), 2),
                mean_abs_margin=round(float(d.margin.abs().mean()), 2))


def pair(a, b, note):
    d = dict(n=int(len(a)) + int(len(b)), top50=block(a), rest=block(b), note=note)
    d['difference'] = {k: round(d['top50'][k] - d['rest'][k], 2) for k in d['top50'] if k != 'n'}
    return d


res = {
    'ranking_quantity': 'late_out_value: the absolute home-minus-away difference of the absence value newly '
                        'ruled Out between the six-hour cutoff and thirty minutes before tip-off. Not the '
                        'total value ruled Out: two teams losing the same value rank near zero.',
    'strict': pair(hoS[topS], hoS[~topS],
                   '2025-26 holdout, priced games where both teams had filed at six hours; models trained on '
                   'the same restriction of 2023-24 and 2024-25. The block the figure shows.'),
    'all_priced': pair(ho[top], ho[~top],
                       '2025-26 holdout, every game with an opening and a closing price; models trained on '
                       '2023-24 and 2024-25. Reported beside the strict block as the original analysis.'),
}
res['all_priced']['top50_with_a_team_that_had_not_filed_at_six_hours'] = unfiled_in_top
json.dump(res, open(ROOT / 'results/paper2_example_aggregate.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
