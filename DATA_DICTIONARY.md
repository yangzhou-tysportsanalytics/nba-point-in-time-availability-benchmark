# Paper 2 release: data dictionary

Generated from the column descriptions in the release build script, which is released with the paper's full code; do not edit by hand.

**Timing** tells you when a value is knowable. `pregame` = known at the stated cutoff and safe as a prediction input; `postgame` = an outcome, never a pregame input (the leakage experiments in the paper use exactly these columns as the leak); `key` / `meta` = identifiers and provenance.

Coverage: 2021-22, 2022-23, 2023-24, 2024-25, 2025-26 NBA regular seasons. Report times are the times printed on the PDFs. Sources: official NBA injury report PDFs (parsed by `league/scripts/parse_injury_pdf.py`), ESPN game summaries (header, box score, pickcenter only; the summary-embedded injuries field is never read).

## games.csv.gz  (6,153 rows)

| column | timing | type | non-null | description |
| --- | --- | --- | --- | --- |
| `game_id` | key | str | 100.0% | ESPN event id. |
| `season_year` | key | int64 | 100.0% | Season end year (2024 = 2023-24). |
| `season` | key | str | 100.0% | Season label, e.g. '2023-24'. |
| `start_utc` | pregame | str | 100.0% | Scheduled tip-off time, UTC. |
| `game_date_et` | pregame | str | 100.0% | Game date in US Eastern time. |
| `home` | key | str | 100.0% | Home team abbreviation (ESPN). |
| `away` | key | str | 100.0% | Away team abbreviation (ESPN). |
| `neutral` | pregame | bool | 100.0% | Neutral-site game flag (ESPN). |
| `home_score` | postgame | int64 | 100.0% | Final home points. |
| `away_score` | postgame | int64 | 100.0% | Final away points. |
| `margin_home` | postgame | int64 | 100.0% | home_score - away_score. |
| `market_provider` | meta | str | 55.9% | Provider of the betting line in the cached ESPN summary (first pickcenter entry); empty if none. |
| `market_spread_home` | pregame | float64 | 55.9% | Home closing point spread (negative = home favored). 2021-22 and 2022-23: multi-book consensus stored by ESPN, verified as the closing line against SportsbookReviewsOnline (95-97% within 0.5; results/paper2_1b_e4c.md). 2025-26: DraftKings close (ESPN pointSpread.home.close, 979/979; results/paper2_1b_e0.md). Missing for 2023-24 and 2024-25. |
| `market_total` | pregame | float64 | 55.9% | Over/under total from the same provider. |
| `market_home_ml` | pregame | float64 | 55.8% | Home moneyline, same provider. |
| `market_away_ml` | pregame | float64 | 55.8% | Away moneyline, same provider. |

## team_games.csv.gz  (12,306 rows)

| column | timing | type | non-null | description |
| --- | --- | --- | --- | --- |
| `game_id` | key | str | 100.0% | ESPN event id. |
| `season_year` | key | int64 | 100.0% | Season end year. |
| `team` | key | str | 100.0% | Team abbreviation. |
| `opp` | key | str | 100.0% | Opponent abbreviation. |
| `home` | key | int64 | 100.0% | 1 if home team. |
| `start_utc` | pregame | str | 100.0% | Scheduled tip-off, UTC. |
| `cutoff_variant` | pregame | str | 100.0% | Which cutoff supplies the primary status for this team-game: '6h' (last report >= 6 h before tip) or '90m' (team block not yet submitted at 6 h, so the last report >= 90 min before tip is used). |
| `cutoff_utc` | pregame | str | 100.0% | Information cutoff for the primary status, UTC. |
| `report_found` | pregame | bool | 100.0% | A submitted report block for this team exists before the primary cutoff. |
| `report_ts_et` | pregame | str | 99.2% | Time (ET) printed in the header of the official report used for the primary status (file names of pre-Dec-2025 reports understate it by 30-45 min; results/paper2_1b_timing.md). |
| `report_age_hours` | pregame | float64 | 99.2% | Hours between that report and the primary cutoff. |
| `report_fallback` | pregame | object | 99.2% | Report taken from an earlier file because the latest pre-cutoff file lacked this team. |
| `no_report` | pregame | bool | 100.0% | No submitted report block before either cutoff; player statuses for this team-game are unknown (status = no_report), NOT healthy. |
| `report_ts_6h_et` | pregame | str | 91.2% | Timestamp (ET) of the report used at the 6-hour cutoff; empty when the team block was not yet submitted at 6 h. |
| `no_report_6h` | pregame | bool | 100.0% | Team block missing at the 6-hour cutoff. |
| `report_ts_90m_et` | pregame | str | 99.2% | Timestamp (ET) of the report used at the 90-minute cutoff; empty when the team block was still missing. |
| `no_report_90m` | pregame | bool | 100.0% | Team block missing at the 90-minute cutoff. |
| `listed_n` | pregame | int64 | 100.0% | Players listed on the primary report for this team. |
| `listed_unmatched` | pregame | int64 | 100.0% | Listed names that could not be matched to an ESPN player id. |
| `listed_misattributed` | pregame | int64 | 100.0% | Listed names matched to a player whose approximate roster team differs. |
| `roster_n` | pregame | int64 | 100.0% | Size of the approximate roster at the cutoff (most-recent-event rule: box listings, NBA transactions, report listings). |
| `points` | postgame | int64 | 100.0% | Team points. |
| `margin` | postgame | int64 | 100.0% | Team points - opponent points. |

## player_availability.csv.gz  (237,944 rows)

| column | timing | type | non-null | description |
| --- | --- | --- | --- | --- |
| `game_id` | key | str | 100.0% | ESPN event id. |
| `season_year` | key | int64 | 100.0% | Season end year. |
| `team` | key | str | 100.0% | Team abbreviation. |
| `espn_player_id` | key | str | 99.9% | ESPN athlete id; empty when a report name could not be matched (see name_report). |
| `name_espn` | meta | str | 99.9% | Player name as in ESPN box scores (any game). |
| `name_report` | meta | str | 27.0% | Player name as printed on the official injury report ('Last, First'), if listed. |
| `in_roster_approx` | pregame | bool | 100.0% | Player is on the approximate roster at the cutoff. |
| `ref_history_games` | pregame | float64 | 99.7% | Positive-minute games in the reference window (last 20 before the cutoff). |
| `ref_minutes` | pregame | float64 | 99.7% | Reference minutes per game from pre-cutoff history (shrunk), used to build rotations. |
| `ref_support` | pregame | str | 99.7% | History support flags ('supported' or reasons such as low_sample, low_minutes, no_history). |
| `status` | pregame | str | 100.0% | Primary pregame status at the team's cutoff_variant: Out, Doubtful, Questionable, Probable, Available, not_listed (team report submitted, player not on it), no_report (team report missing: unknown). |
| `reason_class` | pregame | str | 26.5% | Primary reason class: medical, gleague, personal, rest, suspension, trade_pending, ineligible, unknown. Never imputed. |
| `reason` | pregame | str | 26.5% | Reason text as printed on the primary report. |
| `status_6h` | pregame | str | 100.0% | Status on the last report >= 6 h before tip (not_listed / no_report as above). |
| `reason_class_6h` | pregame | str | 24.3% | Reason class at 6 h. |
| `status_90m` | pregame | str | 100.0% | Status on the last report >= 90 min before tip. |
| `reason_class_90m` | pregame | str | 27.0% | Reason class at 90 min. |
| `post_in_box` | postgame | bool | 100.0% | Player appears in the ESPN box score (played or DNP line). OUTCOME - do not use as a pregame input. |
| `post_played` | postgame | bool | 100.0% | Player logged positive minutes. OUTCOME. |
| `post_minutes` | postgame | float64 | 55.0% | Minutes played (empty if not in box or no minutes). OUTCOME. |
| `post_starter` | postgame | bool | 100.0% | Started the game. OUTCOME. |
| `post_dnp_reason` | postgame | str | 12.1% | ESPN post-game DNP text (e.g. COACH'S DECISION). OUTCOME; ESPN may edit after the game. |

## Value distributions

`player_availability.status`: not_listed 173,071, Out 45,770, Questionable 10,137, Probable 3,426, Available 2,285, no_report 1,807, Doubtful 1,448

`player_availability.reason_class`: medical 44,419, gleague 16,097, personal 1,379, rest 336, unknown 319, suspension 287, trade_pending 173, ineligible 61
