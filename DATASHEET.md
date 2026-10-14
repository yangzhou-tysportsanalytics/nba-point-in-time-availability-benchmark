# Datasheet: Point-in-Time NBA Player Availability (PIT-Avail), 2021-22 to 2025-26

Format follows Gebru et al., "Datasheets for Datasets" (CACM 2021). Version 1.1 (2026-09-24). Version 1.1 syncs the counts and the paper title to the build on the complete
report series and corrects three numbers carried over from the 4,283-file subset.
Plans and audit reports cited below are in the project repository.

## 1. Motivation
**Purpose.** NBA game-prediction models often encode player absence using who actually played. That information does
not exist before tip-off. This dataset records, for every regular-season game from 2021-22 to 2025-26, what the
official NBA injury report said about each player at a fixed pregame cutoff. It keeps this separate from post-game
participation, so that models can be trained and evaluated without information leakage. It is also the basis for the
paper "Who Was Really Out? Point-in-Time Injury Reports and What Hindsight Actually Buys in NBA Prediction".

**Creators and funding.** [anonymised for review]. No external funding.

## 2. Composition
**What an instance is.** The dataset has three levels.

1. **Game.** One row per regular-season game.
   - Coverage: 6,153 games in the main build (1,230 / 1,230 / 1,231 / 1,231 / 1,231 by season); NBA Cup knockout
     games are included. The public build has 6,146 games (see Public build gaps).
   - The public benchmark table (`bench_games_public.csv`) has one row per game with:
     - the NBA game id, ET date, teams and final margin;
     - pregame team form: last-10 margin difference and rest-day difference;
     - absence-value differences under the compliant version F1 and the leaky versions F2–F4 (section 3);
     - counts of reference-rotation players reported Out, and of those who did not play.
2. **Team-game.** Rows (12,306 in the main build) record:
   - the report used and its printed time;
   - which cutoff applied;
   - whether any report was available (`no_report`).
3. **Player-team-game.** Rows (237,944 in the main build) record:
   - the approximate roster at the cutoff and the reference minutes;
   - the pregame status and reason class at the 6-hour and 90-minute cutoffs;
   - post-game fields, all prefixed `post_`.

   This level is **not redistributed**. Users rebuild it locally (section 6).

**Timing label.** Every field carries one of `key`, `pregame`, `postgame` or `meta`
(`DATA_DICTIONARY.md`). Pregame fields use only information available before the cutoff.

**Cutoff definition.**
- A team's status comes from the last official report whose printed (header) time is at least 6 hours before scheduled
  tip-off and in which that team's block is submitted.
- If no such report exists, the 90-minute cutoff is used. This is the "hybrid" sample; 1,082 team-games (8.8%) use
  it.
- If still no report exists, every player is coded `no_report`: status unknown, **not** healthy. This affects 98
  team-games (0.8%). Games with a `no_report` team are excluded from the benchmark models (97 games of 6,153).

**Status and reason.**
- Status is one of Out, Doubtful, Questionable, Probable or Available, as printed. Players on the roster but absent
  from a submitted report are coded `not_listed`.
- Reasons are grouped into medical, rest, personal (including "not with team"), suspension, G League, trade pending,
  ineligible and unknown. PDF line breaks can split reason text; any fragment outside the non-medical classes is
  treated as medical.

**Relationship between status and play.**
- In the main build (2023-24 to 2025-26, team-games with a report), players reported Out played 0.3% of the time and
  Questionable players 47.8%.
- 43.7% of the reference minutes lost to absent rotation players belonged to players who were **not** reported Out at
  the cutoff, and 29.0% to players not on the report at all. This gap is what leaky features exploit.

**Missing information and known defects.**
- **Report timing.** File names are earlier than the time printed in the report header:
  - hour-only names (e.g. `_05PM`) are 30 minutes earlier (14,891 of the 23,485 reports used);
  - 57 files from 2025-12-19 to 12-22 are 45 minutes earlier;
  - the 8,535 names with minutes (from 2025-12-22) match the header;
  - two files, both published on a daylight-saving change day, are the only exceptions.

  All timing uses the printed header time. In 2023-24 to 2025-26, timing by file name would have used a report printed after the
  cutoff for 53.6% of the team-games it can serve (3,846 of 7,171). On the earlier 4,283-file subset the same calculation
  gave 18.3%, because the nearest file by name was usually well before the cutoff.
- **Name matching.** 98.4–99.98% of report rows match a player id, depending on the season (2025-26 is the lowest). The residue is mostly
  players who never appear in a box score that season (season-long injuries, two-way players). Unmatched rows are kept
  with the printed name and an empty id.
- **Chicago 2025-26 (main build only).** One player appears in the ESPN box score without an id or name in 33 rows. He
  scored in 7 games, so player points there sum to less than team points. He never enters the reference rotation.
- **Approximate rosters.** The roster at the cutoff is inferred:
  - the main build uses box appearances plus the NBA transaction log;
  - the public build uses box appearances only (section 6).

  Recall of players who actually played is 99.96–99.99%.
- **Betting lines.**
  - Available for 2021-22 and 2022-23 (multi-book consensus closing lines, checked against SportsbookReviewsOnline:
    95.4% and 96.8% within 0.5 points) and 2025-26 (DraftKings close).
  - **None for 2023-24 or 2024-25.**
  - Lines are never redistributed.
- **Public build gaps.**
  - The Kaggle source lacks 5 rescheduled 2024-25 games, and 2 more have no box score.
  - Its minutes are whole numbers before 2024-25.
- **ESPN embedded injury field.** ESPN game pages embed an injury field that is a snapshot taken when the page is
  fetched, not a pregame record. It must not be used as a pregame feature (results/paper2_embedded.md).

**Confidential or sensitive data.**
- Injury reasons are health-related information that the NBA itself publishes about public figures.
- No private data is added. Users should not use the data to make claims about individuals' health beyond what the
  league published.

## 3. Collection process
**Official NBA injury reports.**
- PDFs published at `ak-static.cms.nba.com/referee/injury/`, one per hour and, from 22 December 2025, one per
  quarter hour.
- The complete series is used: for every game day, every file from midnight (Eastern) to 90 minutes before the day's
  last tip-off, 23,485 files. The SHA-256 of every file behind a parsed row or team block (23,472 of them) is
  listed in `sources.csv`. (An earlier version used a 4,283-file
  subset; completing the series moved the report selected at the cutoff for 6,588 of 12,306 team-games and cut the
  90-minute fallback from 1,636 to 1,082.)
- Reports were parsed with a column-position parser (`parse_injury_pdf.py`). The printed report time is read from
  each PDF header.

**Parser defect found and fixed (2026-09-23).** A human audit of 80 team-games found that a reason continuing at
the top of a page was attached to the first row of that page instead of the row it continues. The parser was fixed
and the whole series re-parsed: rows and statuses are identical, 22,518 report rows (1.19%) and 20,998 rows of the
point-in-time table gained corrected reason text, `reason_class` changed on 14 rows, and no estimate in the paper
changed. The verification counts are in `results/paper2_parse_verify.json`; the audit record names the players involved and is not redistributed.

**Parsing audit.**
- Automated gates, plus a manual audit.
- For 2021-22 and 2022-23: 39 team-games were read from the PDF images by two independent AI review agents. All 39 Out lists
  and 172/172 player statuses agreed.

**Box scores and schedule.**
- Main build: ESPN game summaries, cached by an earlier project. Not redistributed; ESPN's terms forbid scraping.
- Public build: the CC0 Kaggle dataset "NBA Dataset: Box Scores and Stats (1947 - Today)" by Eoin A. Moore, pinned to
  version 515. It is derived from NBA.com.

**Leakage versions** (per reference-rotation player; values are summed per team and differenced home minus away):
- **Absence value** = reference minutes × per-minute efficiency.
- **Efficiency** = (points + rebounds + assists + steals + blocks − turnovers − missed FG − missed FT) / minutes, over
  the player's last 20 positive-minute games that ended before the cutoff, within 365 days.
- **F1 (compliant):** players reported Out at the cutoff.
- **F2:** players who did not play (post-game).
- **F3:** F1 players valued with whole-season efficiency (look-ahead).
- **F4:** both leaks.

**Time frame.**
- Reports and games run from October 2021 to April 2026.
- Collection happened between 2026-08 and 2026-09.

## 4. Preprocessing, cleaning, labelling
- **Report timing:** by the header time, as above.
- **Report selection:** duplicate rows within a report are removed per (report, game, team, player).
- **Block state:** a team block is `listed` if any player row is printed; otherwise it is "not yet submitted".
- **Misattribution:** players attributed to the wrong team by the parser are detected with the last-known-team rule
  and dropped.
- **Raw data:** the raw PDFs are unchanged, and their hashes are published.

## 5. Uses
**Intended uses:**
- leakage-free evaluation of pregame game-outcome models;
- studying how much pregame availability information is worth;
- auditing published models for participation leakage.

**Uses to avoid:**
- Using `post_` fields, or the ESPN embedded injury field, as features for pregame prediction.
- Treating `no_report` as healthy.
- Treating "not listed" as certain availability: 29.0% of lost rotation minutes come from not-listed players.
- Wagering claims. No model built from availability here beats the closing line, and a model handed the price is
  level with it. The market results rest on one season: on the other holdout season with prices (2022-23) the
  inflation on top of the closing price is −0.044 against +0.020 in 2025-26.

**Benchmark protocol:**
- train on 2023-24 and 2024-25, test on 2025-26;
- exclude games with a `no_report` team;
- report RMSE of margin with a cluster bootstrap (home team) confidence interval;
- the headline metric is **inflation** = RMSE(F1) − RMSE(F2);
- as a stronger compliant baseline, also report F7: every listed player weighted by the probability that players with
  that status sit out, with the probability estimated only from games finished before the cutoff.

**Main results the benchmark supports** (main build, OLS; details in the paper):
- **The pregame report has a stable value.** The benchmark is the strict six-hour one, the 4,131 holdout games where
  both teams had filed at least six hours before tip-off. In rolling-origin holdouts (each season in turn, trained on
  all earlier seasons) the report lowers RMSE by 0.108, 0.199, 0.199 and 0.226 points; pooled 0.185, 95% CI
  0.126–0.245 under the home-team x season cluster design the paper reports, and 0.101–0.265 for the average
  over seasons with the season as the unit. For the estimate one unseen season would produce -- which is what the
  registered 2026-27 test observes, and which carries that season's own deviation as well -- it is 0.0001–0.366,
  so the report's value is resolved for these four seasons and for the average over seasons and not for a single
  unseen season. Accepting a 90-minute report where a team had not filed raises the pooled holdout from 4,131
  games to 4,846 rather than to all of them and gives 0.197 (0.132–0.261); that is
  an operational choice rather than information available six hours out (see the README on how the interval is chosen). Three of the four
  per-season intervals exclude zero; 2022-23 reaches it (0.149, −0.006 to 0.302).
- **Participation leakage is small on average and largest in 2025-26.** On the 2025-26 holdout, participation (F2) adds an apparent
  0.166 (0.077–0.255) over the report (F1), a ratio of 1.73; pooled over the four rolling holdouts it is
  0.059 (0.001–0.119), whose lower bound is zero to within the bootstrap's own Monte-Carlo error: across five seeds at
  10,000 replicates it runs −0.0017 to 0.0012, above zero in three and below in two. It is not resolved under any
  reading, and as a statement about a season it is 0.059 (−0.079 to 0.192). Per season only 2025-26's premium is
  clear of zero (0.172, 0.087–0.262). A joint test does not resolve that the seasons differ (p = 0.12 under the
  design these are read under); the linear trend of 0.064 per season has an interval that excludes zero (0.004–0.121) but is
  not read as a trend, since the omnibus test does not resolve, four seasons cannot separate a trend from a
  season-level term, and the training window grows with the holdout season. The inflation is concentrated in March–April (0.184), at cutoffs far
  from tip-off, and in games that were expected to be one-sided: by tercile of the pregame predicted margin it is
  0.044, 0.004 and 0.129, and by tercile of the closing spread 0.001, 0.017 and 0.231, while the log-loss inflation
  stays near zero throughout, so it does not change who is predicted to win.
  - Against the stronger baseline F7 it is 0.031 (−0.017 to 0.077) pooled over the four holdouts, and its interval excludes zero in
    26 of 28 model and feature settings on that holdout under home-team clustering; leaky models fed pregame features lose to honest ones in 27
    of 28.
  - The public rebuild gives 0.176 (0.079–0.271) on the same holdout, against 0.166 in the main build.
- **The leak is a timing quantity.** Re-reading only the Out status at other cutoffs, the compliant gain rises from
  0.139 twelve hours out to 0.185 at six hours and 0.258 at thirty minutes, and the inflation falls to zero by about
  90 minutes. The twelve- and nine-hour points are computed only on the games where both teams had filed by then
  (3,401 of 4,131): on the full set they read 0.128 rather than 0.054, because a team that has not filed contributes
  no absentees, which is the same mistake this dataset exists to avoid. The gain between cutoffs is a paired
  quantity, both against the six-hour report on the same games: the thirty-minute report beats it by 0.073
  (0.026–0.119) and the one-hour report by 0.067 (0.030–0.103).
- **The leak is who played, not how long.**
  - Adding realized minutes on top of participation changes nothing (−0.011, −0.048 to 0.022).
  - In a present-player form, realized minutes tend to hurt (−0.090, −0.190 to 0.010).
  - Users who swap realized minutes for averages but keep the box-score roster have not removed the leak.
- **Harmless shortcuts.** Whole-season efficiency (F3) and status rates estimated on the full sample leak almost
  nothing (|Δ| ≤ 0.005).
- **How much the valuation matters** (both runs corrected on 2026-09-26, then aligned with the build's own
  history rule after an independent code audit). Without the 240-minute
  rescale — same history as the build, last 20 positive-minute games within 365 days of that team-game's own
  cutoff, so the rescaling is the only thing switched off — the report is worth 0.205 (0.143–0.274) and the pooled
  premium is 0.038 (−0.017 to 0.095); the rescaling accounts for about a third of the inflation and none of the
  report's value. A third valuation charges an absence only at the player's margin over a replacement, which is what it costs a team
  that fills the minutes with someone else: the report is worth 0.162 (0.102–0.223) and the premium 0.038
  (−0.004 to 0.079). With a plus-minus impact measure in place of per-minute efficiency they are 0.051
  (0.014–0.086) and −0.030 (−0.053 to −0.004), the second negative and clear of zero. The three valuations differ
  in how much of the fact that someone else plays each already contains, and both quantities fall as more of it
  does. The report's value is positive and clear of zero under all three; the premium is resolved above zero under
  none of them. Users building their own valuation should expect the premium, including its sign, to move with it.
- **The public path reproduces the headline.** Running the same pooled rolling design on the released rebuild
  (4,835 holdout games against 4,846) gives a report value of 0.196 (0.131–0.260) against 0.197 and a premium of
  0.074 (0.012–0.135) against 0.059.
- **A deflating field.** ESPN's page-embedded injury field makes availability look worthless: 0.258 (0.122–0.399)
  worse than the compliant report.
- **One kind of absence is followed by the leak.** The difference between the compliant and the leaky feature
  decomposes exactly — absence declared Out after the cutoff, absence listed later but not as Out, absence on no
  report we hold, minus players Out at the cutoff who played (93 player-games) — and the identity is verified
  player-game by player-game and game by game (`results/paper2_mechanism5.json`; the analysis script is released with the paper's full code). Only the first is followed by the
  premium: −0.026 (−0.087 to 0.033) in games with none against 0.188 (0.046–0.326) in the top group, a
  difference of 0.214 (0.068–0.362), which excludes zero under the design these are read under. The other two spread the features just as far apart and show no detectable
  relation to the inflation (−0.047 and −0.081, both intervals including zero). Two tests that do not sort games
  on a component of what they explain (`results/paper2_mechanism4.json`) give the negative half and the positive half. On the
  headline feature and sample, a retrospective put-back of the absences declared Out after the cutoff whose players
  did not play is worth 0.070 (0.026–0.115), and the difference that then remains from the participation feature,
  −0.012 (−0.046 to 0.027), is not resolved; adding the late-status absences back is worth 0.001 (−0.020 to
  0.022) and the ones on no report we hold −0.033 (−0.058 to −0.010), against the 0.059 that participation is worth
  over the same feature. Entering each component as its own regressor instead, so the data chooses its weight rather
  than inheriting the valuation's, keeps the first at 0.069 (0.029–0.109) and leaves the other two with nothing: the
  never-listed component reads −0.013 (−0.025 to −0.000), which a Diebold-Mariano test would call significantly
  worse at p = 0.04 while the Clark-West adjustment puts it at p = 0.83, because a nested model cannot be worse in
  population and that negative is the price of the extra parameter. Each put-back adds the component inside the absence-value feature, whose coefficient is
  refitted with it at every origin and applied to training as well as the holdout, so a null is a null about that
  encoding rather than a demonstration that the information is absent — and all three components are defined on
  players who did not play, so the put-backs are retrospective and not a build available at the cutoff.
  The later report is not the earlier one with Out designations added: 236 players Out at the cutoff are not Out
  on the last report we hold (151 of them sat), so the paired cutoff contrast measures the net effect of reading
  the report later.
- the player-level table, raw PDFs, box scores and betting lines;
- reason: the NBA.com Terms of Use (2026-07-13) restrict redistribution and "comprehensive" statistics databases, and
  the Disney/ESPN Terms (2024-05-24) forbid scraping and compiling data collections (
  this is not legal advice).

**Rebuilding.**
1. `fetch_sources.py` downloads the official PDFs and the pinned Kaggle files, and checks every SHA-256.
2. `rebuild.py` parses and builds the tables.

Users are responsible for complying with each source's terms.

**Public build versus main build.** Differences:
- scheduled rather than actual tip-off times;
- no transaction log;
- the Kaggle gaps listed above.

Agreement (results/paper2_public_path.md):
- 6,146 games joined; margins identical.
- Correlation of the compliant feature with the main build: 0.986.
- Rotation-player Out counts identical in about 92% of team-games; "did not play" counts in 64%, because of the missing transaction log.
- On the 2025-26 holdout: compliant gain 0.213 (95% CI 0.078–0.340) against 0.226 in the main build, inflation
  0.176 (0.079–0.271) against 0.166 (0.077–0.255).

## 7. Maintenance
- **Contact:** [anonymised for review].
- **Errors:** reported via GitHub issues; fixes are released as new Zenodo versions.
- **2026-27:** a prospective extension, collected in real time, is planned; it depends on the authors' capacity.
- **Source changes:** if the NBA moves or removes the PDFs, the SHA-256 list still identifies the exact files for
  anyone holding copies.
