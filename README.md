# PIT-Avail: a point-in-time NBA player availability benchmark

This repository accompanies the paper *"Who Was Really Out? A Point-in-Time NBA Availability Benchmark, and What
It Says About Hindsight"* (under review).

A box score records who **did** play. It does not record who was **known to be out** before tip-off, and a model
graded on the box score is graded on information nobody had at the cutoff. This repository holds the benchmark that
separates the two: what the official NBA injury report said about every player at a fixed pregame cutoff, for every
regular-season game from 2021-22 to 2025-26, kept apart from what happened afterwards.

## Start here

| If you want to… | Do this |
| --- | --- |
| **Use the benchmark** | `data/bench_games_public.csv` — one row per game, with the compliant and leaky absence features already built. |
| **Check a number in the paper** | `results/` holds the file behind every number. The map is in [Reproducing the numbers](#reproducing-the-numbers). |
| **Rebuild everything from public sources** | Four commands, below. Roughly two hours and 2 GB, no account needed. |

## What is not here, and why

The point-in-time table is derived from the league's published injury-report PDFs and from a box-score dataset
whose uploader released the compilation under CC0. **That dedication covers the compilation, not the underlying
NBA data**, and the NBA.com terms restrict redistribution of the source documents and of comprehensive statistics
databases built from them. So:

- **No player-level tables.** We release game-level and aggregate tables. `rebuild.py` regenerates the
  player-level tables on your machine in one command, and the exact source files are pinned by SHA-256 in
  `league/public/sources.csv` (23,485 injury-report PDFs and 2 Kaggle files).
- **No source documents.** No PDFs, no box scores. The fetch script downloads them from the league's public URL and
  verifies every hash.
- **No betting lines.** They are not public and are not used in anything released here.
- **One redaction.** `results/paper2_ladder_example.json` keeps the worked example's game-level fields and drops its
  per-player rows. The paper describes the same example in prose.

Nothing here is fetched from ESPN or from stats.nba.com.

## What is here

| Path | Contents |
| --- | --- |
| `data/bench_games_public.csv` | **Game-level benchmark**, 6,146 games: final margin, both controls, and the absence-value features F1–F4. |
| `data/status_table_public.csv` | Pregame status against whether the player played (public build). |
| `results/*.json` | Every result file behind the paper's numbers: the rolling-origin holdouts, the cutoff curve, the decomposition and the put-back tests, the valuation runs, the model grid, the market section. |
| `results/figures/` | The cutoff-curve figure, as it appears in the paper and in the abstract. |
| `league/public/sources.csv` | Every source file with URL, size and SHA-256. |
| `league/scripts/fetch_sources.py` | Downloads the sources and verifies every hash. |
| `league/scripts/rebuild.py` | Parses the PDFs, applies the point-in-time rules, builds the tables, runs the model. |
| `league/scripts/parse_injury_pdf.py` | The injury-report parser. |
| `league/public/pit_rules.py` | Name matching, reason classes and reference minutes, copied verbatim from the project code. |
| `league/scripts/cutoff_f7_main_build.py` | Plan Y as it was run on the main build: what reading the report later is worth once the early report is read in full. Published to be read as it ran; the next row runs the same question here. |
| `league/scripts/cutoff_f7_public.py` | Plan BB: the same crossing on this rebuild, with the controls the plan required before anything could be read off it. |
| `league/scripts/roster_audit_sample_main_build.py` | Plan AA as it was run: drawing the sample for the human audit of the roster rule. Published to be read; the sample is a player-level table and is not shipped. |
| `league/scripts/roster_audit_result_main_build.py` | Plan AA section 5 as it was run: the error rates, the control stratum and the corrected build. |
| `league/scripts/roster_audit_reruns_main_build.py` | Plan AA section 5.4 as it was run: the strict headline and the never-listed put-back on the corrected build. |
| `league/scripts/prospective_poll.py` | The collector for the prospective test described below. Not part of the rebuild. |
| `league/tests/` | Tests for the public path. |
| `src/` | The project modules the published scripts import, copied verbatim: bootstrap designs, the two build paths' alignment and paired contrast, the join between them, the cutoff encoding, and the rolling per-status sit-out rates the all-status encoding weights a report with. |
| `docs/plans/` | The plans those scripts were run to, each pinned by hash inside the script. |
| `DATASHEET.md` | Datasheet (Gebru et al., 2021). |
| `croissant.json` | Croissant 1.0 metadata for the two released data tables (validated with `mlcroissant`). |
| `requirements.txt` | Pinned environment. |

**Columns of `data/bench_games_public.csv`:** NBA game id, date, home and away teams, final margin; recent-form and
rest differences; absence-value differences for the compliant version (F1) and the leaky versions (F2 actual
participation, F3 whole-season efficiency, F4 both); counts of rotation players reported Out and of those who did
not play; a flag for games where a team had no submitted report.

## Quick check (no download)

```
pip install -r requirements.txt
python league/scripts/rebuild.py model --bench data/bench_games_public.csv
```

This refits the headline model on the released table and writes `results/paper2_public_path_check.json`, which
should match `results/paper2_public_path.json`.

## Full rebuild from public sources

```
pip install -r requirements.txt
python league/scripts/fetch_sources.py                # ~1.5 GB of PDFs + ~400 MB of Kaggle CSVs; verifies SHA-256
python league/scripts/rebuild.py parse 3600   # re-run until it prints "done" (4 processes, about an hour)
python league/scripts/rebuild.py build
python league/scripts/rebuild.py model
python -m unittest discover league/tests
```

Outputs, including the player-level tables we do not redistribute, go to `data/public/out/`.

## The point-in-time rules, in short

- **Report time.** Each report is timed by the time **printed in its header**, not by its file name. Hour-only file
  names are 30 minutes (occasionally 45) earlier than the printed time. A pipeline that times reports by file name
  selects a report published *after* its own cutoff for 53.6% of the team-games it can serve.
- **Status.** A team's status comes from the last report printed at least six hours before tip-off in which the
  team's block is submitted. If there is none, the 90-minute report is used; that fallback is itself information
  from after the cutoff, and the paper reports the benchmark without it as well. If there is still none, the status
  is `no_report`: **unknown, not healthy**, and the game leaves the benchmark.
- **Absence value.** Reference minutes × per-minute efficiency over the player's prior 20 games that ended before
  the cutoff.
- **Evaluation.** Rolling-origin holdouts: each season from 2022-23 to 2025-26 in turn, trained on all earlier
  seasons, then pooled (4,846 games). Inflation is RMSE(F1) − RMSE(F2). Intervals resample home-team×season clusters, the design the paper reports; the away-side mirror is
  reported beside it, because the placebo calibration scores the two alike and cannot order them, and
  the two-way team bootstrap earlier versions used is withdrawn, covering at essentially one.

## Key results

Pooled over the four rolling-origin holdouts. Two readings are given because they answer different questions,
and each is read under the design the calibration scores at its nominal rate for that question (see *How the
interval is chosen*): home-team x season clusters for these four seasons, a seasons-then-clusters bootstrap for
a statement about a season.

The benchmark is the strict six-hour one: the 4,131 holdout games where **both** teams had filed at least six
hours before tip-off, with the same rule applied to the training set. The paper's other feature accepts a
90-minute report where a team had not filed, which raises the pooled holdout from 4,131 games to 4,846 rather
than to all of them and reaches 8.0% of team-games; that is an operational choice rather than information
available six hours out, and it is reported beside the benchmark.

Three readings, and they are not the same question. The first fixes these four seasons. The second is the
average over seasons drawn from the same process. The third is the estimate one unseen season would produce,
which carries that season's own deviation from the average as well and is therefore 2.24 times wider at four
seasons -- and it is what the registered 2026-27 test will observe. The second and third are centred on the
mean of the four season estimates, 0.183 and 0.046 on the strict sample, which is not the pooled estimate in
the first column: a pooled RMSE difference is not the average of four season-specific ones.

| Quantity | In these four seasons | Averaged over seasons | One unseen season |
| --- | --- | --- | --- |
| Value of the pregame report, strict six hours | **0.185 (0.126-0.245)** | 0.183 (0.101-0.265) | 0.183 (0.0001-0.366) |
| Premium from actual participation, strict six hours | 0.050 (-0.011 to 0.110) | 0.046 (-0.095 to 0.188) | 0.046 (-0.270 to 0.362) |
| The report's value accepting the 90-minute fallback | 0.197 (0.132-0.261) | 0.196 (0.144-0.247) | 0.196 (0.080-0.311) |
| The premium accepting the fallback | 0.059 (0.000-0.118) | 0.057 (-0.079 to 0.192) | 0.057 (-0.246 to 0.359) |
| The last two on this rebuild (4,835 games) | 0.196 (0.131-0.260) and 0.074 (0.012-0.135) | - | - |

So the report's value is resolved for these four seasons and for the average over seasons. The third column is an
illustration of how little four seasons settle rather than a result: its bounds are analytic rather than
bootstrap quantiles, it treats four rolling-origin estimates as independent draws when they share training
history, and the strict sample's lower bound of 0.0001 sits against the fallback's 0.080. Earlier versions of
these documents called the second column a statement about an unseen season, and gave the fallback row's
second and third columns as (0.091-0.300) and (-0.037 to 0.428); that interval was computed with a standard
error that an external review found inflated wherever the season-to-season spread is smaller than the
sampling noise inside a season, which is the case for this quantity.

**The report's own value is the stable part. The premium is not resolved under any reading.** Its lower bound
for these four seasons is zero to within the bootstrap's own Monte-Carlo error: across five seeds at 10,000
replicates it runs -0.0017 to 0.0012, above zero in three of the five and below it in two. Averaged over
seasons it is 0.057 (-0.079 to 0.192) and for one unseen season 0.057 (-0.246 to 0.359). We report the point
estimate and do not call it resolved under any of the three.

- **The premium's magnitude depends on how an absent player is valued.** Charging the whole player gives a
  report value of 0.197 and a premium of 0.059; charging only his margin over a replacement gives 0.162
  (0.102-0.223) and 0.038 (-0.004 to 0.079), so the premium falls by about a third of itself between the two.
  A third feature, built from per-minute plus-minus, gives 0.051 (0.014-0.086) and -0.030 (-0.053 to -0.004),
  but these data cannot separate it from team strength: it correlates -0.03 with the feature it would revalue,
  enters the model with the opposite sign, and keeps 3% of its coefficient once the same measure over the
  whole reference rotation is controlled for, so its answer is not evidence about valuing an absence. The
  report's value is resolved above zero under both valuations and under that feature as well, which are three
  sizes and not three confirmations of a sign: the sign is non-negative for every valuation, because dropping
  the feature is the same model with its coefficient set to zero, and that nesting is also why the premium,
  which has no such guarantee, is read differently.
  Without the 240-minute rescaling the pair reads 0.205 (0.143-0.274) and 0.038 (-0.017 to 0.095).
- **It depends on the cutoff.** On the games where both teams had filed at six hours, the report is worth
  0.185 (0.126-0.245) six hours out and 0.258 (0.185-0.333) thirty minutes out. Read at one hour instead of
  six, on the same games, the compliant feature gains 0.067 (0.030-0.103); at thirty minutes, 0.073
  (0.026-0.119). At thirty minutes the premium is -0.022 (-0.058 to 0.014): the report is already level with
  knowing who played.
- **The twelve-hour point needs care.** On the games where both teams had actually filed that early, the
  report is worth 0.153 (0.089-0.216) and the premium is 0.054 (-0.010 to 0.125). Computed on all games,
  imputing health for the teams that had not filed, the same pair reads 0.108 and 0.128 - which is the mistake
  this paper is about, and the reason the first version of that point was wrong.
- **Stronger compliant baselines close part of it.** Weighting every listed status by its sit-out rate (F7) is
  worth 0.224 (0.152-0.296) against F1's 0.197, and the premium measured against F7 falls to 0.031 (-0.017 to
  0.077). A baseline that needs no report at all - the rotation players who did not play in their team's
  previous game - is worth 0.092 (0.050-0.133), and the report added to a model that keeps it is worth
  0.109 (0.056-0.163), a nested comparison rather than a swap of one feature for the other.
- **Which absences carry it.** The difference between the two features decomposes exactly into absence declared
  Out after the cutoff, absence listed later but not as Out, and absence on no report we hold, minus players
  Out at the cutoff who played. A retrospective put-back of the first is worth 0.070 (0.026-0.115); the
  difference that then remains from the participation feature, -0.012 (-0.046 to 0.027), is not resolved.
  Entering each component as its own regressor instead, so the data chooses its weight, keeps the first at
  0.069 (0.029-0.109) and leaves the other two with nothing: the never-listed component reads -0.013 (-0.025
  to -0.000), which a Diebold-Mariano test would call significantly worse at p = 0.04 and the Clark-West
  adjustment puts at p = 0.83. A nested model cannot be worse in population, so a negative here is one of
  the things a finite sample produces for a feature in which nothing is detected; the nesting bounds the
  population comparison and does not identify what the loss is made of.
- **What a buyer should ask for.** The gap between a model's own backtest and the accuracy it delivers when
  handed only the pregame report, pooled over the four holdouts, is 0.068 (0.015-0.120) with narrow features
  and 0.100 (0.064-0.135) with wide ones; 0.086 (0.035-0.137) on this rebuild. It is not steady across
  seasons, and no single season reproduced it.
- **Who played, not how long.** Once participation is known, realized minutes add nothing.
- **No market edge.** No model built *without* the price beats the closing line. A model handed the closing
  price is not established either to improve on it with availability or to fail to: the point estimates
  favour the model and the intervals contain zero. Against the opening price the premium is clear of zero in
  both lined seasons with opposite signs, which the paper no longer reads as a failure to replicate: the two
  seasons cover different parts of the calendar, overlapping in three months, and use different price
  products, and restricting to the overlap lowers the later estimate and widens it without reconciling the two.

## Reproducing the numbers

Every number in the paper is generated from a file in `results/`. The main ones:

| Quantity | File | Key |
| --- | --- | --- |
| Rolling-origin holdouts, both designs | `paper2_inference.json` | `designs`, `per_season` |
| Cutoff curve, paired cutoff contrasts | `paper2_cutoff_curve.json` | `curve`, `curve_both_filed`, `paired_1h_minus_6h` |
| Decomposition of the two features | `paper2_mechanism5.json` | `identity`, `dose_*` |
| Put-back tests | `paper2_mechanism4.json` | `contrasts_vs_F1_headline`, `contrasts_vs_F2` |
| What the retrospective filter excludes | `paper2_putback_counts.json` | `all_bench_games` |
| Plus-minus valuation | `paper2_impact_valuation.json` | `estimates` |
| Without the 240-minute rescaling | `paper2_valuation_robustness.json` | `estimates` |
| Model grid, deployment gap | `paper2_models.json`, `paper2_models_extra.json` | `A_main`, `B_five_seasons` |
| Deployment gap, pooled over four holdouts | `paper2_deployment_pooled.json` | `feature_sets` |
| Market section | `paper2_close_line.json`, `paper2_open_line.json` | `rmse_vs_lines`, `estimates` |
| Public rebuild against the main build | `paper2_public_path.json`, `paper2_public_path_compare.json` | — |

The files in `results/` keep the names the analysis pipeline writes, so what is published here is what the code produces. A few of them carry that pipeline's own stage names (`paper2_1b_*`, `paper2_stage0_*`, `paper2_text_numbers.json`); they hold the parsing gates, the source inventory and the counts quoted in the dataset section, and they are included because the paper prints numbers from them.

## Differences from the paper's main build

- Tip-off is the scheduled time.
- The roster at the cutoff uses box appearances only; there is no transaction log.
- Box scores come from the CC0 Kaggle dataset, which lacks a few games (6,146 against 6,153).

**Consequences for the released tables.**

- **Departed players.** Without a transaction log, a player who has left a team stays on its roster until he appears
  for another team or his history expires, and then counts as "did not play" for his old team. This raises the F2
  feature relative to the main build: about 0.3–0.5 more "did not play" rotation players per team-game.
- **Status table.** `data/status_table_public.csv` counts roster rows only, while the paper's Table 1 counts rows of
  the main build's release table (roster ∪ report ∪ box score), so the shares differ: the share of not-listed
  players who played is 58.9% here and 70.6% in the paper.

`results/paper2_public_path_compare.json` quantifies the agreement: margins match exactly on the joined games, the
compliant feature correlates 0.986 with the main build and the participation feature 0.973, and the two paths agree
on whether to exclude a game in 99.7% of cases.

## The prospective test

The estimates in this repository are retrospective: the seasons were over and their outcomes were known when the
analyses were designed. A confirmatory test on the 2026-27 regular season was registered on OSF on 2026-09-30,
twenty days before that season's first game, with its data, rules, models, metrics and decision criteria fixed:
[osf.io/fxtup](https://osf.io/fxtup). It is embargoed until 2027-05-31, after the season it tests has ended, so
the page states the timestamp and opens then. The registration pins this repository at commit `04385c9`.

`league/scripts/prospective_poll.py` is the collector that test relies on, published here because it is the part
that cannot be reconstructed afterwards. The benchmark above dates each report by the time printed on it, which is
when the report says it was issued, not when it could first be read. The collector records the second quantity:
every few minutes it probes the league's published file names and writes down, for each file, when it first
answered, the server's own clock, and the validators it returned. It takes no arguments, writes to
`data/prospective_2627/`, and is not part of the rebuild above — running it starts a collection, it does not
reproduce anything.

### How the interval is chosen, and why it changed

The registration first decided every hypothesis on a two-way team bootstrap, which draws the 30 teams once and
weights a game by the product of its two teams' draw counts. Two studies run afterwards, on the retrospective
seasons only, show that design is not calibrated: on placebo data with a known truth it covers a nominal 95%
interval at essentially 1, with an interval about 1.5 times the sampling distribution, because a product of two
resampling counts has variance near 2.6 where an ordinary bootstrap weight has 1. The per-game loss difference
has almost no team structure to justify the extra width — under 1% of its variance.

On one season that costs power and buys nothing, so the registration was amended before the season's first game
to decide on an iid game bootstrap, which covers at the nominal level here. The calibration scores six designs:
the four bootstraps above, the away-team x season mirror of the reported one, and the analytic dyadic estimator,
which are the readings the paper reports from it. Both studies run on the released build:

```
python league/scripts/boot_calibration.py     # -> results/paper2_boot_calibration_public.json
python league/scripts/prospective_power.py    # -> results/paper2_prospective_power_public.json
```

They take a few minutes each and need only `data/bench_games_public.csv`. The same scripts run on the main build
in the originating project; the two agree on every conclusion and differ in the third decimal, as the public path
does for the headline. Note that the paper and the prospective test do not use the same interval, and should not:
the paper generalises across four holdout seasons whose inflation differs from season to season, and that
variance belongs in its interval; the prospective test is one season, where that term does not exist.

### Rerunning the player-level analyses

The valuations, the decomposition and the cutoff curve need a player-level table, and the source terms forbid
redistributing one, so this repository ships none. The rebuild writes it on your machine instead, which the
terms allow, and the released scripts read it from there:

```
python league/scripts/fetch_sources.py
python league/scripts/rebuild.py parse 3600     # until it prints "done"
python league/scripts/rebuild.py build          # also writes data/public/out/bench_ref_players_public.csv
python league/scripts/impact_valuation.py       # the plus-minus valuation
python league/scripts/valuation_robustness.py   # without the 240-minute rescaling
python league/scripts/replacement_valuation.py  # at replacement level
python league/scripts/cutoff_curve.py           # the curve, and the twelve-hour artefact
python league/scripts/decomposition.py          # the three components, both readings of the put-backs
python league/scripts/replacement_mechanism.py  # the curve and the put-backs under all three valuations
python league/scripts/pooled_headline.py        # the pooled rolling headline on this rebuild
python league/scripts/cutoff_f7_public.py       # plan BB: the two encodings crossed with two cutoffs
```

None of them reimplements the build: the report rows, the name matching and the rule for which report is
current at a cutoff come from `rebuild.py` called with a different cutoff, and `history.py` is the originating
project's module shipped unchanged, because those rules are part of what the paper claims. Each script prints
the figures this repository already publishes as a control, so you can see the plumbing is right before
reading anything new off it.

**The shipped decomposition result predates a correction to this script.** Until 2026-10-20 the script
subtracted a fourth term from its "all three" feature --- the player-games the compliant feature calls Out
who then played --- which closes the identity and makes that feature the leaky one exactly. So in
`results/paper2_decomposition_public.json` the `all three` put-back is 0.0736 and so is the `F1 vs F2`
reference row, to the last digit, by construction rather than by result, and neither is the paper's `all
three`, which is 0.057 on the main build because it keeps that fourth term in. The script now computes both
and names them apart; rerunning it after a rebuild produces the corrected file.

**One analysis in this release cannot be rerun here.** It compares this build's availability features with
the main build's on the games both keep, with one outcome, one rolling split and one pair of controls, and
differences the two premiums inside the same bootstrap draw: it needs both builds' game-level tables and
this repository ships one of them. Its script is `league/scripts/paired_paths_main_build.py`, published as it
was run, with the modules it imports in `src/` and the plan it was run to in `docs/plans/`, and its
result is `results/paper2_paired_paths.json`. What is missing there is a table, not a file of ours.

**Plan Y's crossing was the second, and has now been ported.** It asks what reading the report at one hour
instead of six is worth to a model that already reads every status the six-hour report publishes. The run on
the main build is still published as `league/scripts/cutoff_f7_main_build.py`, unchanged, with its result in
`results/paper2_cutoff_f7.json`, and it reads the originating project's paths. Beside it,
`league/scripts/cutoff_f7_public.py` asks the same question of this rebuild, to a plan fixed before the code
(`docs/plans/paper2_BB_f7_public_path.md`, hashed inside the script), and writes
`results/paper2_cutoff_f7_public.json`. Its two controls passed: the out-or-not arm reproduces the published
cutoff curve at both cutoffs to the fourth decimal, and the rolling sit-out rates agree exactly with a direct
count. On this build the five hours are worth 0.059 (0.025-0.095) to a model that reads the report as
out-or-not and 0.021 (-0.001 to 0.043) to one that reads it in full, a difference inside one set of draws of
-0.038 (-0.061 to -0.018) against the main build's -0.037 (-0.060 to -0.016). The rule the plan fixed in
advance is met, so the ordering of the two encodings does not depend on the main build's roster attribution,
its tip-off times or its name matching. One reading does not carry: the full reader's own contrast reaches
zero here where on the main build it does not, so what this corroborates is the difference between the
encodings and not that waiting is worth something resolved to a model that already reads the report in full.
The encoding both scripts read the report with is `src/cutoff_encoding.py`, with the thirteen cases it was
built against in `league/tests/test_cutoff_encoding.py`, among them that a report printed exactly on the
cutoff counts, that a team which had not filed is unknown rather than healthy, and that a player the report
does not list is charged nothing even though "Available" carries a sit-out rate of its own.

**The roster audit is here as a result, not as a worksheet.** The paper's roster rule keeps a player on his
last known team's list until something moves him off it, and the error that rule can make --- a player kept
after he has left --- is not visible to a recall check. Plan AA drew 150 stints, one (player, team, season)
each, in four strata fixed before any was read, including a control stratum the selecting proxy does not
flag, and an author judged each against the league's transaction record. The two scripts are published as
they ran, with the plan they were run to (`docs/plans/paper2_AA_roster_audit.md`, hashed inside each) and
their results in `results/paper2_roster_audit_sample.json` and `results/paper2_roster_audit_result.json`.

The worksheet is not here and will not be: 150 named players with a team, a season and the absence value
each carries is a player-level table, and the source terms forbid redistributing one. Unlike the other
player-level analyses, a rebuild cannot reproduce it either, because what it holds is human judgements about
transactions rather than anything derivable from the sources. The result files carry no player rows: the
population counts, the strata and the seed in the first, and in the second the error rates by stints, by
rows and by absence value, the control stratum's rate beside them, both readings of the unconfirmable, and
the corrected build. 26 of the 150 had left. Reweighted to the flagged population that is 32.6%
(22.6-42.7) of stints; counting only the absences at or after the day each departure took effect, and only in
the rows where the proxy can be read, 27.0% (17.9-36.0) of those rows and 20.3% (12.7-28.7) of their
absence value, which is 6.0% (3.8-8.6) of that row set as a whole. In the control nothing had left, so the
proxy does enrich confirmed departures; and correcting the 26 moves the pooled premium from 0.059 to 0.058, a move of 0.0009.
On the benchmark sample the premium moves by 0.0001 and the never-listed put-back, where 428 of the 429
corrected rows sat, by 0.0004. All three are smaller than the Monte-Carlo error of a reported bound,
0.006, which is the threshold rule AA-a fixed in advance; the moves are reported rather than read as
equivalence, since more bootstrap replicates would shrink that error and leave the moves where they are,
and the correction is of the confirmed stints and not of the population. The audit is not an upper bound on
the population's error: the control found no departure in 30 stints, which is consistent with a stint rate
up to 9.5%, and the 21 `other` stints are counted as no error throughout although some hold roster artefact
that no single departure date describes.

Three results differ on this build and the scripts say so rather than leaving you to notice. Rescaling the raw
minutes back to 240 does not return this build's reference minutes for 32,118 of 182,004 rows, where on the
paper's build all but ten do, so the no-rescale ablation is less clean here. The fixed-weight put-back of
the never-listed absences is -0.016 (-0.048 to 0.016) here, not resolved, where the paper's build has it
resolved below zero. And under the plus-minus valuation the premium at six hours is -0.026 (-0.049 to 0.001)
here, which reaches zero, where the paper's build resolves it below zero at -0.027 (-0.048 to -0.003); the
reading that matters -- that the premium is negative under that valuation and that neither mechanism test
supports the account there -- is the same on both builds.

`replacement_mechanism.py` is the one that answers the question the paper's Section 4.6 ends on. The two tests
agree with the account at replacement level, where the put-back of the late Out designations is 0.040
(0.004-0.077) and reading the report an hour out rather than six is 0.043 (0.013-0.072), and they do not under
the plus-minus valuation, where the same two are 0.008 (-0.013 to 0.029) and -0.003 (-0.018 to 0.011). What
holds under all three is the direction: the premium falls as the cutoff moves later on every one of them.

### The deployment gap, pooled

The paper's model grid measures the deployment gap -- what a model loses between its own backtest and the
accuracy it delivers when handed only the pregame report -- on one holdout season, as a range across settings.
That range is not an estimate, so the paper also pools the gap over the four rolling holdouts, and this script
reruns it here:

```
python league/scripts/deployment_pooled.py    # -> results/paper2_deployment_pooled_public.json
```

On the released build the gap is 0.086 points of margin RMSE (0.035 to 0.137) for these four seasons, against
0.068 (0.015 to 0.120) on the main build; both exclude zero. Neither build resolves it as a statement about an
NBA season in general -- 0.084 (-0.052 to 0.220) here, 0.065 (-0.072 to 0.203) there -- because the gap is not
steady across seasons, and 2025-26, the season the grid uses, is the largest of the four on both builds. The
implied season-to-season standard deviation is 0.068 on each, close to the 0.0675 the inflation carries. The
narrow feature set runs here; the wide one needs the home and away columns the released file does not carry.

## The literature review's own records

Section 7 of the paper counts how published work handles availability: of 113 included studies, 11 use
availability information, none could be confirmed to use a timestamped official report with a stated
cutoff, 17 use statistics of the game being predicted, and the leakage risk reads 24 high, 24 low and 50
unclear. Those counts are the paper's case that the problem is general rather than hypothetical, so the
records behind them are here and not summarised.

| File | What it holds |
| --- | --- |
| `results/paper2_litreview_search_log.csv` | The five OpenAlex queries as strings, with the database, the date, hits, records retrieved, the de-duplicated total and how many passed the basketball-game-prediction screen. |
| `results/paper2_litreview_coded.csv` | The main search's 351 screened records, each with its OpenAlex id, DOI, title, year, venue, the screening decision and the reason, and the coding for the 74 that were included. |
| `results/paper2_litreview_studies.csv` | All 113 included studies across the five sources (74 from the main search, 21 from two remedial re-screens, 11 from citation tracking, 7 from Google Scholar), with the coding for the 98 that could be coded. |
| `results/paper2_litreview_irr_second_coder.csv` | The second coder's records for the 22 studies recoded against the same codebook. |
| `results/paper2_litreview_irr_agreement.json` | The agreement and Cohen's kappa per field, with every disagreement listed. |

Three things to know before reading them.

**The join key.** `paper2_litreview_irr_second_coder.csv` keys on `sid`, which is the main search's own
numbering. `paper2_litreview_studies.csv` carries it as `irr_sid`, filled for the 74 main-search studies
and blank for the rest. Do not join on the number inside `ref`: the namespaces collide, and `OA-sid#292`
and `OA#292` are different studies.

**What the second pass covers.** The 22 recoded studies are a 30% random sample of the 74 from the main
search, so the 39 added by the re-screens, citation tracking and Google Scholar were coded once and not
checked by a second reader. The sampled fraction is of the main search, not of the 113.

**What the kappas mean for the counts.** Agreement is perfect on factual fields --- whether a study
compares itself with the market reads kappa = 1.00 --- and poor on the judgement that carries the most
weight: leakage risk reads kappa = 0.38 on 0.59 raw agreement. The paper reports the leakage-risk counts as
one coder's reading checked by a second pass rather than as an independently coded variable, and a reader
who disagrees with a particular study's risk can now find that study and that judgement and say so.

## Licences and terms

- **Code:** MIT (`LICENSE`).
- **Our derived tables** in `data/` and `results/`: CC BY 4.0 (`LICENSE-DATA`).
- **Sources keep their own terms.**
  - Official injury reports are published by the NBA (NBA.com Terms of Use).
  - The box-score dataset is *NBA Dataset: Box Scores and Stats (1947 – Today)* by Eoin A. Moore on Kaggle, CC0,
    version 515, derived from NBA.com. The CC0 dedication covers the uploader's compilation; it does not waive the
    NBA.com terms that apply to the underlying data.

Users who rebuild locally are responsible for complying with the source terms.

## Citation

See `CITATION.cff`.
