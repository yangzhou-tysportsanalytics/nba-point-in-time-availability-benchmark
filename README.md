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
| `league/tests/` | Tests for the public path. |
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
  seasons, then pooled (4,846 games). Inflation is RMSE(F1) − RMSE(F2). Intervals resample both teams (a two-way
  team bootstrap); home-team×season cluster intervals are reported beside them in the paper.

## Key results

Pooled over the four rolling-origin holdouts, two-way intervals:

| Quantity | Estimate |
| --- | --- |
| Value of the pregame report, RMSE(F0) − RMSE(F1) | **0.197 (0.084–0.304)** |
| Inflation from actual participation, RMSE(F1) − RMSE(F2) | **0.059 (−0.032 to 0.152)** |
| The same two on the public rebuild (4,835 games) | 0.196 (0.088–0.299) and 0.074 (−0.015 to 0.164) |

The report's own value is the stable part. The inflation is not:

- **It depends on how an absent player is valued.** Under a plus-minus valuation the report is worth 0.051
  (0.001–0.107) and the inflation is −0.030 (−0.067 to 0.004) — the other side of zero. Without the 240-minute
  rescaling they read 0.205 (0.093–0.309) and 0.038 (−0.042 to 0.122).
- **It depends on the cutoff.** On the games where both teams had filed at six hours, the report is worth 0.185
  (0.067–0.296) six hours out and 0.258 (0.124–0.389) thirty minutes out. Read at one hour instead of six, on the
  same games, the compliant feature gains 0.067 (0.010–0.122); at thirty minutes, 0.073 (0.001–0.141).
- **The twelve-hour point needs care.** On the games where both teams had actually filed that early, the report is
  worth 0.153 (0.026–0.265) and the inflation is 0.054 (−0.052 to 0.159). Computed on all games, imputing health
  for the teams that had not filed, the same pair reads 0.108 and 0.128 — which is the mistake this paper is about,
  and the reason the first version of that point was wrong.
- **A stronger compliant baseline closes part of it.** Weighting every listed status by its sit-out rate (F7) is
  worth 0.224 (0.111–0.340) against F1's 0.197, and the inflation measured against F7 falls to 0.031 (−0.050 to
  0.108).
- **Which absences carry it.** The difference between the two features decomposes exactly into absence declared Out
  after the cutoff, absence listed later but not as Out, and absence on no report we hold, minus players Out at the
  cutoff who played. A retrospective put-back of the first is worth 0.070 (0.006–0.131); the difference that then
  remains from the participation feature, −0.012 (−0.063 to 0.041), is not resolved. The other two put-backs buy
  nothing measurable.
- **Who played, not how long.** Once participation is known, realized minutes add nothing (−0.011, −0.048 to 0.022).
- **No market edge.** No model built from availability beats the closing line, and a model handed the closing price
  shows no improvement from adding availability.

Single-season figures, for comparison with the public build (2025-26 holdout, two training seasons, home-team
cluster intervals — the design those runs used):

| Result | Public build (this repository) | Paper's main build |
| --- | --- | --- |
| Value of the pregame report | 0.213 (0.078–0.340) | 0.226 (0.105–0.351) |
| Inflation from actual participation | 0.176 (0.079–0.271) | 0.166 (0.077–0.255) |

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
