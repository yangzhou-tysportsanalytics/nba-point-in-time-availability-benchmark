# Plan M: pool the secondary results, and test three things the reviewers doubted (2026-09-24, before computing)

Three independent reviews of the 29-page draft agreed on one structural complaint: the paper's headline is pooled
over four rolling holdouts, but Sections 3.3–3.6, the model grid and Section 5 are computed on 2025-26 alone — the
season with the largest inflation. Anything that does not need a betting line can be pooled, and should be. They
also doubted three specific results. This plan fixes what is computed and how it will be read before any of it runs.

## M1. Pool what does not need a price
- **The all-status baseline (F7).** Rolling-origin holdouts, the same construction as Section 3.4: every listed
  player weighted by the sit-out rate of his status, estimated only from games that ended before the cutoff.
  Report the pooled compliant gain, the pooled inflation F7 − F2, and the per-season values.
- **Decision units.** Log loss, Brier, winner accuracy per 100 games and the compliant/leaky contrasts, pooled over
  the rolling holdouts instead of the 2025-26 holdout alone. The share of the gap to the closing line stays
  single-season, because only 2025-26 has prices.
- **The decomposition of Section 4.5**, per holdout season, so that the reader can see whether the finding is the
  2025-26 season again.

**W-M1.** If a pooled interval contains zero where the single-season one did not, the paper reports the pooled value
as the headline for that quantity and labels the single-season number as one draw, exactly as it does for the main
inflation. No quantity keeps a single-season headline merely because it is larger.

## M2. Is the "never listed" null a valuation artefact?
Two reviewers observed that the reference rotation keeps a player for up to 20 games after his last appearance, so a
player who has dropped out of the rotation keeps generating "absences" that cost his team nothing. Of the
never-listed absences, 9,427 are recorded as a coach's decision. If those rows are noise, the null we report for
that component is mechanical rather than informative.

**Test.** Recompute the never-listed component restricted to players who appeared in at least one of their team's
three most recent completed games before the cutoff, and repeat the dose-response.

**W-M2.** If the restricted component shows a rising inflation with an interval excluding zero, Section 4.5 withdraws
the claim that absences the report never confirms carry nothing, and says instead that the earlier null came from
including players who were no longer part of the rotation. If it does not rise, the paper reports the restriction as
a robustness check and keeps the claim.

## M3. Does file-name timing change the estimates, or only the paperwork?
Section 2.1 leans on the fact that a name-timed pipeline would use a report published after its own cutoff for 53.6%
of team-games, but the paper never says what that does to an estimate.

**Test.** Rebuild the compliant feature on the complete series with reports timed by file name, and rerun the
headline and the rolling design.

**W-M3.** If the name-timed estimates differ materially from the printed-time ones, the paper reports both and the
53.6% keeps its place. If they do not, Section 2.1 says so plainly: the convention matters for what a pipeline is
doing, not for what it measures here.

## M4. A baseline ladder
The compliant model carries no team-strength control, and adding Elo absorbs part of the inflation, which suggests
the leaky feature partly proxies for team strength. Report, on the 2025-26 holdout where prices exist, the report's
value and the inflation under four baselines: no control, plus Elo, plus the opening line, plus the closing line.

**W-M4.** If the inflation falls monotonically as the baseline strengthens, the paper says that part of what the leak
buys is team strength that a better baseline already has, and the honest value is what survives the strongest
baseline available.

## M5. A second worked example
The Denver--Oklahoma City game in Section 7 is the largest report-versus-outcome disagreement, but it is an example
of the component that carries *no* inflation: seven Denver players were listed Questionable and sat, and the leaky
model was the worse of the two that night. It illustrates the dataset, not the mechanism.

**Rule, fixed now.** Among the 2025-26 holdout games with a closing line, take the one with the largest absence value
ruled Out between the six-hour cutoff and thirty minutes before tip-off — the component that does carry the
inflation — and report it beside the existing example, whatever the two models do in that game.

Scripts: `league/scripts/p6_ds_pooled_secondary.py` (M1), `p6_ds_artefact_timing.py` (M2, M3),
`p6_ds_ladder_example.py` (M4, M5). Results in `results/paper2_pooled_secondary.{json,md}`,
`paper2_artefact_timing.{json,md}`, `paper2_ladder_example.{json,md}`.
