# Plan BB — plan Y's crossing, on the public rebuild path

Fixed before the code was written. The rules below, and in particular which reading counts as corroboration
and which does not, are written out here so that the run cannot choose them afterwards.

## 1. Why

Plan Y crossed the paper's two compliant encodings with its two cutoffs: what is reading the report at one
hour instead of six worth to a model that already reads every status the six-hour report publishes? On the
main build the answer was that most of it is status hardening the full reading has already priced ---
F1 gains `+0.0666 [0.0305, 0.1031]` from waiting, F7 gains `+0.0297 [0.0057, 0.0528]`, and the paired
difference is `-0.0369 [-0.0600, -0.0157]` under the reported design.

That result is the one analysis in the release with **no public-path control**. Every other player-level
result in the paper has one, because `rebuild.py build` writes the player-level table on the reader's own
machine from official injury-report PDFs and the pinned CC0 box scores, and the released scripts read it from
there. Plan Y's script reads the originating project's paths instead, so nothing independent has ever
produced its number. The release says so in as many words: *porting the analysis onto this rebuild is the
obvious next step and has not been done.*

This plan does the port. The public rebuild is a different build, not a copy: its tip-off is the scheduled
time in the Kaggle game table, its roster at the cutoff comes from box appearances with no transaction log,
and its own eligibility rule leaves 4,117 games against the main build's 4,131. So this is a **control on a
second build, not a replication**, and §6 says what each outcome licenses.

## 2. Inputs, and what is held fixed

Everything comes through the released loaders, so that a reader can run this script beside the others:

  * `public_inputs.ref_players()` --- the reference-rotation player-games the rebuild writes, 182,004 rows
    with `base_minutes`, `eff_prior`, `status`, `out_real`;
  * `public_inputs.games()` --- the game-level benchmark, with `F2_diff`, `margin`, the two controls and
    `any_block_missing`;
  * `rebuild.status_at(..., cut)` --- the status of every listed player on the last report filed at or
    before a cutoff, and which team-games had filed. Nothing here reimplements the build.

Following the cutoff curve, so that the two can be compared: the rotation, the reference minutes and the
prior efficiency are the six-hour build's, and **only the status is re-read** at the later cutoff. The
per-status sit-out rates are likewise held at their six-hour values (the main build's rule Y-g), with the
re-estimated version reported beside them as a sensitivity.

The encoding is `src/cutoff_encoding.weight`, unchanged and already tested in this repository, so the two
builds read a report through the same thirteen cases.

## 3. The sample

The cutoff curve's rule, applied identically: team-games where both teams had filed by six hours, minus the
games the build already excludes for a missing block. That is 4,117 games on this build. The eligibility rule
is applied to the training seasons as well as the holdout, because an earlier external review found that
restricting only the holdout leaves teams that had not filed, and therefore imputed health, inside the fit.

## 4. The quantity

For each encoding, the paired contrast `RMSE(six hours) - RMSE(one hour)` inside the same bootstrap draw, and
then the difference of the two contrasts inside one set of draws:

    (F7 at six - F7 at one) - (F1 at six - F1 at one)

A difference of two intervals is not the interval of the difference, which is why it is computed in one
draw. Reported under three designs --- `home_season_cluster` (the paper's), `two_way_team`, and
`season_then_cluster` --- with the first the one the rules below are read on.

## 5. Controls, fixed now

  * **BB-a.** The F1 arm must reproduce the published public-path cutoff curve at both cutoffs: the gain and
    the premium at six hours and at one hour, each to `1e-4`, on the same 4,117 games
    (`results/paper2_cutoff_curve_public.json`). If this fails, the feature construction differs from the
    published one and **nothing below is interpreted**.
  * **BB-b.** The rolling sit-out rate table must agree, for every status, with a direct count: among
    reference-rotation player-games of that status whose game ended before the cutoff, the share who did not
    play, under the same Beta(1,1) prior, to `1e-9`. This is the one piece of machinery with no published
    counterpart, so it is checked against a definition rather than against a number.
  * **BB-c.** The five statuses this build records must be the five the encoding carries rates for. A status
    the league has not used before must raise rather than score zero, which the shipped tests already fix.

## 6. The rules, unfavourable branch first

Read on `ci95_home`, the design the paper reports.

  * **BB-d --- not corroborated.** If the paired difference's interval contains zero, **or** its point
    estimate is positive, then plan Y's finding does not carry to the public build, and the paper must say
    that the one place the two builds were crossed did not agree. A negative point whose interval contains
    zero is this branch, not a weaker version of the next one: it is the same sign and an unresolved
    difference, and it is reported in those words.
  * **BB-e --- corroborated.** Only if the point estimate is negative **and** the interval excludes zero is
    the main build's reading reported as holding on a second build. The two builds' point estimates are not
    expected to match to the decimal and no rule is read off their distance.
  * **BB-f --- the rate convention.** If re-estimating the sit-out rates at one hour flips the sign of the
    difference, the result is reported as sensitive to a convention rather than as a finding, whichever
    branch above it fell into.
  * **BB-g --- F7's own level.** The public path's F7 gain at six hours is reported beside F1's, with no rule
    attached: the two builds' levels differ for reasons the paper already documents, and this plan is about
    the contrast, not the level.

## 7. Output

`results/paper2_cutoff_f7_public.json`, carrying this plan's SHA-256, the controls with their verdicts, the
three designs, the sensitivity, and the status transitions between the two cutoffs for interpretation.
The script is `league/scripts/cutoff_f7_public.py` in the public repository, which is where it runs.

## 8. What this does not settle

The two builds share their sources, so agreement is not independent evidence about the league --- it is
evidence that the result does not depend on the main build's roster attribution, its tip-off times or its
name matching. Those are exactly the three places an external review has raised, which is why the control is
worth running, and it is all it is worth.
