# Plan Y: what reading the report later is worth once the early report is read in full

Written 2026-10-19, after a third cold referee read pointed at the cell this paper has never computed. No
number in section 3 is known: this is the first plan in the series whose result is not already in a
scratchpad. The rules in section 4 are fixed before any code is written, and the branch that costs the
paper's current wording is written first.

## 1. The gap

The paper has three of the four cells of a two-by-two in the report's encoding and the cutoff:

| encoding | six hours | one hour |
|---|---|---|
| F1, out-or-not | 0.185 (0.126--0.245) | **+0.067 (0.030--0.103)** over six hours, paired |
| F7, all statuses | 0.220 (0.152--0.292) | **never computed** |

The cutoff curve of Section 4.5 and the paired contrasts of Section 4.7 are built on F1 alone: they re-read
the **Out** status at each cutoff and hold everything else at its six-hour value. F7 already uses the
statuses short of Out. So the paper cannot answer the question a model owner actually faces:

> If the six-hour report were read in full, how much would waiting until one hour before tip-off still add?

Of the 0.067 that waiting buys under an out-or-not reading, some is a player moving from Questionable or
Doubtful to Out --- a transition a full-status reading at six hours has already priced at that status's
sit-out rate --- and some is information that did not exist at six hours, a player added to the report or
ruled out having never been listed. Those two have opposite implications for practice and the paper
currently cannot separate them. Section 3.4 shows that the encoding matters for the premium; nobody has asked
whether it matters for the cutoff.

## 2. Why the existing code cannot answer it

`league/scripts/p6_ds_cutoff_curve.py` selects, per team-game and per cutoff, the last report filed by then,
and marks a reference-rotation player absent if that report calls him Out. `f1_at(hours)` returns one value
per team-game. `paired(late, early)` differences two RMSEs inside the same bootstrap draws.

`league/scripts/p6_ds_pooled_secondary.py` builds F7 from `bench_ref_players.csv`'s `status` column --- the
player's status on the report selected at six hours --- weighted by that status's sit-out rate among
reference-rotation players, estimated with a Beta(1,1) prior from games that ended before the row's cutoff.

Neither reads a non-Out status at a cutoff other than six hours. The report series
(`full_series/injury_rows.csv`) carries every status of every row, so the information is there.

## 3. What is computed

`league/scripts/p6_ds_cutoff_f7.py`, new, writing `results/paper2_cutoff_f7.json`

1. `q_six`: the sit-out rate per (game, status) at the six-hour cutoff, built exactly as
   `p6_ds_pooled_secondary.py` builds `q_roll`, and stored as a map so that it can be applied at any cutoff.
2. `f7_at(hours)`: per team-game, the sum over reference-rotation players of
   `base_minutes * eff_prior * q_six(game, status)` with the status read from the report selected at that
   cutoff and `not_listed` contributing zero.
3. `f1_at(hours)`, copied from the cutoff-curve script unchanged, so the two encodings are read from the same
   report selection and the controls below can be exact.
4. The paired contrast `F7@1h - F7@6h`, inside the same bootstrap draws as `F1@1h - F1@6h`, on the strict
   sample the cutoff curve uses (both teams filed at six hours, 4,131 holdout games), 2,000 replicates, seed
   20260921, home-team x season clusters with the two-way and seasons-then-clusters readings beside it.
5. The same on the archive-coverage subsample `_covered(1)`: the games where a report printed after the
   one-hour moment is held for both teams, because the archive stops 90 minutes before the day's last
   tip-off and the published 0.067 is subject to that limit.
6. **The difference of the two paired contrasts**, `(F7@1h - F7@6h) - (F1@1h - F1@6h)`, inside the same
   draws, because that is the quantity the question asks for and comparing two intervals is not it.
7. A sensitivity with the sit-out rates re-estimated at the one-hour cutoff rather than held at six hours.
8. Counts, for interpretation rather than inference: how many team-games have a player whose status changes
   between the two cutoffs, split into transitions among listed statuses and players not listed at six hours
   at all.

## 4. The rules, fixed now

**Controls, before anything new is reported.**

- **Y-a.** The script's own F1 path must reproduce `results/paper2_cutoff_curve.json` exactly: gain 0.1852
  and inflation 0.0502 at six hours on 4,131 games, and the paired one-hour contrast 0.0666 with home-cluster
  interval (0.0305, 0.1031). If any of those differs at the fourth decimal, nothing is written.
- **Y-b.** The script's F7 at six hours, built from the re-read report series rather than from the build's
  status column, must reproduce the published strict F7 gain of 0.2198 to within 0.002. If it does not, the
  two status paths disagree and that is a defect to report rather than a contrast to interpret: the paired
  F7 numbers are written to the result file, the paper says nothing from them, and the discrepancy goes in
  the analysis history.

**The result.**

- **Y-c (written first, and the branch that costs the paper its current wording).** If the F7 paired
  contrast reaches zero, or its point estimate is below half the F1 one, then most of what waiting buys is
  statuses hardening rather than news, and a model that reads the six-hour report in full gets most of the
  benefit without waiting. Then: Section 7's cutoff rule is rewritten to put reading the report in full
  ahead of reading it later; the abstract's and Section 4.5's 0.067 are labelled as specific to the
  out-or-not encoding; and Section 4.7's "reading the report later improves the forecast" keeps its interval
  and loses its generality.
- **Y-d.** If the F7 paired contrast excludes zero and its point estimate is within 0.02 of the F1 one, then
  waiting carries information a full reading at six hours does not, the cutoff advice stands as written, and
  the paper adds that it is not an artefact of the out-or-not encoding --- which is a strengthening, and the
  outcome I would prefer, which is why Y-c is written above it.
- **Y-e.** Neither reading may be taken from the difference of the two intervals. Rule Y-c and Y-d are
  decided on the paired difference of item 6, and if that difference contains zero the paper says the two
  encodings cannot be ordered at this cutoff, whichever way the point estimates fall.
- **Y-f.** Sample. If the full sample and the archive-coverage subsample disagree on whether an interval
  excludes zero, the coverage-restricted one carries the reading and both are printed, as Section 4.5
  already does for the late end of the curve.
- **Y-g.** Rates. The held-at-six-hours version is the primary, because it is the cutoff curve's own
  convention --- hold the valuation, re-read the status. If the re-estimated version differs from it by more
  than the Monte-Carlo error of a bound, both are reported and neither is called the answer.
- **Y-h.** No published number moves. This is a new script and a new result file; nothing already in
  `results/` is rewritten.

## 5. What the paper then says

A paragraph in Section 3.4, where the encoding is introduced, and a sentence in Section 7, where the cutoff
advice is given. Under Y-c the practice section changes its order of operations; under Y-d it gains a
sentence saying the cutoff result survives the stronger encoding. Either way Section 4.5's curve stays as it
is, built on F1, with the new cell named as the one point of it computed under the other encoding.
