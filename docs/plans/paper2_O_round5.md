# Plan O — answering review round 5 (written 2026-09-25, before running)

Round 5 (`docs/paper/paper2_review_round5_2026-09-24.md`) read the version this session produced. Its verdict — "the
method is converging, the numeric hygiene is regressing" — is fair: several of its findings are errors this session
introduced. This plan fixes the computational half. Reading rules are fixed here, before anything is run.

## O1. One canonical run per quantity
A single script produces the headline quantities: per-season and pooled (F0, F1, F2) on the rolling origins, under
the **two-way (pigeonhole) team bootstrap at 10,000 replicates**, percentile intervals, with the Monte-Carlo standard
error of every bound. Table 11 is regenerated from it, caption included. The home-cluster and seasons-then-clusters
runs stay in the supplement as comparisons, each labelled.

*W-O1.* After this, any interval in the paper is either from this run, or carries its design in the sentence. No
quantity appears with two intervals without both being named. If the two-way per-season intervals are wider than the
home-cluster ones already printed, the table shows the two-way ones and the text describes the per-season estimates
as unresolved, whatever that does to the season story.

## O2. The declared design applied where it matters
The model-class comparison (least squares, LightGBM, neural net) and the season trend are recomputed under the
two-way design.

*W-O2.* If the model classes' intervals all contain zero under two-way, the paper says so and drops "two of the three
intervals exclude zero". If the season trend's two-way interval contains zero, the trend becomes descriptive and the
p-value is removed. Neither outcome is allowed to be reported under the narrower design instead.

## O3. The put-back tests on the headline's base feature (round 5, 1.3)
The put-back tests and the joint regression are rerun with the headline compliant feature F1 (six hours with the
90-minute fallback, the feature the whole paper uses) as the base, on the pooled 4,846 holdout games. The six-hour
no-fallback version is reported beside it as the sensitivity it always was.

*W-O3.* The text reports F2 − F1 = the headline inflation as the reference quantity, and the put-back contrasts
against the same F1. If putting a component back is worth more than the headline inflation, that is reported as a
contradiction of our own account, not as a finding. The F7 comparison (ex-ante weighting of listed statuses, worth
+0.028) and the late-status put-back (realised sitters) are reported side by side with the statement that we cannot
say which is the better use of the same information.

## O4. A split that does not condition on the outcome (round 5, method 2)
Section 3.3's absolute-margin terciles are recomputed on the **pregame predicted** margin of the no-availability
model F0, and on the pregame closing spread where it exists.

*W-O4.* The predicted-margin split replaces the realised-margin split in the paper. If the pattern disappears under
the predicted-margin split, the sentence "the leak sits in lopsided games" is withdrawn from the abstract and the
paper says the RMSE/log-loss divergence is unexplained.

## O5. The ladder rebuilt on one sample and one training set (round 5, 1.5)
All four rungs are refitted on the 977 lined holdout games of 2025-26, trained on the 2021-22 and 2022-23 games that
carry both prices, so the rungs differ only in the baseline. The table's RMSE column is defined as the holdout RMSE
of that rung's no-availability model.

*W-O5.* If the ladder cannot be built on one sample and one training set without losing the opening-line rung, the
ladder framing is dropped and the two market baselines are reported in Section 5 only.

## O6. The market analysis on a second holdout season (round 5, method 7)
M2 (line as a feature plus availability) is replicated on the 2022-23 holdout, whose consensus closing prices exist,
trained on 2021-22.

*W-O6.* Both seasons are reported, whatever they show. If they disagree in sign, the market section says the
comparison rests on two seasons that disagree.

## O7. The betting intervals (round 5, 1.7)
The win-rate intervals are recomputed with the bet set recomputed inside each bootstrap draw, clustered on the home
team, and the method is stated in the text with the binomial width beside it.

*W-O7.* If the corrected interval is wider than the binomial width, that is what the paper prints.

## O8. What this plan does not do
It does not cut the paper to 15-18 pages, replace Figure 4A, or remove the process narration; those are the
presentation pass that follows the Sloan template. It does not add an external impact metric: none is held under a
licence we can release, and the paper will state that absolute magnitudes are specific to our valuation, with the
rescaling sensitivity (0.059 to 0.033) promoted out of the footnote-like sentence it now sits in.
