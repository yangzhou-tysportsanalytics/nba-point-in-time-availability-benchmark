# Plan AA: a human audit of whether a player counted absent was still on the team

Written 2026-10-20, after three referee reads in a row put this ahead of any further modelling. The sample
is drawn by a seeded script from rules fixed here, the fields the auditor fills are fixed here, and the
analysis that reads them is fixed here, before a single row is judged.

## 1. The question the existing audits do not answer

Two checks exist. The parsing audit asks whether the right PDF was selected and read correctly. The roster
recall check asks whether a player who actually appeared was on the team's candidate list, and answers above
99.9%. Neither asks the opposite question:

> A player kept on the list, carrying reference minutes, and counted absent by the participation feature ---
> was he still on that team at that cutoff?

The proxy for it is a reference-rotation player who never appears for that team again in the season. It is
not an error on its own, because a season-ending injury looks identical, and Section 2.3 reports it as a
negative result: the proxy is `\nRosterProxyRatio` times as common within seven days of a transaction
involving the team as away from one, an elevation the proxy cannot tell from injuries clustering near the
same dates.

The scale is why this matters more than another model class. The flagged rows carry
`\nRosterProxyValueShare` of the realized-participation feature's total absence value, against a premium of
`\nNoFbInflPt`. A correction a tenth that size would move the quantity this paper is about.

**The proxy may not become a roster rule.** "Never appears again" reads the rest of the season, so a build
that used it would not be point-in-time. It is a sampling device for this audit and a post-hoc sensitivity,
and nothing else.

## 2. The unit, and why it is not a row

A player who has left a team carries one error, not one per game: every remaining game of that season
inherits it. Auditing rows would spend the auditor's time re-judging the same fact. The unit is therefore a
**stint** --- one (player, team, season) --- judged once, and the verdict is mapped back onto every row of
that stint when the rates are computed, which is what makes a value-weighted rate possible.

## 3. The sample

Drawn by `league/scripts/p6_ds_roster_audit_sample.py` at seed 20261020 from the same rows and the same
proxy as `results/paper2_roster_precision.json`, over the benchmark games with no missing block.

| stratum | definition | stints |
|---|---|---|
| **A** | flagged, in the top decile of absence value carried by the stint | 50 |
| **B** | flagged, first flagged game within seven days of a transaction involving the team | 40 |
| **C** | flagged, the remainder, drawn at random | 30 |
| **D** | **control**: not flagged, and the player is counted absent in at least one game of the stint | 30 |

150 stints. Stratum D is what makes the others readable: without it a non-zero error rate among flagged
stints says nothing about whether the proxy selects errors.

The worksheet carries the player, the team, the season, the first and last flagged game dates, the number of
rows and the absence value the stint carries, whether it is near a transaction, and its stratum. It carries
no indication of what the answer is expected to be.

## 4. What the auditor fills

- `verdict`: one of `still_on_team`, `had_left`, `other`, `cannot_confirm`.
- `left_date`: for `had_left`, the date the player ceased to be on that team, to the day.
- `source`: a URL or citation for the judgement.
- `note`: free text, required for `other` and `cannot_confirm`.

`cannot_confirm` is a real answer and is never to be recorded as `still_on_team`.

## 5. The analysis, fixed now

Computed by a second script once the worksheet is returned, and reported whatever it says:

1. The share of stints, of rows, and **of absence value** judged `had_left`, within each stratum and
   reweighted to the flagged population, each with a bootstrap interval over stints.
2. The same with `cannot_confirm` counted as an error and as not an error, reported as a range, so the
   unknowns keep their own denominator rather than being assumed correct.
3. The control comparison: stratum D's rate against the flagged strata, with the difference and its
   interval.
4. A corrected build: every row of a stint judged `had_left`, from `left_date` onward, removed from the
   participation feature's absence set, and the pooled headline, the strict headline and the never-listed
   put-back rerun on it.

## 6. The rules, fixed now

- **AA-a.** If the corrected premium differs from the published one by no more than the Monte-Carlo error of
  a bound (`\nMCmax`), the audit is reported as a bound on this error and the headline does not move.
- **AA-b.** If it differs by more than that, both are reported and Section 2.3 states the corrected figure
  beside the published one.
- **AA-c.** If it differs by more than half the premium itself, roster precision becomes a first-order
  limitation rather than a listed one: the abstract says the premium is sensitive to it, and the corrected
  build carries the headline.
- **AA-d.** If stratum D's error rate is not separable from the flagged strata's, the proxy does not select
  errors, Section 2.3 says so, and the flagged share stops being used as an upper bound on anything.
- **AA-e.** If more than a third of the sampled stints come back `cannot_confirm`, the audit is reported as
  inconclusive and no corrected build is published from it, because a correction built on two thirds of a
  sample selects on confirmability.
- **AA-f.** No published number moves on the strength of this plan alone; the corrected build is a new
  result file beside the old, never a rewrite of it.

## 7. What this cannot do

It bounds one error in one direction on a sample. It does not establish the roster rule is right elsewhere,
it cannot see a player wrongly removed from a list, and `left_date` to the day will sometimes be a judgement
about when a trade became effective rather than when it was announced. Those limits go in Section 2.3 with
the result.
