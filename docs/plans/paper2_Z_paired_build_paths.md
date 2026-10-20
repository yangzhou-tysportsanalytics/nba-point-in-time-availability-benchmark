# Plan Z: the two build paths compared on one sample, with only the availability build swapped

Written 2026-10-20, after a fourth cold referee read pointed out that Section 2.4 says the comparison has not
been made and Section 3.2 then drew a conclusion from it. The numbers in section 1 were known before this
plan; the result in section 3 is not, and the rules in section 4 are fixed before any code is written, with
the branch that costs the paper a sentence written first.

## 1. What is wrong with the comparison as it stands

`results/paper2_path_compare.json` restricts both paths to the game identifiers they share and runs the same
rolling design on each. That removes most of the sample difference and not all of it, and it does not pair:

| | |
|---|---|
| games both paths keep | **4,833** |
| kept by the main build only | 6 |
| kept by the released path only | 2 |
| the main build's own sample | 4,839 |
| the released path's own sample | 4,835 |

Each path's premium is read on **its own** sample --- `0.057 (-0.002 to 0.116)` on ours, `0.074
(0.012-0.136)` on theirs --- and their difference, `0.017`, is formed between two point estimates computed
on two different sets of games under two separately drawn bootstraps. Section 2.4 now says that the
comparison with the Monte-Carlo error of a bound is a comparison of magnitudes and not a test, and names the
paired bootstrap as not run. Section 3.2 is less careful and this plan exists to settle which of them is
right.

What differs between the two paths, measured on the 6,146 joined games and already in that result file:

| column | share identical | sd of the difference |
|---|---|---|
| `margin` | **1.000** | 0.000 |
| `rest_diff` | 0.998 | 0.061 |
| `strength_diff` | 0.979 | 0.310 |
| `F1_diff` | 0.023 | 3.761 |
| `F2_diff` | 0.000 | 5.473 |

So the outcome is identical, the controls nearly so, and the availability features differ almost everywhere.
That is what makes a clean swap possible.

## 2. What is computed

`league/scripts/p6_ds_paired_paths.py` writes `results/paper2_paired_paths.json`

1. The 4,833 games both paths keep, joined on season, date and the two teams, because the two tables carry
   different identifiers and share none.
2. One outcome, one pair of controls and one training set for both arms: the main build's `margin`,
   `strength_diff` and `rest_diff`, and the same rolling-origin split, so that the **only** thing that
   differs between the arms is the pair `(F1_diff, F2_diff)`.
3. Each arm's gain and premium on that sample, and **the difference of the two premiums inside the same
   bootstrap draw**, under the design of Section 3.1 with the two-way and seasons-then-clusters readings
   beside it, 10,000 replicates, seed 20260921.
4. The same for the gain, which the paper reports as reproducing, so that the claim is tested rather than
   asserted.
5. A sensitivity that swaps the controls as well as the features, since the controls are themselves a
   construction difference, and the difference between the two sensitivities.
6. Counts: how many of the 4,833 games have a different `F1_diff`, a different `F2_diff`, a different
   `strength_diff`.

`src/paired_build.py`, new, holds the two pieces that can be wrong quietly and is built against
`tests/test_paired_build_rules.py`, written first: aligning two tables that share no identifier, and forming
a paired contrast that uses one draw for both arms.

## 3. The rules, fixed now

- **Z-a (control).** On the main build's own full sample, with its own features, this script must reproduce
  the published pooled gain and premium, `0.197` and `0.059`, at the fourth decimal. If it does not, the
  plumbing is wrong and nothing is written.
- **Z-b (alignment control).** The two arms must have identical length, identical game order and identical
  outcome vectors, checked by assertion rather than by eye. A single misaligned row would make the pairing
  meaningless while leaving every number plausible.
- **Z-c (written first).** If the paired difference of the premiums contains zero, then the two paths cannot
  be ordered on this evidence: Section 3.2's "it is the feature construction that carries the difference"
  goes, Section 2.4 says the two causes cannot be separated even on a fixed sample, and the paper reports
  the difference as unresolved. This is the branch that costs the paper a sentence it currently has.
- **Z-d.** If the paired difference excludes zero, the paper may say that on identical games, training,
  outcome and controls the released construction gives a larger premium than ours by the stated amount, and
  that the sample difference is not what carries it. It must still say that this does not establish which
  construction is right.
- **Z-e.** If the gain's paired difference excludes zero, the claim that the report's value reproduces
  across paths is narrowed to the size of that difference rather than left as a reproduction.
- **Z-f.** If swapping the controls as well moves the premium difference by more than `\nMCmax`, both
  readings are reported and neither is called the answer.
- **Z-g.** No published number moves. New script, new result file.

## 4. What the paper then says

Section 2.4 replaces the sentence naming this test as not run with its result, under either branch. Section
3.2's sentence is corrected to match. The `work_claims` entry recording that the test has not been run is
removed from the self-audit registry, because it will then be false.
