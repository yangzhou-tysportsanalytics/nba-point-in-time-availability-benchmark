"""Plan Y: reading the official report at a cutoff under either of the paper's two compliant encodings.

The cutoff curve of Section 4.5 re-reads the **Out** status at each cutoff and holds the rotation, the
reference minutes and the prior efficiency at their six-hour values. Section 3.4's stronger baseline reads
every status the report publishes, weighted by the sit-out rate of that status among reference-rotation
players. Those two have never been crossed: the paper knows what waiting five hours is worth to a model that
reads the report as out-or-not, and not what it is worth to one that reads it in full.

This module is the part of that crossing that can be wrong in ways a decimal would not show, so it is
separate from the script that runs it and is tested against the cases in
`tests/test_cutoff_encoding_rules.py`:

  * a report printed exactly on the cutoff counts, because 52.8% of selected team-games are;
  * a team that had not filed is unknown, not healthy;
  * a player the report does not list is charged nothing, even though "Available" carries a sit-out rate of
    its own, because that is the convention F7 is built under;
  * a status that softens between two cutoffs lowers the value rather than keeping it;
  * a status the league has not used before is an error rather than a silent zero.

The sit-out rates are passed in. The cutoff curve's convention is to hold the valuation fixed and re-read
only the status, so the script holds them at the values estimated for the six-hour cutoff; passing a
different map is how plan Y's sensitivity re-estimates them.
"""
from __future__ import annotations

NOT_LISTED = 'not_listed'
ENCODINGS = ('out_only', 'all_statuses')


def selected_time(times, cutoff):
    """The last report time at or before `cutoff`, or None if the team had not filed by then.

    At or before, not before: most selected reports are printed exactly on the cutoff moment.
    """
    best = None
    for t in times:
        if t <= cutoff and (best is None or t > best):
            best = t
    return best


def weight(status, rates, encoding):
    """The share of a player's absence value an encoding charges, given his status on that report.

    `out_only` is F1: the whole player when the report calls him Out, nothing otherwise, including for a
    status that carries a sit-out probability. `all_statuses` is F7: the sit-out rate of his status, and
    nothing at all for a player the report does not list, which is how F7 is built and is not the same as
    charging him the not-listed rate.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f'unknown encoding: {encoding!r}')
    if status == NOT_LISTED:
        return 0.0
    if status not in rates:
        raise KeyError(f'no sit-out rate for status {status!r}: the rates are estimated per status and a '
                       f'status the league has not used before cannot be scored as zero')
    if encoding == 'out_only':
        return 1.0 if status == 'Out' else 0.0
    return float(rates[status])


def team_value(players, status_of, rates, encoding):
    """The team's absence value on one report: a sum over the reference rotation, not over the report.

    `players` is (player id, value) for the reference rotation; `status_of` maps a player id to his status on
    the selected report. A rotation player the report does not mention is `not_listed`; a listed player who
    is not in the rotation is not in `players` and does not enter.
    """
    total = 0.0
    for pid, value in players:
        total += value * weight(status_of.get(pid, NOT_LISTED), rates, encoding)
    return total
