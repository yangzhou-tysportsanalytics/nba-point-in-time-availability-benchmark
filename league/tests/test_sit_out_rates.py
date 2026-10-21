"""The rolling per-status sit-out rates plan BB's control is read against.

The originating project's plan-Y script computes these inline and the number it produced is published.
This module is the same construction, written out so the public-path port can be checked against the
definition rather than against that number, and these are the cases where a rolling rate goes wrong without changing a decimal: a game that
ends on the cutoff, a status with no history yet, rows arriving out of order, a status the rate table was not
asked for, and an end time the build does not know.
"""
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.sit_out_rates import rolling_rates  # noqa: E402

STATUSES = ('Out', 'Questionable')


def reference(status, out_real, end, statuses, cutoffs, prior_a=1.0, prior_n=2.0):
    """The definition, counted one cutoff at a time. Slow on the real table and clear here."""
    out = np.empty((len(cutoffs), len(statuses)))
    for i, c in enumerate(cutoffs):
        for j, s in enumerate(statuses):
            k = [o for st, o, e in zip(status, out_real, end) if st == s and e <= c]
            out[i, j] = (sum(k) + prior_a) / (len(k) + prior_n)
    return out


class RollingSitOutRates(unittest.TestCase):

    def test_a_game_ending_on_the_cutoff_counts(self):
        # the main build selects with side='right', so the boundary is at-or-before; pinned here because a
        # silent change of side moves every rate by one game
        r = rolling_rates(['Out'], [True], np.array([100]), STATUSES, np.array([100]))
        self.assertAlmostEqual(r[0, 0], 2.0 / 3.0)
        r = rolling_rates(['Out'], [True], np.array([101]), STATUSES, np.array([100]))
        self.assertAlmostEqual(r[0, 0], 0.5)

    def test_a_status_with_no_history_is_the_prior_and_not_zero(self):
        r = rolling_rates(['Out'], [True], np.array([10]), STATUSES, np.array([50]))
        self.assertAlmostEqual(r[0, 1], 0.5)  # Questionable, never seen
        self.assertNotEqual(r[0, 1], 0.0)

    def test_rows_out_of_order_give_the_same_answer(self):
        st = ['Out', 'Out', 'Questionable', 'Out', 'Questionable']
        o = [True, False, True, True, False]
        e = np.array([50, 10, 30, 20, 40])
        cuts = np.array([25, 45, 100])
        a = rolling_rates(st, o, e, STATUSES, cuts)
        k = [3, 0, 4, 1, 2]
        b = rolling_rates([st[i] for i in k], [o[i] for i in k], e[k], STATUSES, cuts)
        np.testing.assert_allclose(a, b)

    def test_it_agrees_with_the_definition_on_a_random_table(self):
        rng = np.random.default_rng(20261021)
        st = rng.choice(['Out', 'Questionable', 'Probable'], 400)
        o = rng.random(400) < 0.4
        e = rng.integers(0, 1000, 400)
        cuts = np.sort(rng.integers(0, 1000, 25))
        np.testing.assert_allclose(rolling_rates(st, o, e, STATUSES, cuts),
                                   reference(st, o, e, STATUSES, cuts))

    def test_a_status_the_table_was_not_asked_for_does_not_leak_into_another(self):
        # 'Probable' rows must not raise the Out rate, which is what a fillna or a default would do
        st = ['Out', 'Probable', 'Probable']
        r = rolling_rates(st, [False, True, True], np.array([1, 2, 3]), STATUSES, np.array([10]))
        self.assertAlmostEqual(r[0, 0], 1.0 / 3.0)

    def test_an_end_time_the_build_does_not_know_raises(self):
        with self.assertRaises(ValueError):
            rolling_rates(['Out'], [True], np.array([np.nan]), STATUSES, np.array([10.0]))

    def test_it_works_on_the_datetimes_the_build_actually_holds(self):
        e = np.array(['2024-01-01T02:30', '2024-01-05T02:30'], dtype='datetime64[ns]')
        cuts = np.array(['2024-01-03T00:00', '2024-01-09T00:00'], dtype='datetime64[ns]')
        r = rolling_rates(['Out', 'Out'], [True, True], e, STATUSES, cuts)
        self.assertAlmostEqual(r[0, 0], 2.0 / 3.0)
        self.assertAlmostEqual(r[1, 0], 3.0 / 4.0)

    def test_the_prior_is_the_one_the_main_build_used(self):
        # Beta(1,1): one sit-out and one appearance added, so an unseen status is a half and not a zero
        r = rolling_rates([], [], np.array([], dtype=float), STATUSES, np.array([0.0]))
        np.testing.assert_allclose(r, np.full((1, 2), 0.5))


if __name__ == '__main__':
    unittest.main()
