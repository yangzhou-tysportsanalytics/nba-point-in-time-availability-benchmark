"""The encoding plan Y reads the report with, and the cases it has to get right.

Vendored from the originating project as `league/public/cutoff_encoding.py`. The analysis that uses it needs
the player-level table the rebuild writes on the reader's machine, so it cannot be rerun here; this module
is the part of it that goes wrong quietly, and these are the cases it was built against. That it has not drifted from the project's copy is checked in the project, which is the only
place both repositories exist.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cutoff_encoding import selected_time, weight, team_value  # noqa: E402


class SelectedReport(unittest.TestCase):
    TIMES = (100, 200, 300, 400)

    def test_the_selected_report_is_the_last_one_filed_by_the_cutoff(self):
        self.assertEqual(selected_time(self.TIMES, 250), 200)

    def test_a_report_printed_exactly_on_the_cutoff_counts(self):
        # 52.8% of selected team-games are printed exactly on the six-hour boundary, so an exclusive
        # comparison here would drop most of the sample
        self.assertEqual(selected_time(self.TIMES, 300), 300)

    def test_a_team_that_had_not_filed_is_unknown_rather_than_healthy(self):
        self.assertIsNone(selected_time(self.TIMES, 50))

    def test_a_later_cutoff_never_selects_an_earlier_report(self):
        for early, late in ((150, 250), (250, 350), (300, 400)):
            self.assertGreaterEqual(selected_time(self.TIMES, late), selected_time(self.TIMES, early))


class Weight(unittest.TestCase):
    RATES = {'Out': 0.99, 'Doubtful': 0.97, 'Questionable': 0.49, 'Probable': 0.09, 'Available': 0.18,
             'not_listed': 0.5}

    def test_out_only_charges_the_whole_player_for_out_and_nothing_else(self):
        self.assertEqual(weight('Out', self.RATES, 'out_only'), 1.0)
        for s in ('Doubtful', 'Questionable', 'Probable', 'Available', 'not_listed'):
            self.assertEqual(weight(s, self.RATES, 'out_only'), 0.0)

    def test_all_statuses_charges_the_sit_out_rate_of_the_status(self):
        self.assertAlmostEqual(weight('Questionable', self.RATES, 'all_statuses'), 0.49)
        self.assertAlmostEqual(weight('Available', self.RATES, 'all_statuses'), 0.18)

    def test_a_player_the_report_does_not_list_is_charged_nothing_even_with_a_rate_for_it(self):
        # F7 is built as (status != 'not_listed') * q * w; a rate map carrying a not_listed entry must not
        # turn every unlisted rotation player into a partial absence
        self.assertEqual(weight('not_listed', self.RATES, 'all_statuses'), 0.0)

    def test_the_rates_come_from_the_map_and_are_not_fixed_in_the_code(self):
        other = dict(self.RATES, Questionable=0.2)
        self.assertNotAlmostEqual(weight('Questionable', self.RATES, 'all_statuses'),
                                  weight('Questionable', other, 'all_statuses'))

    def test_a_status_the_league_has_not_used_before_is_an_error_rather_than_a_zero(self):
        with self.assertRaises(KeyError):
            weight('Game Time Decision', self.RATES, 'all_statuses')


class TeamValue(unittest.TestCase):
    RATES = {'Out': 1.0, 'Doubtful': 0.9, 'Questionable': 0.5, 'Probable': 0.1, 'Available': 0.2}
    PLAYERS = (('a', 10.0), ('b', 20.0), ('c', 5.0))

    def test_a_player_not_listed_early_and_out_late_is_added_at_his_whole_value(self):
        early = {'b': 'Out'}
        late = {'b': 'Out', 'a': 'Out'}
        for enc in ('out_only', 'all_statuses'):
            self.assertAlmostEqual(team_value(self.PLAYERS, late, self.RATES, enc)
                                   - team_value(self.PLAYERS, early, self.RATES, enc), 10.0)

    def test_a_status_that_softens_lowers_the_value_instead_of_keeping_it(self):
        early = {'a': 'Out'}
        late = {'a': 'Questionable'}
        self.assertAlmostEqual(team_value(self.PLAYERS, late, self.RATES, 'all_statuses'), 5.0)
        self.assertAlmostEqual(team_value(self.PLAYERS, late, self.RATES, 'out_only'), 0.0)
        self.assertLess(team_value(self.PLAYERS, late, self.RATES, 'all_statuses'),
                        team_value(self.PLAYERS, early, self.RATES, 'all_statuses'))

    def test_statuses_short_of_out_are_what_separates_the_two_encodings(self):
        s = {'a': 'Questionable', 'b': 'Probable'}
        self.assertEqual(team_value(self.PLAYERS, s, self.RATES, 'out_only'), 0.0)
        self.assertAlmostEqual(team_value(self.PLAYERS, s, self.RATES, 'all_statuses'),
                               10.0 * 0.5 + 20.0 * 0.1)

    def test_a_player_outside_the_reference_rotation_is_not_counted(self):
        # the report lists players the rotation does not contain; the feature is a sum over the rotation
        s = {'a': 'Out', 'z': 'Out'}
        self.assertAlmostEqual(team_value(self.PLAYERS, s, self.RATES, 'out_only'), 10.0)


if __name__ == '__main__':
    unittest.main()
