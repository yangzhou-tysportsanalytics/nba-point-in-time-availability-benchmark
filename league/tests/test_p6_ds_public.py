"""Paper 2 stage E: the public path's vendored rules must match the project code, and the source list must be sound."""
import csv, inspect, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'league/public'))
import pit_rules

VENDORED = [('name_key', 'league/src/build.py', 'SUFFIX ='), ('name_keys', 'league/src/build.py', 'def team(code):'),
            ('reason_class', 'league/src/build.py', 'def load_box(out):'),
            ('reference_minutes', 'src/measurement_rules.py', 'def max_lp(A,b,c):')]


class PublicPath(unittest.TestCase):
    def test_vendored_rules_match_project(self):
        for fn, path, end in VENDORED:
            p = ROOT / path
            if not p.exists(): self.skipTest('project code not present (public repo)')
            s = p.read_text(encoding='utf-8'); i = s.index(f'def {fn}(')
            with self.subTest(fn=fn):
                self.assertEqual(inspect.getsource(getattr(pit_rules, fn)).strip(), s[i:s.index(end, i)].strip())

    def test_sources_list(self):
        rows = list(csv.DictReader(open(ROOT / 'league/public/sources.csv')))
        self.assertEqual(len({r['file'] for r in rows}), len(rows))
        self.assertTrue(all(len(r['sha256']) == 64 for r in rows))
        self.assertEqual({r['url'].split('/')[2] for r in rows}, {'ak-static.cms.nba.com', 'www.kaggle.com'})  # no ESPN, no stats.nba.com


if __name__ == '__main__':
    unittest.main()
