"""The cutoff curve of Section 4.4, on this repository's own rebuild.

What the compliant feature is worth, and what the premium is, when the report is read at other cutoffs than
six hours. The curve is the paper's evidence that the premium is a timing quantity rather than a fixed
property of the sport, so it is the one a reader is most likely to want to redraw.

Nothing here reimplements the build. The report rows, the name matching and the rule for which report is
current at a cutoff all come from rebuild.py, called with a different cutoff each time; only the loop and the
bootstrap live here. That matters because the eligibility rule is part of the result: an external review of
the originating project found that restricting only the holdout left teams that had not filed, and therefore
imputed health, inside the fit, so the rule is applied to the training set too.

    python league/scripts/rebuild.py build     # not required, but it is what produces the tables below
    python league/scripts/cutoff_curve.py      # -> results/paper2_cutoff_curve_public.json

The twelve- and nine-hour points are computed on the games where both teams had actually filed that early,
which is a smaller sample; computing them on every game imputes health for a team that has not filed and is
the mistake this benchmark exists to avoid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'league/scripts'))
import public_inputs as pi                                                   # noqa: E402
import rebuild as rb                                                         # noqa: E402

OUT = ROOT / 'results/paper2_cutoff_curve_public.json'
CUTS_H = [12, 9, 6, 4, 3, 2, 1.5, 1, 0.5]
SEASONS = (2023, 2024, 2025, 2026)
BASE = ['strength_diff', 'rest_diff']
SEED, REPS = 20260921, 2000


def main() -> int:
    g, box, _ = rb.load_games_box()
    rows, blocks, _ = rb.load_reports(g)
    R = pi.ref_players()
    R['val'] = R.base_minutes * R.eff_prior
    G = pi.games()

    def feature_at(hours):
        """The compliant feature read `hours` before tip-off, and which team-games had filed by then."""
        best, st = rb.status_at(g, rows, blocks, box, pd.Timedelta(hours=hours))
        out = st[(st.status == 'Out') & st.pid.notna()][['game_id', 'team', 'pid']].drop_duplicates()
        o = set(zip(out.game_id.astype(str), out.team, out.pid.astype(str)))
        flag = np.array([(gm, t, p) in o for gm, t, p in
                         zip(R.game_id.astype(str), R.team, R.espn_player_id.astype(str))])
        v = pd.Series(np.where(flag, R.val, 0.0)).groupby([R.game_id, R.team]).sum()
        # `best` is the latest listed block per team-game, as a frame, so a team-game is covered at this
        # cutoff exactly when it appears in it
        have = set(zip(best.game_id.astype(str), best.team))
        return v, have

    cols = {}
    filed = {}
    for h in CUTS_H:
        v, have = feature_at(h)
        name = f'F1c_{h}'
        cols[name] = [v.get((gm, hm), 0.0) - v.get((gm, aw), 0.0)
                      for gm, hm, aw in zip(G.game_id, G.home, G.away)]
        filed[h] = np.array([((gm, hm) in have) and ((gm, aw) in have)
                             for gm, hm, aw in zip(G.game_id, G.home, G.away)])
        print(f'{h:>5} h: feature built, both teams filed in {filed[h].sum():,} games', flush=True)
    for k, v in cols.items():
        G[k] = v

    fixed = filed[6] & ~G.any_block_missing.values
    H = G[fixed].reset_index(drop=True)
    print(f'\nstrict six-hour sample: {len(H):,} games')

    def evaluate(col, mask=None, reps=REPS):
        """Rolling-origin fit with the eligibility rule applied to training as well as the holdout."""
        ys, P, rowsout = [], {k: [] for k in ('F0', 'F1', 'F2')}, []
        for s in SEASONS:
            tr = H[H.season_year < s] if mask is None else H[(H.season_year < s) & mask]
            sel = (H.season_year == s) if mask is None else ((H.season_year == s) & mask)
            ho = H[sel]
            if not len(tr) or not len(ho):
                continue
            X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])
            for k, c in (('F0', []), ('F1', [col]), ('F2', ['F2_diff'])):
                P[k].append(X(ho, c) @ np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0])
            ys.append(ho.margin.values); rowsout.append(ho)
        y = np.concatenate(ys); Rr = pd.concat(rowsout)
        sq = {k: (y - np.concatenate(v)) ** 2 for k, v in P.items()}
        code, keys = pd.factorize(Rr.home.astype(str) + '_' + Rr.season_year.astype(str))
        rng = np.random.default_rng(SEED)
        W = [np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)).astype(float)[code]
             for _ in range(reps)]

        def st(w, a, b):
            t = w.sum()
            return float(np.sqrt((w * sq[a]).sum() / t) - np.sqrt((w * sq[b]).sum() / t))

        out = {'n': int(len(y))}
        for key, a, b in (('gain_F1', 'F0', 'F1'), ('inflation', 'F1', 'F2')):
            out[key] = round(st(np.ones(len(y)), a, b), 4)
            draws = [st(w, a, b) for w in W]
            out[key + '_ci'] = [round(float(np.percentile(draws, q)), 4) for q in (2.5, 97.5)]
        return out

    res = {'build': 'public rebuild', 'strict_six_hour_games': int(len(H)), 'reps': REPS, 'curve': {},
           'curve_both_filed': {}}
    print(f'\n{"cutoff":>8}{"n":>8}{"gain":>9}{"interval":>22}{"infl":>9}{"interval":>22}')
    for h in CUTS_H:
        e = evaluate(f'F1c_{h}')
        res['curve'][str(h)] = e
        print(f'{h:>8}{e["n"]:>8,}{e["gain_F1"]:>9.4f}{str(e["gain_F1_ci"]):>22}'
              f'{e["inflation"]:>9.4f}{str(e["inflation_ci"]):>22}')
    for h in (12, 9):
        both = filed[h][fixed]
        e = evaluate(f'F1c_{h}', mask=both)
        res['curve_both_filed'][str(h)] = e
        print(f'{h:>6} h*{e["n"]:>8,}{e["gain_F1"]:>9.4f}{str(e["gain_F1_ci"]):>22}'
              f'{e["inflation"]:>9.4f}{str(e["inflation_ci"]):>22}   (both teams filed that early)')

    OUT.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(f'\n-> {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
