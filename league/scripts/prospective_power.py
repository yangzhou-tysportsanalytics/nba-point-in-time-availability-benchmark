"""What the choice of bootstrap design costs the 2026-27 prospective test in power.

The registration decides every hypothesis on the lower bound of a 90% two-sided (= 95% one-sided) interval from
the two-way team bootstrap. The calibration in boot_calibration.py found that design covering at 1.000 with
an interval 1.5-1.7 times the nominal width on the pooled sample. The prospective test is a single season, so the
season-to-season variance that makes a wide interval defensible in the paper does not arise there: one season is
the whole sample, and an interval wider than the sampling distribution only costs detections.

This quantifies the cost. For one season on the real schedule it measures, per design, the bootstrap SD the
design reports and the true sampling SD of the same statistic, then reads power off

    power(theta) = Phi( theta / sigma_true  -  z * sigma_design / sigma_true )

with z = 1.645 for the registered one-sided rule. A design that reports the truth has sigma_design/sigma_true = 1
and the usual power curve; one that reports 1.5 times the truth pays 1.5 z before it starts.

  python league/scripts/prospective_power.py -> results/paper2_prospective_power.json

Covers the two contrasts the released features support exactly: H1 (F0 against F1) and H3 (F1 against F2). H2,
H3b, H4, H5 and H7 need cutoff-specific or all-status features that are not in the released build; for those the
design ratio still applies and the script prints the power curve against effect size so they can be read off once
their standard errors are known.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[2]

def resolve_bench() -> tuple[Path, str]:
    """The same script serves the main build and the released one; the file present decides which.

    P2_BENCH overrides. The released build has fewer games and excludes a few differently, so the two runs are
    reported side by side rather than one standing in for the other.
    """
    env = os.environ.get('P2_BENCH')
    if env:
        return Path(env), 'custom'
    for rel, label in ((Path('data/league/paper2_ext/bench_games.csv'), 'main'),
                       (Path('data/bench_games_public.csv'), 'public')):
        if (ROOT / rel).exists():
            return ROOT / rel, label
    raise SystemExit('no benchmark file found: set P2_BENCH')


BENCH, BUILD = resolve_bench()
OUT = ROOT / f'results/paper2_prospective_power{"" if BUILD == "main" else "_" + BUILD}.json'
SEED = 20261005
TRUTH = int(os.environ.get('P2_PW_TRUTH', 4000))
REPS = int(os.environ.get('P2_PW_REPS', 4000))
Z = 1.6448536269514722                      # the registered rule: lower bound of the 95% one-sided interval
SEASONS = (2023, 2024, 2025, 2026)
TARGET = 2026                               # the most recent full holdout, as the stand-in for 2026-27
DESIGNS = ('home_season_cluster', 'two_way_team', 'iid_game')

G = pd.read_csv(BENCH, dtype={'game_id': str})
G = G[~G.any_block_missing].reset_index(drop=True)
BASE = ['strength_diff', 'rest_diff']
X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])

tr, ho = G[G.season_year < TARGET], G[G.season_year == TARGET].copy()
for k, c in (('F0', []), ('F1', ['F1_diff']), ('F2', ['F2_diff'])):
    beta = np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0]
    ho['p_' + k] = X(ho, c) @ beta
y = ho.margin.values
SQ = {k: (y - ho['p_' + k].values) ** 2 for k in ('F0', 'F1', 'F2')}
teams = np.array(sorted(set(ho.home) | set(ho.away)))
hi_, ai_ = np.searchsorted(teams, ho.home.values), np.searchsorted(teams, ho.away.values)
home_code, home_keys = pd.factorize(ho.home)
n, n_teams, n_home = len(y), len(teams), len(home_keys)
PAIRS = {'H1_gain_F1': ('F0', 'F1'), 'H3_inflation': ('F1', 'F2')}
EXPECTED = {'H1_gain_F1': 0.20, 'H3_inflation': 0.06}


def rmse_diff(a, b):
    return float(np.sqrt(a.mean()) - np.sqrt(b.mean()))


def rmse_diff_batch(a, b, W, Wsum):
    return np.sqrt(W @ a / Wsum) - np.sqrt(W @ b / Wsum)


def weight_bank(rng, design, reps):
    W = np.empty((reps, n))
    for r in range(reps):
        if design == 'home_season_cluster':      # within one season this is clustering on the home team
            c = np.bincount(rng.integers(0, n_home, n_home), minlength=n_home).astype(float)
            W[r] = c[home_code]
        elif design == 'two_way_team':
            c = np.bincount(rng.integers(0, n_teams, n_teams), minlength=n_teams).astype(float)
            W[r] = c[hi_] * c[ai_]
        elif design == 'iid_game':
            W[r] = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        else:
            raise ValueError(design)
    return W


def components(d):
    """Team and game variance of the per-game difference, within this one season."""
    dc = d - d.mean()
    tot = float(dc.var(ddof=1))
    num = den = 0.0
    for t in range(n_teams):
        idx = np.flatnonzero((hi_ == t) | (ai_ == t))
        if len(idx) < 2:
            continue
        v = dc[idx]
        num += v.sum() ** 2 - (v ** 2).sum()
        den += len(idx) ** 2 - len(idx)
    cov = num / den if den else 0.0
    var_team = max(cov, 0.0)
    return dict(total=tot, team=var_team, game=max(tot - var_team, 0.0))


def main() -> int:
    rng = np.random.default_rng(SEED)
    out = dict(season=TARGET, n_games=int(n), n_teams=int(n_teams), z_one_sided=round(Z, 4),
               reps=REPS, truth_draws=TRUTH, quantities={})

    for q, (ka, kb) in PAIRS.items():
        a, b = SQ[ka], SQ[kb]
        point = rmse_diff(a, b)
        comp = components(a - b)
        mu = float((a - b).mean())

        # true sampling SD on this schedule, from the fitted components
        sd_team, sd_game = np.sqrt(comp['team']), np.sqrt(comp['game'])
        tr_rng = np.random.default_rng(SEED + 11)
        truth = np.empty(TRUTH)
        for i in range(TRUTH):
            u = tr_rng.normal(0, sd_team, n_teams) if sd_team else np.zeros(n_teams)
            d = mu + u[hi_] + u[ai_] + tr_rng.normal(0, sd_game, n)
            truth[i] = rmse_diff(b + d, b)
        sigma_true = float(truth.std(ddof=1))

        # coverage on this one season, the setting the prospective test actually runs in
        true_mean = float(truth.mean())
        SIMS = 300
        banks = {d: [weight_bank(rng, d, 600) for _ in range(3)] for d in DESIGNS}
        sums = {d: [W.sum(1) for W in banks[d]] for d in DESIGNS}
        cover = {d: 0 for d in DESIGNS}
        cr = np.random.default_rng(SEED + 23)
        for i in range(SIMS):
            u = cr.normal(0, sd_team, n_teams) if sd_team else np.zeros(n_teams)
            aa = b + mu + u[hi_] + u[ai_] + cr.normal(0, sd_game, n)
            for d in DESIGNS:
                k = i % 3
                vv = rmse_diff_batch(aa, b, banks[d][k], sums[d][k])
                lo, hi = np.percentile(vv, [2.5, 97.5])
                cover[d] += int(lo <= true_mean <= hi)

        rows = {}
        for design in DESIGNS:
            W = weight_bank(rng, design, REPS)
            v = rmse_diff_batch(a, b, W, W.sum(1))
            sd_design = float(v.std(ddof=1))
            ratio = sd_design / sigma_true
            theta = EXPECTED[q]
            power = float(norm.cdf(theta / sigma_true - Z * ratio))
            # the effect this design needs for 80% power
            mde = float((norm.ppf(0.80) + Z * ratio) * sigma_true)
            rows[design] = dict(sd_design=round(sd_design, 4), ratio_to_true=round(ratio, 3),
                                coverage_one_season=round(cover[design] / SIMS, 3),
                                power_at_expected=round(power, 3), mde_80=round(mde, 4),
                                ci95=[round(float(np.percentile(v, 2.5)), 4),
                                      round(float(np.percentile(v, 97.5)), 4)])

        curve = {}
        for design in DESIGNS:
            r = rows[design]['ratio_to_true']
            curve[design] = {f'{t:.2f}': round(float(norm.cdf(t / sigma_true - Z * r)), 3)
                             for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)}

        out['quantities'][q] = dict(contrast=f'RMSE({ka}) - RMSE({kb})', point_2025_26=round(point, 4),
                                    expected_size_registered=EXPECTED[q], sigma_true=round(sigma_true, 4),
                                    team_share_of_per_game_variance=round(comp['team'] / comp['total'], 4),
                                    designs=rows, power_curve_by_effect=curve)

        print(f'\n=== {q}  ({TARGET} holdout, {n} games) ===', flush=True)
        print(f'  true sampling SD {sigma_true:.4f}; registered expected size {EXPECTED[q]}', flush=True)
        for design, r in rows.items():
            print(f'  {design:22} SD {r["sd_design"]:.4f}  ratio {r["ratio_to_true"]:.2f}  '
                  f'coverage {r["coverage_one_season"]:.3f}  power {r["power_at_expected"]:.3f}  '
                  f'80%-MDE {r["mde_80"]:.4f}', flush=True)
        print('  power by effect size:', flush=True)
        for design in DESIGNS:
            print(f'    {design:22} ' + '  '.join(f'{k}:{v:.2f}' for k, v in curve[design].items()), flush=True)

    OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'\n-> {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
