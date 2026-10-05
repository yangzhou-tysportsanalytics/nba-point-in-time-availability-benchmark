"""Round 15, item 1: is the two-way team bootstrap calibrated, and if not, by how much and for which quantity?

The review's argument is that the two-way design weights a game by the product of its two teams' draw counts, that
such a product has variance about 3 where an ordinary bootstrap weight has 1, and that the paper's own width
ratios (1.6-1.7 against home clustering) are what you see when there is no team-level dependence to capture. If
that is right, the interval the paper reports is too wide by a factor it did not intend, and "0.059, not separable
from zero" is a statement about the design rather than about the data.

This does not take that on faith and does not assume it is wrong. Four questions:

  1. How much team-level dependence is there? Variance components of the per-game loss difference, by moments on
     pairs of games sharing a team, within a season and across seasons.
  2. Under a known truth on the real schedule, what coverage does each design have, and how wide is its interval
     against the true sampling SD? Placebo differences are generated from the fitted components and again with
     the team structure removed, and the statistic is the RMSE difference the paper reports, not a mean.
  3. What does an analytic dyadic cluster-robust standard error say beside the bootstrap?
  4. For each quantity separately: how much of the two-way width is the product weight, and how much is
     dependence the design is right to carry? The paper's third design, season-then-cluster, has no product
     weights, so the gap between the two separates them.

  python league/scripts/boot_calibration.py   -> results/paper2_boot_calibration.json
  P2_CAL_SIMS, P2_CAL_REPS, P2_CAL_TRUTH override the counts.

Read-only with respect to the manuscript: it writes one result file and changes nothing else.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
OUT = ROOT / f'results/paper2_boot_calibration{"" if BUILD == "main" else "_" + BUILD}.json'
SEED = 20261005
SIMS = int(os.environ.get('P2_CAL_SIMS', 300))      # placebo datasets scored for coverage, per scenario
REPS = int(os.environ.get('P2_CAL_REPS', 600))      # bootstrap replicates inside each placebo
TRUTH = int(os.environ.get('P2_CAL_TRUTH', 4000))   # datasets used to pin the true sampling distribution
SEASONS = (2023, 2024, 2025, 2026)
DESIGNS = ('home_season_cluster', 'two_way_team', 'season_then_cluster', 'iid_game')

# ---------------------------------------------------------------- the holdout, exactly as the paper builds it
G = pd.read_csv(BENCH, dtype={'game_id': str})
G = G[~G.any_block_missing].reset_index(drop=True)
BASE = ['strength_diff', 'rest_diff']
X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])

parts = []
for s in SEASONS:
    tr, ho = G[G.season_year < s], G[G.season_year == s].copy()
    for k, c in (('F0', []), ('F1', ['F1_diff']), ('F2', ['F2_diff'])):
        beta = np.linalg.lstsq(X(tr, c), tr.margin.values, rcond=None)[0]
        ho['p_' + k] = X(ho, c) @ beta
    parts.append(ho)
H = pd.concat(parts, ignore_index=True)
y = H.margin.values
SQ = {k: (y - H['p_' + k].values) ** 2 for k in ('F0', 'F1', 'F2')}
season = H.season_year.values
teams = np.array(sorted(set(H.home) | set(H.away)))
hi_, ai_ = np.searchsorted(teams, H.home.values), np.searchsorted(teams, H.away.values)
hs_code, hs_keys = pd.factorize(pd.Series(H.home + '_' + H.season_year.astype(str)))
n, n_hs, n_teams = len(y), len(hs_keys), len(teams)
season_cluster_ids = {s: np.flatnonzero(pd.Series(hs_keys).str.endswith('_' + str(s)).values) for s in SEASONS}
season_idx = {s: np.flatnonzero(season == s) for s in SEASONS}
# team-season index per game side, for generating placebo data with the right structure
ts_home = hi_ * 10 + np.searchsorted(SEASONS, season)
ts_away = ai_ * 10 + np.searchsorted(SEASONS, season)
ts_codes, _ = pd.factorize(pd.Series(np.concatenate([ts_home, ts_away])))
ts_h, ts_a = ts_codes[:n], ts_codes[n:]
n_ts = ts_codes.max() + 1
# unordered matchup id, for the dyadic estimator's "shares both teams" term
pair_code, _ = pd.factorize(pd.Series(np.minimum(hi_, ai_) * 1000 + np.maximum(hi_, ai_)))
n_pair = pair_code.max() + 1

PAIRS = {'gain_F1': ('F0', 'F1'), 'infl': ('F1', 'F2')}


def rmse_diff(a, b, w=None):
    if w is None:
        return float(np.sqrt(a.mean()) - np.sqrt(b.mean()))
    t = w.sum()
    return float(np.sqrt((w * a).sum() / t) - np.sqrt((w * b).sum() / t))


def rmse_diff_batch(a, b, W, Wsum):
    return np.sqrt(W @ a / Wsum) - np.sqrt(W @ b / Wsum)


# ---------------------------------------------------------------- 1. variance components
def variance_components(d: np.ndarray) -> dict:
    """Moments on pairs of games sharing a team.

    Two games sharing a team and a season have covariance var_team + var_team_season; sharing a team across
    seasons, var_team alone. What is left in a single game is var_game. Sums over a team's games give both
    without enumerating pairs.
    """
    dc = d - d.mean()
    tot = float(dc.var(ddof=1))
    num_s = den_s = num_c = den_c = 0.0
    for t in range(n_teams):
        idx = np.flatnonzero((hi_ == t) | (ai_ == t))
        if len(idx) < 2:
            continue
        v, s = dc[idx], season[idx]
        tot_sum, tot_sq = v.sum(), (v ** 2).sum()
        all_pairs = tot_sum ** 2 - tot_sq                       # ordered pairs g != h
        same = 0.0
        for ss in SEASONS:
            m = v[s == ss]
            same += m.sum() ** 2 - (m ** 2).sum()
        num_s += same
        den_s += sum(max((s == ss).sum() ** 2 - (s == ss).sum(), 0) for ss in SEASONS)
        num_c += all_pairs - same
        den_c += (len(idx) ** 2 - len(idx)) - sum(max((s == ss).sum() ** 2 - (s == ss).sum(), 0) for ss in SEASONS)
    same_cov = num_s / den_s if den_s else 0.0
    cross_cov = num_c / den_c if den_c else 0.0
    var_team = max(cross_cov, 0.0)
    var_team_season = max(same_cov - cross_cov, 0.0)
    var_game = max(tot - var_team - var_team_season, 0.0)
    return dict(total=round(tot, 6), team=round(var_team, 6), team_season=round(var_team_season, 6),
                game=round(var_game, 6), cov_same_season=round(same_cov, 6), cov_across_seasons=round(cross_cov, 6),
                share_team=round(var_team / tot, 4), share_team_season=round(var_team_season / tot, 4),
                share_game=round(var_game / tot, 4))


# ---------------------------------------------------------------- the designs
def weight_bank(rng, design, reps):
    W = np.empty((reps, n))
    for r in range(reps):
        if design == 'home_season_cluster':
            c = np.bincount(rng.integers(0, n_hs, n_hs), minlength=n_hs).astype(float)
            W[r] = c[hs_code]
        elif design == 'two_way_team':
            c = np.bincount(rng.integers(0, n_teams, n_teams), minlength=n_teams).astype(float)
            W[r] = c[hi_] * c[ai_]
        elif design == 'season_then_cluster':
            c = np.zeros(n_hs)
            for s in rng.choice(SEASONS, len(SEASONS)):
                ids = season_cluster_ids[s]
                c += np.bincount(rng.choice(ids, len(ids)), minlength=n_hs).astype(float)
            W[r] = c[hs_code]
        elif design == 'iid_game':
            W[r] = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        else:
            raise ValueError(design)
    return W


# ---------------------------------------------------------------- 3. analytic dyadic cluster-robust SE
def dyadic_se(a, b):
    """Aronow, Samii & Assenova (2015) on the delta-method influence function of sqrt(ma) - sqrt(mb).

    V = (1/n^2) * sum over ordered game pairs sharing at least one team. Summing psi over each team's games and
    squaring counts a pair once per shared team, so pairs sharing both teams (including a game with itself) are
    counted twice and are subtracted once, grouped by unordered matchup.
    """
    ma, mb = a.mean(), b.mean()
    psi = (a - ma) / (2 * np.sqrt(ma)) - (b - mb) / (2 * np.sqrt(mb))
    S = np.zeros(n_teams)
    np.add.at(S, hi_, psi)
    np.add.at(S, ai_, psi)
    T = np.zeros(n_pair)
    np.add.at(T, pair_code, psi)
    v = (float((S ** 2).sum()) - float((T ** 2).sum())) / n ** 2
    return float(np.sqrt(max(v, 0.0)))


# ---------------------------------------------------------------- 2. placebo calibration
def calibrate(b: np.ndarray, comps: dict, scenario: str, rng) -> dict:
    """Generate per-game differences with a known truth on the real schedule and score every design.

    d_g = mu + u_home + u_away + v_(home,season) + v_(away,season) + e_g, with the fitted variances, or with the
    team and team-season variances set to zero. a = b + d, and the statistic is sqrt(mean a) - sqrt(mean b), the
    paper's. The weights do not depend on the data, so one bank of draws per design serves every placebo.
    """
    mu = comps['_mu']
    sd_team = np.sqrt(comps['team']) if scenario == 'fitted' else 0.0
    sd_ts = np.sqrt(comps['team_season']) if scenario == 'fitted' else 0.0
    sd_game = np.sqrt(comps['total'] - (comps['team'] + comps['team_season'] if scenario == 'fitted' else 0.0))

    def make(r):
        u = r.normal(0, sd_team, n_teams) if sd_team else np.zeros(n_teams)
        v = r.normal(0, sd_ts, n_ts) if sd_ts else np.zeros(n_ts)
        return mu + u[hi_] + u[ai_] + v[ts_h] + v[ts_a] + r.normal(0, sd_game, n)

    # true sampling distribution of the statistic under this DGP
    tr = np.random.default_rng(SEED + 101)
    truth = np.array([rmse_diff(b + make(tr), b) for _ in range(TRUTH)])
    true_mean, true_sd = float(truth.mean()), float(truth.std(ddof=1))

    banks = {d: [weight_bank(rng, d, REPS) for _ in range(3)] for d in DESIGNS}
    sums = {d: [W.sum(1) for W in banks[d]] for d in DESIGNS}

    res = {}
    for design in DESIGNS:
        cover, widths = 0, []
        sr = np.random.default_rng(SEED + 202)
        for i in range(SIMS):
            a = b + make(sr)
            k = i % 3
            vals = rmse_diff_batch(a, b, banks[design][k], sums[design][k])
            lo, hi = np.percentile(vals, [2.5, 97.5])
            widths.append(hi - lo)
            cover += int(lo <= true_mean <= hi)
        mw = float(np.mean(widths))
        res[design] = dict(coverage=round(cover / SIMS, 3), mean_width=round(mw, 4),
                           width_over_true=round(mw / (2 * 1.959964 * true_sd), 3))
    return dict(scenario=scenario, true_value=round(true_mean, 4), true_sd=round(true_sd, 4),
                nominal_width=round(2 * 1.959964 * true_sd, 4), designs=res)


def season_level(a, b) -> dict:
    """What the four seasons say when a season, not a game, is the unit.

    The per-game components below come out almost entirely game-level, yet every design that resamples a higher
    level stays wide. That is because the seasons differ in the statistic itself -- each has its own refit -- and
    a season-level shift barely moves the variance of a single game while dominating the variance of an average
    over a season. So the width question is really a question about the target: the value in these four seasons,
    or the value in a season drawn from the same process.
    """
    per, se2 = [], []
    rng = np.random.default_rng(SEED + 7)
    for s in SEASONS:
        m = season_idx[s]
        per.append(rmse_diff(a[m], b[m]))
        # within-season sampling variance, from an iid game bootstrap inside that season
        W = np.bincount(rng.integers(0, len(m), (800, len(m))).ravel() + np.repeat(np.arange(800), len(m)) * len(m),
                        minlength=800 * len(m)).reshape(800, len(m)).astype(float)
        v = rmse_diff_batch(a[m], b[m], W, W.sum(1))
        se2.append(float(v.var(ddof=1)))
    per = np.array(per)
    between = float(per.var(ddof=1))
    within = float(np.mean(se2))
    tau2 = max(between - within, 0.0)                       # season-to-season variance of the true value
    se_mean = np.sqrt((tau2 + within) / len(SEASONS))
    from scipy import stats
    t3 = float(stats.t.ppf(0.975, len(SEASONS) - 1))
    return dict(per_season={str(s): round(float(p), 4) for s, p in zip(SEASONS, per)},
                between_season_var=round(between, 6), mean_within_season_var=round(within, 6),
                season_variance_tau2=round(tau2, 6), season_sd=round(float(np.sqrt(tau2)), 4),
                se_of_four_season_mean=round(float(se_mean), 4),
                ci95_normal=[round(float(per.mean() - 1.959964 * se_mean), 4),
                             round(float(per.mean() + 1.959964 * se_mean), 4)],
                ci95_t3=[round(float(per.mean() - t3 * se_mean), 4),
                         round(float(per.mean() + t3 * se_mean), 4)],
                width_t3=round(float(2 * t3 * se_mean), 4),
                note='target is a season drawn from the same process, not these four seasons')


def main() -> int:
    rng = np.random.default_rng(SEED)
    out = dict(n_games=int(n), n_teams=int(n_teams), sims=SIMS, reps=REPS, truth_draws=TRUTH, seed=SEED,
               quantities={})

    for q, (ka, kb) in PAIRS.items():
        a, b = SQ[ka], SQ[kb]
        point = rmse_diff(a, b)
        comps = variance_components(a - b)
        comps['_mu'] = float((a - b).mean())
        se_dy = dyadic_se(a, b)

        observed = {}
        for design in DESIGNS:
            W = weight_bank(rng, design, 3000)
            vals = rmse_diff_batch(a, b, W, W.sum(1))
            lo, hi = np.percentile(vals, [2.5, 97.5])
            observed[design] = dict(ci95=[round(float(lo), 4), round(float(hi), 4)],
                                    width=round(float(hi - lo), 4), boot_sd=round(float(vals.std(ddof=1)), 4))

        print(f'\n=== {q}: RMSE({ka}) - RMSE({kb}) = {point:.4f} ===', flush=True)
        print(f'  variance of the per-game difference: team {comps["share_team"]:.1%}, '
              f'team-season {comps["share_team_season"]:.1%}, game {comps["share_game"]:.1%}', flush=True)
        print(f'  dyadic cluster-robust SE {se_dy:.4f}  ->  95% CI '
              f'[{point - 1.96 * se_dy:.4f}, {point + 1.96 * se_dy:.4f}]  width {2 * 1.96 * se_dy:.4f}', flush=True)
        seas = season_level(a, b)
        print(f'  season as the unit: per-season {list(seas["per_season"].values())}, '
              f'season SD {seas["season_sd"]:.4f}, SE of the mean {seas["se_of_four_season_mean"]:.4f}', flush=True)
        print(f'     normal CI {seas["ci95_normal"]} ; t(3) CI {seas["ci95_t3"]} width {seas["width_t3"]:.4f}',
              flush=True)
        for d, o in observed.items():
            print(f'  {d:22} {o["ci95"]}  width {o["width"]:.4f}', flush=True)

        cal = {}
        for scenario in ('fitted', 'no_team'):
            cal[scenario] = calibrate(b, comps, scenario, rng)
            c = cal[scenario]
            print(f'  -- placebo, {scenario}: true SD {c["true_sd"]:.4f}', flush=True)
            for d, r in c['designs'].items():
                print(f'     {d:22} coverage {r["coverage"]:.3f}  width/nominal {r["width_over_true"]:.2f}',
                      flush=True)

        comps.pop('_mu')
        out['quantities'][q] = dict(contrast=f'RMSE({ka}) - RMSE({kb})', point=round(point, 4),
                                    variance_components=comps, dyadic_se=round(se_dy, 4),
                                    dyadic_ci95=[round(point - 1.959964 * se_dy, 4),
                                                 round(point + 1.959964 * se_dy, 4)],
                                    observed_bootstrap=observed, season_level=seas, calibration=cal)

    OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f'\n-> {OUT.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
