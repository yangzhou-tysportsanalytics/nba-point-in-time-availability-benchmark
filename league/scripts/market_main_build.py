"""The three market comparisons of the paper, against the opening and the closing price.

Published unchanged below its original header. It reads the originating project's paths, so
it is not rerun here; its output is `results/paper2_market2.json`, which carries RMSE differences and intervals and no player rows.

The original header follows.

Paper 2, plan O items O6 and O7.

O6: the market-as-a-feature test (M2) replicated on the 2022-23 holdout, whose consensus closing prices exist,
    trained on 2021-22, beside the 2025-26 holdout the paper already reports.
O7: the win-rate intervals recomputed three ways -- binomial, an iid game bootstrap and the home-team cluster
    bootstrap the paper used -- so the method can be stated and the width checked.

  python league/scripts/p6_ds_market2.py  -> results/paper2_market2.json
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / 'data/league/paper2_ext'
SEED, REPS = 20260921, 2000
G = pd.read_csv(EXT / 'bench_games.csv', dtype={'game_id': str})
L = pd.read_csv(EXT / 'lines_open.csv', dtype={'game_id': str})[['game_id', 'open_spread_home', 'close_spread_home']]
G = G[~G.any_block_missing].merge(L, on='game_id', how='left')
# plan U item U2: the month each game was played, so the two seasons that have an opening price can be
# compared on a matched calendar window. They cover almost disjoint parts of the season, and the paper's own
# month profile runs from -0.037 in October-December to +0.184 in March-April.
_g = (pd.read_csv(EXT / 'rebuild/data/games.csv', dtype={'game_id': str})
        .drop_duplicates('game_id')[['game_id', 'start_utc']])
G = G.merge(_g, on='game_id', how='left')
G['month'] = pd.to_datetime(G.start_utc, utc=True).dt.month
X = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in c])
ci = lambda v: [round(float(np.percentile(v, q)), 4) for q in (2.5, 97.5)]
res = {}


# The strict benchmark's membership, for the sensitivity that asks whether the table survives the
# paper's own six-hour rule: these comparisons otherwise accept the 90-minute fallback.
STRICT_IDS = set(json.load(open(ROOT / 'results/paper2_strict_sample.json'))['game_ids'])


def m2(train_seasons, holdout_season, price='spread_home', label='', months=None, strict=False):
    _G = G[G.game_id.isin(STRICT_IDS)] if strict else G
    tr = _G[_G.season_year.isin(train_seasons)].dropna(subset=[price, 'F1_diff', 'F2_diff', 'margin']).assign(line=lambda d: -d[price])
    ho = _G[(_G.season_year == holdout_season)]
    if months is not None:
        ho = ho[ho.month.isin(months)]
    ho = ho.dropna(subset=[price, 'F1_diff', 'F2_diff', 'margin']).assign(line=lambda d: -d[price]).reset_index(drop=True)
    if len(tr) < 200 or len(ho) < 200:
        return dict(skipped=True, n_train=len(tr), n_holdout=len(ho))
    y = ho.margin.values
    P = {k: X(ho, ['line'] + c) @ np.linalg.lstsq(X(tr, ['line'] + c), tr.margin.values, rcond=None)[0]
         for k, c in (('only', []), ('F1', ['F1_diff']), ('F2', ['F2_diff']))}
    P['raw'] = ho.line.values
    teams = np.array(sorted(set(ho.home) | set(ho.away)))
    hi_, ai_ = np.searchsorted(teams, ho.home.values), np.searchsorted(teams, ho.away.values)
    hs = pd.Series(ho.home); code, keys = pd.factorize(hs); n_k = len(keys)
    rng = np.random.default_rng(SEED)
    W = [np.bincount(rng.integers(0, n_k, n_k), minlength=n_k).astype(float)[code] for _ in range(REPS)]
    W2 = []
    for _ in range(REPS):
        cnt = np.bincount(rng.integers(0, len(teams), len(teams)), minlength=len(teams)).astype(float)
        W2.append(cnt[hi_] * cnt[ai_])

    def stat(w, a, b):
        tot = w.sum()
        r = {k: np.sqrt(np.sum(w * (y - P[k]) ** 2) / tot) for k in (a, b)}
        return r[a] - r[b]

    out = dict(n_train=int(len(tr)), n_holdout=int(len(ho)), label=label)
    if months is not None:
        out['months'] = sorted(months)
        out['month_counts'] = {str(m): int(c) for m, c in sorted(ho.month.value_counts().items())}
    for lab, a, b in (('report_adds_to_M2', 'only', 'F1'), ('participation_adds_to_M2', 'only', 'F2'),
                      ('inflation_on_top_of_price', 'F1', 'F2'),
                      ('M2_F1_vs_raw_price', 'raw', 'F1'), ('M2_F2_vs_raw_price', 'raw', 'F2'),
                      ('M2_price_only_vs_raw_price', 'raw', 'only')):
        out[lab] = dict(point=round(stat(np.ones(len(y)), a, b), 4),
                        ci95_home=ci([stat(w, a, b) for w in W]), ci95_two_way=ci([stat(w, a, b) for w in W2]))
    return out


res['closing_2026'] = m2((2022, 2023), 2026, 'spread_home', 'holdout 2025-26, trained on 2021-22 and 2022-23')
res['closing_2026_strict'] = m2((2022, 2023), 2026, 'spread_home',
                                'holdout 2025-26, both teams filed at six hours, trained on the same restriction',
                                strict=True)
res['opening_2026'] = m2((2022, 2023), 2026, 'open_spread_home', 'holdout 2025-26, opening price, trained on 2021-23')
res['closing_2023'] = m2((2022,), 2023, 'spread_home', 'holdout 2022-23, trained on 2021-22')
res['opening_2023'] = m2((2022,), 2023, 'open_spread_home', 'holdout 2022-23, opening price, trained on 2021-22')

# ---- plan U item U2: the same opening-price comparison on a matched calendar window.
# 2022-23's archive holds an opening price only for October to January; 2025-26's holds none for October and
# runs to April. The premium on top of a price is the quantity whose month profile the paper reports, so the
# two seasons as they stand are not comparable and the matched window is what a replication would mean.
_OPEN_MONTHS = {}
for _s in sorted(G.season_year.unique()):
    _sub = G[(G.season_year == _s) & G.open_spread_home.notna()]
    if len(_sub):
        _OPEN_MONTHS[int(_s)] = {str(m): int(c) for m, c in sorted(_sub.month.value_counts().items())}
_MATCH = sorted({int(m) for m in _OPEN_MONTHS.get(2023, {})})
_REST = sorted({int(m) for m in _OPEN_MONTHS.get(2026, {})} - set(_MATCH))
res['opening_month_coverage'] = _OPEN_MONTHS
res['opening_2026_matched'] = m2((2022, 2023), 2026, 'open_spread_home',
                                 'holdout 2025-26, opening price, restricted to the months 2022-23 covers',
                                 months=_MATCH)
res['opening_2026_rest'] = m2((2022, 2023), 2026, 'open_spread_home',
                              'holdout 2025-26, opening price, the months 2022-23 does not cover',
                              months=_REST)
_prov = pd.read_csv(EXT / 'lines_open.csv', dtype={'game_id': str})
res['price_provider_by_season'] = {str(int(s)): sorted(set(d.provider.dropna()))
                                   for s, d in _prov.groupby('season_year')}

# ---- O7: the win-rate interval, three ways, on the 2025-26 opening-line bets the paper reports
ho = G[(G.season_year == 2026) & G.open_spread_home.notna() & G.close_spread_home.notna()].reset_index(drop=True)
tr = G[G.season_year.isin((2024, 2025))]
BASE = ['strength_diff', 'rest_diff']
Xb = lambda d, c: np.column_stack([np.ones(len(d))] + [d[x].values for x in BASE + c])
pred = {k: Xb(ho, c) @ np.linalg.lstsq(Xb(tr, c), tr.margin.values, rcond=None)[0]
        for k, c in (('F1', ['F1_diff']), ('F2', ['F2_diff']))}
line = -ho.open_spread_home.values
y = ho.margin.values
teams = np.array(sorted(set(ho.home) | set(ho.away)))
hi_, ai_ = np.searchsorted(teams, ho.home.values), np.searchsorted(teams, ho.away.values)
hs = pd.Series(ho.home); code, keys = pd.factorize(hs); n_k = len(keys)
res['betting'] = {}
for k in ('F1', 'F2'):
    d = pred[k] - line
    bet = (np.abs(d) >= 1) & ((y - line) != 0)
    cover = (np.sign(y - line) == np.sign(d)).astype(float)
    n = int(bet.sum()); p = float(cover[bet].mean())
    se = np.sqrt(p * (1 - p) / n)
    rng = np.random.default_rng(SEED)
    iid, clus, two = [], [], []
    for _ in range(REPS):
        i = rng.integers(0, len(ho), len(ho))
        m = bet[i]
        if m.sum(): iid.append(cover[i][m].mean())
        w = np.bincount(rng.integers(0, n_k, n_k), minlength=n_k).astype(float)[code] * bet
        if w.sum(): clus.append(float(np.sum(w * cover) / w.sum()))
        cnt = np.bincount(rng.integers(0, len(teams), len(teams)), minlength=len(teams)).astype(float)
        w2 = cnt[hi_] * cnt[ai_] * bet
        if w2.sum(): two.append(float(np.sum(w2 * cover) / w2.sum()))
    res['betting'][k] = dict(bets=n, win_rate=round(p, 4),
                             binomial_ci=[round(p - 1.96 * se, 4), round(p + 1.96 * se, 4)],
                             iid_game_bootstrap_ci=ci(iid), home_cluster_ci=ci(clus), two_way_ci=ci(two))

# the closing-line record the paper prints comes from M2 with the two controls (p6_ds_models.py); recompute it
# the same three ways so the method and the width can be stated
hc = G[(G.season_year == 2026)].dropna(subset=['spread_home', 'F1_diff', 'F2_diff', 'margin']).reset_index(drop=True)
trc = G[G.season_year.isin((2022, 2023))].dropna(subset=['spread_home', 'F1_diff', 'F2_diff', 'margin'])
for d in (hc, trc):
    d['line'] = -d.spread_home
yc = hc.margin.values; linec = hc.line.values
predc = {k: np.column_stack([np.ones(len(hc))] + [hc[x].values for x in ['line'] + BASE + c]) @
            np.linalg.lstsq(np.column_stack([np.ones(len(trc))] + [trc[x].values for x in ['line'] + BASE + c]),
                            trc.margin.values, rcond=None)[0]
         for k, c in (('F1', ['F1_diff']), ('F2', ['F2_diff']))}
teams_c = np.array(sorted(set(hc.home) | set(hc.away)))
hic, aic = np.searchsorted(teams_c, hc.home.values), np.searchsorted(teams_c, hc.away.values)
hsc = pd.Series(hc.home); codec, keysc = pd.factorize(hsc); n_kc = len(keysc)
res['betting_closing'] = {}
for k in ('F1', 'F2'):
    d = predc[k] - linec
    bet = (np.abs(d) >= 1) & ((yc - linec) != 0)
    cover = (np.sign(yc - linec) == np.sign(d)).astype(float)
    n = int(bet.sum()); pw = float(cover[bet].mean()); se = np.sqrt(pw * (1 - pw) / n)
    rng = np.random.default_rng(SEED); iid, clus, two = [], [], []
    for _ in range(REPS):
        i = rng.integers(0, len(hc), len(hc)); m = bet[i]
        if m.sum(): iid.append(cover[i][m].mean())
        w = np.bincount(rng.integers(0, n_kc, n_kc), minlength=n_kc).astype(float)[codec] * bet
        if w.sum(): clus.append(float(np.sum(w * cover) / w.sum()))
        cnt = np.bincount(rng.integers(0, len(teams_c), len(teams_c)), minlength=len(teams_c)).astype(float)
        w2 = cnt[hic] * cnt[aic] * bet
        if w2.sum(): two.append(float(np.sum(w2 * cover) / w2.sum()))
    res['betting_closing'][k] = dict(bets=n, win_rate=round(pw, 4),
                                     binomial_ci=[round(pw - 1.96 * se, 4), round(pw + 1.96 * se, 4)],
                                     iid_game_bootstrap_ci=ci(iid), home_cluster_ci=ci(clus), two_way_ci=ci(two))

json.dump(res, open(ROOT / 'results/paper2_market2.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
