"""The four recomputations the late window of Section 4 reaches, over the report series read whole.

Published unchanged below its original header. It reads the originating project's paths, so
it is not rerun here; its outputs are `results/paper2_cutoff_curve_late_window.json`, `results/paper2_cutoff_f7_late_window.json`, `results/paper2_example_aggregate_late_window.json` and `results/paper2_cutoff_units.json`, which carry point estimates, intervals and counts and no player rows.

The original header follows.

Recompute the quantities the late window of Section 4 reaches, with the scripts that first produced them.

    python -I late_window_runs.py

The report series of `data/league/paper2_ext/full_series` runs from midnight Eastern to 90 minutes before
each game day's last tip-off, which is where collection stopped and not where the league stops publishing.
The remaining window was collected afterwards into `data/league/paper2_ext/late_window/series` in the same
schema. Folding it in cannot move a quantity read at six hours or at ninety minutes, because a file printed
after that endpoint cannot satisfy an earlier cutoff; what it reaches is the one-hour and thirty-minute
readings, which a game near the day's last tip previously took from the last report held rather than the
last one published.

This runs plan I's cutoff curve, plan N's worked-example aggregate, plan Y's F7 crossing and plan CC's
decision units over the concatenated series. Each is run from its own committed source with a listed set of
substitutions and nothing else: where the series is read, where the result is written, and -- for plan Y,
whose controls are its own published pre-collection figures -- the one control value the collection moves.
Every substitution must match exactly once or the run stops, so a script that has been edited since fails
loudly here rather than being redirected into something it no longer is.

The frozen pre-collection files are not touched. These write the `*_late_window` siblings beside them,
except plan CC's `paper2_cutoff_units.json`, which belongs to the conference paper alone and was replaced
rather than duplicated. Nothing is written into the audited `results/` of the main project.

The outputs are committed, so reproduction is checked by rerunning this and reading `git status`: a run that
reproduces them leaves the working tree clean.
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE / 'results'
SCRIPTS = ROOT / 'league/scripts'
LW = ROOT / 'data/league/paper2_ext/late_window/series'

# The series, read as one. Both files carry the same columns, and the selection of a report per team-game
# is what changes: `listed` now holds the blocks published inside the last 90 minutes as well.
_MERGED = """pd.concat([pd.read_csv(FS / 'injury_%(w)s.csv', dtype={'game_id': str}),
    pd.read_csv(Path(r'%(lw)s') / 'injury_%(w)s.csv', dtype={'game_id': str})], ignore_index=True)"""


def _merged(what: str) -> str:
    return _MERGED % {'w': what, 'lw': LW}


# plan I and plan N read the series on one line; plan Y reads it on two
ONE_LINE = ("rows = pd.read_csv(FS / 'injury_rows.csv', dtype={'game_id': str}); "
            "blocks = pd.read_csv(FS / 'injury_blocks.csv', dtype={'game_id': str})",
            f"rows = {_merged('rows')}\nblocks = {_merged('blocks')}")
TWO_LINE = ("""rows = pd.read_csv(FS / 'injury_rows.csv', dtype={'game_id': str})
blocks = pd.read_csv(FS / 'injury_blocks.csv', dtype={'game_id': str})""",
            f"rows = {_merged('rows')}\nblocks = {_merged('blocks')}")

# plan Y's rules Y-a and Y-b hold it to its published figures, and the paired one-hour contrast is exactly
# what this collection moves: 0.0666 [0.0305, 0.1031] before it, 0.0655 [0.0291, 0.1018] on the series
# read whole. The six-hour controls are unchanged, which is the check that nothing earlier moved.
Y_CONTROL = ("""CONTROL = {'gain_F1_6h': 0.1852, 'infl_6h': 0.0502, 'paired_F1': 0.0666,
           'paired_F1_ci': [0.0305, 0.1031], 'n': 4131, 'strict_F7_gain': 0.2198}""",
             """CONTROL = {'gain_F1_6h': 0.1852, 'infl_6h': 0.0502, 'paired_F1': 0.0655,
           'paired_F1_ci': [0.0291, 0.1018], 'n': 4131, 'strict_F7_gain': 0.2198}""")

CF7_TMP = SCRIPTS / '_late_window_cf7.py'

# what each output says about itself, so a file read on its own says which series it is on
SIBLING_NOTE = (
    '%s, run on the merged report series: the collected series of league/public/sources.csv plus the 1,787'
    " files of league/public/sources_late_window.csv, which cover the window from 90 minutes before each"
    " game day's last tip-off to the tip itself. The frozen counterpart without the `_late_window` suffix"
    ' is the collected-series run that the journal manuscript reports. Quantities at six hours and at'
    ' ninety minutes are identical in the two by construction, and results/paper2_late_window.json records'
    ' that check.')
UNITS_NOTE = (
    'Run on the merged report series of results/paper2_late_window.json: the collected series plus the'
    " 1,787 files of the window from 90 minutes before each day's last tip-off to the tip itself. Produced"
    " by the late-window runner, league/scripts/late_window_runs.py in the released repository, which runs"
    " plan Y's build over the two series"
    " read as one and restates the one-hour control of that script, which this collection moves; plan CC's"
    ' own control carries the recomputed value and holds. The six-hour controls are the published ones and'
    ' reproduced exactly.')

RUNS = [
    dict(label='plan I, the cutoff curve', source=SCRIPTS / 'p6_ds_cutoff_curve.py',
         tmp=SCRIPTS / '_late_window_curve.py', call_main=False,
         out=RES / 'paper2_cutoff_curve_late_window.json', note_key='late_window_note',
         subs=[ONE_LINE,
               ("json.dump(res, open(ROOT / 'results/paper2_cutoff_curve.json', 'w'), indent=1)",
                "json.dump(res, open(Path(r'%s') / 'paper2_cutoff_curve_late_window.json', 'w'), indent=1)"
                % RES)]),
    dict(label='plan N, the worked-example aggregate', source=SCRIPTS / 'p6_ds_example_aggregate.py',
         tmp=SCRIPTS / '_late_window_aggregate.py', call_main=False,
         out=RES / 'paper2_example_aggregate_late_window.json', note_key='late_window_note',
         subs=[ONE_LINE,
               ("json.dump(res, open(ROOT / 'results/paper2_example_aggregate.json', 'w'), indent=1)",
                "json.dump(res, open(Path(r'%s') / 'paper2_example_aggregate_late_window.json', 'w'),"
                ' indent=1)' % RES)]),
    dict(label='plan Y, the F7 crossing', source=SCRIPTS / 'p6_ds_cutoff_f7.py', tmp=CF7_TMP,
         call_main=True, out=RES / 'paper2_cutoff_f7_late_window.json', note_key='late_window_note',
         subs=[TWO_LINE, Y_CONTROL,
               ("OUT = ROOT / 'results/paper2_cutoff_f7.json'",
                "OUT = Path(r'%s') / 'paper2_cutoff_f7_late_window.json'" % RES)]),
    # plan CC imports plan Y's script by path and builds the sample through it, so it is pointed at the
    # patched copy above rather than patched again itself. Its own control already carries 0.0655, and its
    # output is this directory's, not the audited results/. `built_from` is restored to the committed
    # script, which is what it was built from: the patched copy is that script plus the substitutions above.
    dict(label='plan CC, the decision units', source=HERE / 'cutoff_units.py',
         tmp=HERE / '_late_window_units.py', call_main=True, out=RES / 'paper2_cutoff_units.json',
         note_key='built_from_note', built_from='league/scripts/p6_ds_cutoff_f7.py',
         subs=[("CF7 = ROOT / 'league/scripts/p6_ds_cutoff_f7.py'", "CF7 = Path(r'%s')" % CF7_TMP)]),
]


def patch(source: Path, subs: list[tuple[str, str]]) -> str:
    """The source with every substitution applied, or a loud failure naming the one that did not match."""
    src = source.read_text(encoding='utf-8')
    for old, new in subs:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f'{source.name} no longer matches this runner: {n} matches of {old[:70]!r}')
        src = src.replace(old, new, 1)
    return src


def stamp(out: Path, note_key: str, note: str, built_from: str | None) -> None:
    """Record on the file itself which series it was read on, as the last key, and nothing else."""
    res = json.loads(out.read_text(encoding='utf-8'))
    if built_from:
        res['built_from'] = built_from
    res[note_key] = note
    out.write_text(json.dumps(res, indent=1), encoding='utf-8')


def run(label: str, source: Path, tmp: Path, subs, call_main: bool, out: Path, note_key: str,
        built_from: str | None = None) -> None:
    print(f'\n=== {label}: {source.name}', flush=True)
    tmp.write_text(patch(source, subs), encoding='utf-8')
    spec = importlib.util.spec_from_file_location(tmp.stem, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[tmp.stem] = mod          # plan Y's module is imported again by plan CC
    spec.loader.exec_module(mod)         # plan I and plan N write on import
    if call_main and mod.main() != 0:
        raise SystemExit(f'{source.name} returned non-zero; nothing further is run')
    note = UNITS_NOTE if note_key == 'built_from_note' else SIBLING_NOTE % f'league/scripts/{source.name}'
    stamp(out, note_key, note, built_from)
    print(f'  -> {out.relative_to(ROOT)}', flush=True)


def main() -> int:
    if not LW.is_dir():
        raise SystemExit(f'the collected late window is not here: {LW}')
    try:
        for kw in RUNS:
            run(**kw)
    finally:
        for kw in RUNS:
            kw['tmp'].unlink(missing_ok=True)
    print('\na run that reproduces the committed files leaves git status clean')
    return 0


if __name__ == '__main__':
    sys.exit(main())
