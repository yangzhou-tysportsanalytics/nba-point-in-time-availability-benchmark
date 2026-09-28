"""Column-position parser for official NBA injury report PDFs (2023-24 onward layout).

Each word is assigned to a column by the x position of the header words. Date, time, matchup and
team carry forward; a line with a status word starts a player row; reason-only lines attach to the
vertically nearest player row. Teams shown as NOT YET SUBMITTED are recorded as such.
"""
import csv, os, re, sys
import pdfplumber

COLS = ['GameDate', 'GameTime', 'Matchup', 'Team', 'PlayerName', 'CurrentStatus', 'Reason']
# the words a reason can begin with; anything else at the start of a line is a continuation
CATEGORY = re.compile(r"(Injury/Illness|G League|Rest|Personal Reasons|Not [Ww]ith Team|Suspension|"
                      r"League Suspension|Team Suspension|Trade Pending|Concussion Protocol|Ineligible|"
                      r"Return to Competition|Health and Safety|Coach's Decision)\b")
STATUSES = {'Out', 'Questionable', 'Doubtful', 'Probable', 'Available'}
TEAMS = {'AtlantaHawks': 'ATL', 'BostonCeltics': 'BOS', 'BrooklynNets': 'BKN', 'CharlotteHornets': 'CHA', 'ChicagoBulls': 'CHI',
         'ClevelandCavaliers': 'CLE', 'DallasMavericks': 'DAL', 'DenverNuggets': 'DEN', 'DetroitPistons': 'DET',
         'GoldenStateWarriors': 'GSW', 'HoustonRockets': 'HOU', 'IndianaPacers': 'IND', 'LAClippers': 'LAC',
         'LosAngelesClippers': 'LAC', 'LosAngelesLakers': 'LAL', 'MemphisGrizzlies': 'MEM', 'MiamiHeat': 'MIA',
         'MilwaukeeBucks': 'MIL', 'MinnesotaTimberwolves': 'MIN', 'NewOrleansPelicans': 'NOP', 'NewYorkKnicks': 'NYK',
         'OklahomaCityThunder': 'OKC', 'OrlandoMagic': 'ORL', 'Philadelphia76ers': 'PHI', 'PhoenixSuns': 'PHX',
         'PortlandTrailBlazers': 'POR', 'SacramentoKings': 'SAC', 'SanAntonioSpurs': 'SAS', 'TorontoRaptors': 'TOR',
         'UtahJazz': 'UTA', 'WashingtonWizards': 'WAS'}


def lines_of(page):
    words = page.extract_words(keep_blank_chars=False, use_text_flow=False, x_tolerance=1.5)
    rows = []
    for w in sorted(words, key=lambda w: (round(w['top']), w['x0'])):
        if rows and abs(rows[-1][0] - w['top']) < 3:
            rows[-1][1].append(w)
        else:
            rows.append([w['top'], [w]])
    return rows


def parse(path):
    out, blocks = [], []
    bounds = None
    carry = None  # last player row of the previous page, for reasons that continue across the break
    cur = dict(GameDate='', GameTime='', Matchup='', Team='')
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pending = []  # reason-only fragments (top, text)
            players = []
            for top, ws in lines_of(page):
                texts = [w['text'] for w in ws]
                if 'Matchup' in texts and 'Reason' in texts:
                    games = [w['x0'] for w in ws if w['text'] == 'Game']
                    xs = {w['text']: w['x0'] for w in ws}
                    bounds = [games[0], games[1], xs['Matchup'], xs['Team'], xs['Player'], xs['Current'], xs['Reason']]
                    continue
                if bounds is None or 'Report:' in texts or (any(x.startswith('Page') for x in texts) and 'of' in ' '.join(texts)):
                    continue
                cells = {c: [] for c in COLS}
                for w in ws:
                    i = max(k for k, b in enumerate(bounds) if w['x0'] >= b - 4) if w['x0'] >= bounds[0] - 4 else 0
                    cells[COLS[i]].append(w['text'])
                cells = {c: ' '.join(v) for c, v in cells.items()}
                if cells['GameDate'].startswith('Page') or 'Page' in cells['Reason'] and 'of' in cells['Reason']:
                    continue
                for c in ('GameDate', 'GameTime', 'Matchup'):
                    if cells[c]: cur[c] = cells[c]
                if cells['Team']:
                    cur['Team'] = cells['Team'].replace(' ', '')
                    if cur['Team'] in TEAMS or 'NOTYETSUBMITTED' in (cells['PlayerName'] + cells['Reason']).replace(' ', ''):
                        pass
                joined = (cells['PlayerName'] + cells['CurrentStatus'] + cells['Reason']).replace(' ', '')
                if 'NOTYETSUBMITTED' in joined:
                    blocks.append(dict(file=os.path.basename(path), date=cur['GameDate'], matchup=cur['Matchup'],
                                       team=TEAMS.get(cur['Team'], cur['Team']), state='not_yet_submitted'))
                    continue
                status = cells['CurrentStatus'].split()[0] if cells['CurrentStatus'] else ''
                if status in STATUSES and cells['PlayerName']:
                    row = dict(file=os.path.basename(path), date=cur['GameDate'], time=cur['GameTime'], matchup=cur['Matchup'],
                               team=TEAMS.get(cur['Team'], cur['Team']), player=cells['PlayerName'],
                               status=status, reason=cells['Reason'], top=top)
                    players.append(row); out.append(row)
                    blocks.append(dict(file=row['file'], date=row['date'], matchup=row['matchup'], team=row['team'], state='listed'))
                elif cells['Reason'] and not cells['PlayerName']:
                    pending.append((top, cells['Reason']))
            # A reason-only line is either the first line of a wrapped reason, whose player name is centred
            # and therefore printed below it, or the continuation of an earlier row -- which at the top of a
            # page is the last row of the previous page, possibly another team's. Tell them apart by the text:
            # a line that begins a reason starts with a category, a continuation does not.
            # See league/tests/test_parse_page_continuation.py for one PDF of each shape.
            for top, text in pending:
                below = next((r for r in players if r['top'] >= top), None)
                above = [r for r in players if r['top'] < top]
                starts_reason = bool(CATEGORY.match(text))
                if starts_reason and below is not None and not CATEGORY.match(below['reason']):
                    below['reason'] = (text + ' ' + below['reason']).strip()
                    continue
                target = above[-1] if above else carry
                if target is not None:
                    target['reason'] = (target['reason'] + ' ' + text).strip()
                elif below is not None:  # nothing above it anywhere: keep the text rather than drop it
                    below['reason'] = (text + ' ' + below['reason']).strip()
            if players:
                carry = players[-1]
    for r in out: r.pop('top', None)
    return out, blocks


if __name__ == '__main__':
    files = sorted(os.listdir(sys.argv[1]))
    allr, allb, fails = [], [], []
    for f in files:
        try:
            r, b = parse(os.path.join(sys.argv[1], f)); allr += r; allb += b
            if not r and not b: fails.append((f, 'empty'))
        except Exception as e:
            fails.append((f, repr(e)[:200]))
    for name, rows in (('reparsed_rows.csv', allr), ('reparsed_blocks.csv', allb)):
        with open(name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open('reparse_failures.csv', 'w', newline='') as fh:
        csv.writer(fh).writerows([('file', 'error')] + fails)
    print(len(files), 'files', len(allr), 'rows', len(fails), 'failures')
