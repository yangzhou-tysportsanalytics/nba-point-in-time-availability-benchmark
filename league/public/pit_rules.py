"""Rules used by the public rebuild path, copied verbatim from the committed project code
(league/src/build.py: name_key, name_keys, reason_class; src/measurement_rules.py: reference_minutes).
tests/test_pit_rules.py checks that the copies still match the originals when those are present.
"""
import re, unicodedata

def name_key(name):
    s = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode().lower()
    if ',' in s:
        last, first = s.split(',', 1)
        s = first + ' ' + last
    return re.sub('[^a-z0-9]', '', s)


SUFFIX = re.compile(r'(jr|sr|ii|iii|iv)$')


def name_keys(name):
    """Progressively looser keys: exact, suffix-stripped, last name + first initial."""
    s = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode().lower()
    if ',' in s:
        last, first = s.split(',', 1)
    else:
        parts = s.split(); first, last = (parts[0], ' '.join(parts[1:])) if len(parts) > 1 else ('', s)
    clean = lambda x: re.sub('[^a-z0-9]', '', x)
    last_c = SUFFIX.sub('', clean(last)) or clean(last)
    return [clean(first) + clean(last), clean(first) + last_c, last_c + ':' + clean(first)[:1]]


def reason_class(reason):
    """Official report reasons are often split across PDF lines, leaving fragments such as
    'Strain' or 'Surgery'; any non-empty reason outside the listed non-medical classes is medical.
    A bare 'Management' fragment is the tail of 'Injury Management' (confirmed against the audited
    four-team reports, where these rows are medical), so it is medical."""
    r = str(reason).lower().replace(' ', '').strip(';')
    if r in ('', 'nan', '-'): return 'unknown'
    if 'gleague' in r: return 'gleague'
    if 'suspension' in r: return 'suspension'
    if 'tradepending' in r: return 'trade_pending'
    if 'ineligible' in r: return 'ineligible'
    if 'personal' in r or 'notwithteam' in r: return 'personal'
    if r.startswith('rest') and 'injury' not in r: return 'rest'
    return 'medical'


def reference_minutes(players,total=240):
    ps=[p for p in players if p['supported']]
    if sum(p['capacity_minutes'] for p in ps)<total-1e-9:return None
    remaining=total;result={};active=list(ps)
    while active:
        weight=sum(p['mean_minutes'] for p in active)
        if weight<=0:return None
        lam=remaining/weight
        capped=[p for p in active if lam*p['mean_minutes']>p['capacity_minutes']+1e-10]
        if not capped:
            result.update({p['canonical_player_id']:lam*p['mean_minutes'] for p in active});break
        for p in capped:
            result[p['canonical_player_id']]=p['capacity_minutes'];remaining-=p['capacity_minutes']
        ids={p['canonical_player_id'] for p in capped};active=[p for p in active if p['canonical_player_id'] not in ids]
    if abs(sum(result.values())-total)>1e-7:raise ValueError('reference minute balance failed')
    return result
