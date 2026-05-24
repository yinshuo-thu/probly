"""
Expand fixture_market_matches by finding more Polymarket football markets
that match our 195 available LSports fixtures.

Uses Polymarket Gamma API (gamma-api.polymarket.com) for search,
then CLOB to verify token availability.
"""
import sys, json, re, requests, time
from pathlib import Path
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

BASE     = Path('/Volumes/T7/probly')
MSGS_DIR = BASE / 'data/hyper/football'
FMM_FILE = BASE / 'data/polymarket/fixture_market_matches.parquet'
GAMMA    = 'https://gamma-api.polymarket.com'
CLOB     = 'https://clob.polymarket.com'


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def team_in_question(team: str, question: str) -> bool:
    q = question.lower()
    t = team.lower()
    # Try full name and common abbreviations
    words = t.split()
    return any(w in q for w in words if len(w) > 3)


def fetch_gamma_page(offset: int, limit: int = 200) -> list:
    try:
        r = requests.get(f'{GAMMA}/markets', params={
            'tag_id': 12,  # soccer/football tag
            'limit': limit,
            'offset': offset,
            'closed': 'true',
        }, timeout=20)
        if r.ok:
            return r.json()
    except Exception as e:
        print(f'  gamma error at offset={offset}: {e}')
    return []


def get_clob_tokens(condition_id: str) -> list:
    try:
        r = requests.get(f'{CLOB}/markets/{condition_id}', timeout=10)
        if r.ok:
            tokens = r.json().get('tokens', [])
            return [t['token_id'] for t in tokens]
    except Exception:
        pass
    return []


def load_lsports_fixtures():
    """Load all LSports fixtures with metadata."""
    rows = []
    for path in MSGS_DIR.glob('*/*/messages.parquet'):
        fid  = int(path.parent.name)
        date = path.parent.parent.name
        try:
            msgs = pd.read_parquet(path, columns=['fixture_id','home','away','event_date'])
            r    = msgs.iloc[0]
            rows.append({
                'fixture_id': fid,
                'date':       date,
                'home':       str(r.home),
                'away':       str(r.away),
                'event_date': str(r.event_date),
                'path':       str(path),
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


def main():
    fmm = pd.read_parquet(FMM_FILE)
    already_matched = set(fmm['fixture_id'].astype(int).unique())
    print(f'Already matched: {len(already_matched)} fixtures')

    fixtures = load_lsports_fixtures()
    unmatched = fixtures[~fixtures['fixture_id'].isin(already_matched)].copy()
    print(f'LSports fixtures to match: {len(unmatched)}')

    # Fetch Polymarket football markets
    print('Fetching Polymarket football markets...')
    all_markets = []
    for offset in range(0, 2000, 200):
        page = fetch_gamma_page(offset)
        if not page:
            break
        all_markets.extend(page)
        print(f'  Fetched {len(all_markets)} markets so far...')
        time.sleep(0.2)

    print(f'Total Polymarket football markets: {len(all_markets)}')

    # Filter to 2026 markets (our LSports data)
    markets_2026 = []
    for m in all_markets:
        end = m.get('endDate', '') or m.get('end_date_iso', '')
        if '2026' in str(end):
            markets_2026.append(m)
    print(f'Markets from 2026: {len(markets_2026)}')

    # Try to match each unmatched fixture
    new_rows = []
    for _, fix in unmatched.iterrows():
        home  = fix['home']
        away  = fix['away']
        date  = fix['date']  # YYYY-MM-DD

        best_score  = 0.4   # minimum threshold
        best_market = None

        for m in markets_2026:
            q    = m.get('question', '')
            end  = m.get('endDate', '')[:10]
            if abs((pd.Timestamp(end) - pd.Timestamp(date)).days) > 2:
                continue
            # Score based on team name presence
            h_score = max(sim(home, q), int(team_in_question(home, q)) * 0.6)
            a_score = max(sim(away, q), int(team_in_question(away, q)) * 0.6)
            score   = (h_score + a_score) / 2
            if score > best_score:
                best_score  = score
                best_market = m

        if best_market is not None:
            cid = best_market.get('conditionId', best_market.get('condition_id', ''))
            if not cid:
                continue
            tokens = get_clob_tokens(cid)
            if not tokens:
                continue
            q = best_market.get('question', '')
            new_rows.append({
                'fixture_id':    fix['fixture_id'],
                'condition_id':  cid,
                'question':      q,
                'market_type':   'match_winner',
                'home':          fix['home'],
                'away':          fix['away'],
                'event_date':    fix['event_date'],
                'match_score':   round(best_score, 3),
            })
            print(f'  MATCHED: {home} vs {away} ({date}) → "{q[:60]}" (score={best_score:.2f})')
        else:
            print(f'  NO MATCH: {home} vs {away} ({date})')

    if new_rows:
        new_df  = pd.DataFrame(new_rows)
        combined = pd.concat([fmm, new_df], ignore_index=True)
        combined.to_parquet(FMM_FILE, index=False)
        print(f'\nAdded {len(new_rows)} new fixture-market matches')
        print(f'Total: {combined["fixture_id"].nunique()} fixtures')
    else:
        print('No new matches found')


if __name__ == '__main__':
    main()
