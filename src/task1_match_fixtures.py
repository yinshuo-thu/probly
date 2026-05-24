#!/usr/bin/env python3
"""Task 1: Match LSports fixtures to Polymarket markets using fuzzy matching."""

import json
import re
from difflib import SequenceMatcher
from collections import defaultdict
import pandas as pd

print("=== Task 1: Matching LSports fixtures to Polymarket markets ===")

# Load data
fixture_df = pd.read_parquet('/Volumes/T7/probly/data/fixture_index.parquet')
print(f"Loaded {len(fixture_df)} LSports fixtures")

with open('/Volumes/T7/probly/data/polymarket/football_markets.json') as f:
    markets = json.load(f)
print(f"Loaded {len(markets)} Polymarket markets")

# Parse Polymarket questions to extract home/away team names
def parse_question(question):
    """Extract home and away team from 'Home vs. Away: Market Type'"""
    match = re.match(r'^(.+?)\s+vs\.\s+(.+?):\s+(.+)$', question)
    if match:
        return match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
    return None, None, None

# Group markets by (home, away) pair
market_groups = defaultdict(list)
for m in markets:
    home, away, market_type = parse_question(m['question'])
    if home and away:
        key = (home.lower(), away.lower())
        market_groups[key].append(m)

print(f"Unique (home, away) pairs in Polymarket: {len(market_groups)}")

# Fuzzy matching helper
def fuzzy_score(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def best_match(name, candidates, threshold=0.65):
    """Find best matching candidate for a name."""
    best_score = 0
    best_cand = None
    for cand in candidates:
        score = fuzzy_score(name, cand)
        if score > best_score:
            best_score = score
            best_cand = cand
    if best_score >= threshold:
        return best_cand, best_score
    return None, 0

# Build candidate list from polymarket (lower-cased unique team names)
pm_home_names = list(set(k[0] for k in market_groups.keys()))
pm_away_names = list(set(k[1] for k in market_groups.keys()))
all_pm_teams = list(set(pm_home_names + pm_away_names))

# Match each LSports fixture to polymarket
THRESHOLD = 0.65
matched_records = []
match_cache = {}  # Cache team name matches

def get_cached_match(lsports_name, pm_list):
    key = lsports_name.lower()
    if key not in match_cache:
        match_cache[key] = best_match(lsports_name, pm_list, THRESHOLD)
    return match_cache[key]

print(f"\nRunning fuzzy matching (threshold={THRESHOLD})...")

for idx, row in fixture_df.iterrows():
    ls_home = str(row['home'])
    ls_away = str(row['away'])

    # Try to find matching PM pair
    best_pair = None
    best_combined_score = 0

    # For each PM (home, away) pair, compute combined score
    for (pm_home, pm_away), mlist in market_groups.items():
        score_h = fuzzy_score(ls_home, pm_home)
        score_a = fuzzy_score(ls_away, pm_away)
        if score_h >= THRESHOLD and score_a >= THRESHOLD:
            combined = (score_h + score_a) / 2
            if combined > best_combined_score:
                best_combined_score = combined
                best_pair = (pm_home, pm_away, mlist)

    if best_pair:
        pm_home, pm_away, mlist = best_pair
        condition_ids = [m['condition_id'] for m in mlist]
        questions = [m['question'] for m in mlist]
        matched_records.append({
            'fixture_id': row['fixture_id'],
            'event_date': row['event_date'],
            'home': row['home'],
            'away': row['away'],
            'league_name': row['league_name'],
            'start_date_utc': row['start_date_utc'],
            'pm_home': pm_home,
            'pm_away': pm_away,
            'match_score': round(best_combined_score, 3),
            'condition_ids': condition_ids,
            'questions': questions
        })

print(f"\nMatched {len(matched_records)} fixtures")

if matched_records:
    matched_df = pd.DataFrame(matched_records)
    # Sort by match score desc
    matched_df = matched_df.sort_values('match_score', ascending=False)
    print(f"\nTop 10 matches:")
    for _, r in matched_df.head(10).iterrows():
        print(f"  [{r['match_score']:.3f}] {r['home']} vs {r['away']} -> {r['pm_home']} vs {r['pm_away']} ({len(r['condition_ids'])} markets)")

    # Save
    out_path = '/Volumes/T7/probly/data/matched_fixtures.parquet'
    matched_df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")

    league_counts = matched_df['league_name'].value_counts().head(10)
    print(f"\nTop leagues in matched fixtures:")
    print(league_counts.to_string())
else:
    print("No matches found! Check thresholds.")

print("\n=== Task 1 Complete ===")
