"""
Task 2: Match LSports fixtures to Polymarket markets.

Parses team names from market questions and fuzzy-matches against fixture_index.
"""

import json
import re
import pandas as pd
from difflib import SequenceMatcher
from datetime import datetime, timezone

MARKETS_PATH = "data/polymarket/football_markets.json"
FIXTURE_PATH = "data/fixture_index.parquet"
OUTPUT_PATH = "data/polymarket/fixture_market_matches.parquet"

# Tokens to strip when normalizing team names
NOISE_TOKENS = re.compile(
    r'\b(fc|sc|ac|cf|rc|fk|sk|bk|if|hb|sf|afc|sfc|rsc|vfb|vfl|bvb|asc|'
    r'united|city|town|athletic|athletics|sport|sporting|'
    r'borussia|real|atletico|deportivo|club|de|la|le|les)\b',
    re.IGNORECASE,
)


def normalize(name: str) -> str:
    """Lowercase, remove noise tokens and punctuation, collapse whitespace."""
    name = name.lower()
    name = re.sub(r"[''´`]", "", name)
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    name = NOISE_TOKENS.sub(" ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def best_match(query: str, candidates: list[str]) -> tuple[str, float]:
    """Return (best_candidate, score) for a query against a list of candidates."""
    best, score = "", 0.0
    qn = normalize(query)
    for c in candidates:
        cn = normalize(c)
        s = SequenceMatcher(None, qn, cn).ratio()
        if s > score:
            score = s
            best = c
    return best, score


def parse_question(question: str) -> tuple[str, str, str]:
    """Return (team_a, team_b, market_type) from 'Team A vs. Team B: type'."""
    if ": " in question:
        matchup, market_type = question.rsplit(": ", 1)
    else:
        matchup, market_type = question, "unknown"

    if " vs. " in matchup:
        team_a, team_b = matchup.split(" vs. ", 1)
    elif " vs " in matchup:
        team_a, team_b = matchup.split(" vs ", 1)
    else:
        team_a, team_b = matchup, ""

    return team_a.strip(), team_b.strip(), market_type.strip()


def market_date_to_ymd(date_str: str) -> str:
    """Convert '2026-05-10T18' → '2026-05-10'."""
    return date_str[:10]


def main():
    print("Loading data...")
    with open(MARKETS_PATH) as f:
        markets = json.load(f)

    fixtures = pd.read_parquet(FIXTURE_PATH)

    # Build per-date fixture lookup
    fixtures["event_date"] = fixtures["event_date"].astype(str)
    fixtures["home_norm"] = fixtures["home"].apply(normalize)
    fixtures["away_norm"] = fixtures["away"].apply(normalize)

    # Deduplicate markets by matchup (keep one representative condition_id per game-type)
    print(f"Total markets: {len(markets)}")

    # Group markets by (team_a, team_b, date) to get unique matches
    seen_matchups: dict[tuple, list] = {}
    for m in markets:
        ta, tb, mt = parse_question(m["question"])
        date_ymd = market_date_to_ymd(m["date"])
        key = (ta.lower(), tb.lower(), date_ymd)
        seen_matchups.setdefault(key, []).append(m)

    print(f"Unique (teamA, teamB, date) combos: {len(seen_matchups)}")

    # For each unique match, find best fixture
    matches = []
    SCORE_THRESHOLD = 0.50  # minimum fuzzy match score for each team

    fixture_list = fixtures.to_dict("records")

    processed = 0
    for (ta, tb, date_ymd), mkt_list in seen_matchups.items():
        # Filter fixtures to same date (±1 day tolerance)
        date_fixtures = [
            fx for fx in fixture_list
            if fx["event_date"] == date_ymd
        ]
        if not date_fixtures:
            # Try adjacent dates
            try:
                d = datetime.strptime(date_ymd, "%Y-%m-%d")
                prev_d = d.replace(day=d.day - 1).strftime("%Y-%m-%d")
                next_d = d.replace(day=d.day + 1).strftime("%Y-%m-%d")
            except Exception:
                date_fixtures = []
            else:
                date_fixtures = [
                    fx for fx in fixture_list
                    if fx["event_date"] in (prev_d, date_ymd, next_d)
                ]

        if not date_fixtures:
            continue

        ta_norm = normalize(ta)
        tb_norm = normalize(tb)

        # Score each candidate fixture
        best_score = 0.0
        best_fx = None
        for fx in date_fixtures:
            s_home = SequenceMatcher(None, ta_norm, fx["home_norm"]).ratio()
            s_away = SequenceMatcher(None, tb_norm, fx["away_norm"]).ratio()
            s_total = (s_home + s_away) / 2.0
            # Also try reversed (market sometimes lists away first)
            s_home_r = SequenceMatcher(None, tb_norm, fx["home_norm"]).ratio()
            s_away_r = SequenceMatcher(None, ta_norm, fx["away_norm"]).ratio()
            s_total_r = (s_home_r + s_away_r) / 2.0
            s = max(s_total, s_total_r)
            if s > best_score and min(s_home, s_away) > SCORE_THRESHOLD - 0.15 or \
               s > best_score and min(s_home_r, s_away_r) > SCORE_THRESHOLD - 0.15:
                best_score = s
                best_fx = fx

        if best_fx is None or best_score < SCORE_THRESHOLD:
            continue

        # Add one row per market condition_id for this match
        for m in mkt_list:
            _, _, mt = parse_question(m["question"])
            matches.append({
                "fixture_id": best_fx["fixture_id"],
                "condition_id": m["condition_id"],
                "question": m["question"],
                "market_type": mt,
                "home": best_fx["home"],
                "away": best_fx["away"],
                "market_team_a": ta,
                "market_team_b": tb,
                "event_date": best_fx["event_date"],
                "start_date_utc": best_fx["start_date_utc"],
                "league_name": best_fx["league_name"],
                "match_score": round(best_score, 4),
                "market_date": date_ymd,
                "market_hour": m["date"][11:] if len(m["date"]) > 10 else "18",
            })

        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed}/{len(seen_matchups)} unique matchups, {len(matches)} matches so far")

    print(f"\nTotal matches found: {len(matches)}")

    if not matches:
        print("WARNING: No matches found!")
        return

    result = pd.DataFrame(matches)
    result = result.sort_values("match_score", ascending=False)

    print("\nTop 20 matches:")
    print(result[["home", "away", "market_team_a", "market_team_b", "event_date", "match_score", "league_name"]].head(20).to_string())

    print(f"\nScore distribution:")
    print(result["match_score"].describe())

    # Save
    import os
    os.makedirs("data/polymarket", exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result)} rows to {OUTPUT_PATH}")

    # Summary: unique fixtures matched
    n_fixtures = result["fixture_id"].nunique()
    print(f"Unique fixtures matched: {n_fixtures}")

    # Show score threshold breakdown
    for thresh in [0.7, 0.75, 0.8, 0.85, 0.9]:
        n = (result["match_score"] >= thresh).sum()
        print(f"  score >= {thresh}: {n} markets ({result[result['match_score'] >= thresh]['fixture_id'].nunique()} fixtures)")


if __name__ == "__main__":
    main()
