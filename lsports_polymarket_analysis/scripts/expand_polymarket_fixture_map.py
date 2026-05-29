"""
Expand config/polymarket_fixture_map.csv from cached LSports fixtures and
Polymarket Gamma soccer events.

The output schema matches the BBO analysis scripts:
fixture_id,event_date,team_home,team_away,polymarket_slug,primary_market_slug,
primary_market_id,condition_id,polymarket_market_id,outcome
"""
import _bootstrap  # noqa: F401

import argparse
import glob
import os
import re
import sys
import time
from difflib import SequenceMatcher

import pandas as pd
import requests

from src import load_config, resolve_path


NOISE = re.compile(
    r"\b(fc|sc|ac|cf|rc|fk|sk|bk|if|afc|club|de|la|the|united|city|town)\b",
    re.IGNORECASE,
)


def norm(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = NOISE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def safe(text) -> str:
    return str(text).encode(sys.stdout.encoding or "utf-8", "replace").decode(
        sys.stdout.encoding or "utf-8"
    )


def score_pair(a_home: str, a_away: str, b_home: str, b_away: str) -> tuple[float, bool]:
    ah, aa, bh, ba = map(norm, [a_home, a_away, b_home, b_away])
    direct = (
        SequenceMatcher(None, ah, bh).ratio()
        + SequenceMatcher(None, aa, ba).ratio()
    ) / 2
    reverse = (
        SequenceMatcher(None, ah, ba).ratio()
        + SequenceMatcher(None, aa, bh).ratio()
    ) / 2
    return (reverse, True) if reverse > direct else (direct, False)


def load_cached_fixtures() -> pd.DataFrame:
    pattern = (
        "lsports_polymarket_analysis/data/raw/sport_id=6046__sport=Football/"
        "event_date=*/fixture_id=*/**/fixtures.parquet"
    )
    rows = []
    for path in glob.glob(pattern, recursive=True):
        if ".cache" in path:
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        rows.append({
            "fixture_id": str(row.get("fixture_id")),
            "event_date": str(row.get("event_date"))[:10],
            "team_home": row.get("home"),
            "team_away": row.get("away"),
            "start_date_utc": row.get("start_date_utc"),
            "league_name": row.get("league_name"),
        })
    return pd.DataFrame(rows).drop_duplicates("fixture_id")


def fetch_events(gamma_base: str, start_date: str, end_date: str,
                 max_events: int) -> list[dict]:
    events = []
    for offset in range(0, max_events, 100):
        params = {
            "closed": "true",
            "limit": 100,
            "offset": offset,
            "end_date_min": start_date,
            "end_date_max": end_date,
            "order": "endDate",
            "ascending": "true",
        }
        r = requests.get(f"{gamma_base}/events", params=params, timeout=30)
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        events.extend(page)
        print(f"[gamma] fetched {len(events)} events")
        if len(page) < 100:
            break
        time.sleep(0.15)
    return events


def is_match_event(event: dict) -> bool:
    title = str(event.get("title") or "")
    if " vs. " not in title and " vs " not in title:
        return False
    sport = str(event.get("sport") or "").lower()
    tags = " ".join(str(t.get("label") or t.get("slug") or "") for t in event.get("tags", []))
    hay = f"{title} {sport} {tags}".lower()
    excluded = ["nfl", "nba", "mlb", "nhl", "ufc", "tennis", "cs2", "dota", "lol"]
    if any(x in hay for x in excluded):
        return False
    return True


def event_teams(event: dict) -> tuple[str, str]:
    teams = event.get("teams") or []
    if len(teams) >= 2:
        names = [t.get("name") if isinstance(t, dict) else str(t) for t in teams[:2]]
        return str(names[0]), str(names[1])
    title = str(event.get("title") or "")
    parts = re.split(r"\s+vs\.?\s+", title, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return title, ""


def pick_market(event: dict, home: str, away: str) -> dict | None:
    markets = event.get("markets") or []
    if not markets:
        return None
    candidates = []
    for market in markets:
        q = str(market.get("question") or "")
        token_ids = market.get("clobTokenIds")
        if isinstance(token_ids, str):
            token_ids = re.findall(r"\d+", token_ids)
        if not token_ids:
            continue
        qn = norm(q)
        hn, an = norm(home), norm(away)
        is_winner = " win " in f" {qn} " or qn.startswith("will ")
        team_score = max(
            SequenceMatcher(None, qn, hn).ratio(),
            SequenceMatcher(None, qn, an).ratio(),
        )
        if is_winner or team_score > 0.35:
            candidates.append((team_score, market))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=25)
    ap.add_argument("--min-score", type=float, default=0.62)
    ap.add_argument("--max-events", type=int, default=2500)
    args = ap.parse_args()

    cfg = load_config()
    mapping_path = resolve_path(cfg["polymarket"]["mapping_file"])
    fixtures = load_cached_fixtures()
    dates = sorted(fixtures["event_date"].dropna().unique())
    start = min(dates)
    end = (pd.Timestamp(max(dates)) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    events = fetch_events(cfg["polymarket"]["gamma_base_url"], start, end, args.max_events)
    match_events = [e for e in events if is_match_event(e)]
    print(f"[match] cached fixtures={len(fixtures)} candidate events={len(match_events)}")

    existing = pd.read_csv(mapping_path) if os.path.exists(mapping_path) else pd.DataFrame()
    used_fixtures = set(existing.get("fixture_id", pd.Series(dtype=str)).astype(str))
    rows = existing.to_dict("records") if not existing.empty else []

    for event in match_events:
        event_date = str(event.get("endDate") or "")[:10]
        if not event_date:
            continue
        eh, ea = event_teams(event)
        cand = fixtures[
            (pd.to_datetime(fixtures["event_date"]) - pd.Timestamp(event_date)).abs()
            <= pd.Timedelta(days=1)
        ].copy()
        best = None
        for _, fx in cand.iterrows():
            if str(fx["fixture_id"]) in used_fixtures:
                continue
            score, reversed_order = score_pair(fx["team_home"], fx["team_away"], eh, ea)
            if best is None or score > best[0]:
                best = (score, reversed_order, fx)
        if best is None or best[0] < args.min_score:
            continue
        score, reversed_order, fx = best
        home = fx["team_away"] if reversed_order else fx["team_home"]
        away = fx["team_home"] if reversed_order else fx["team_away"]
        market = pick_market(event, home, away)
        if not market:
            continue
        token_ids = market.get("clobTokenIds")
        if isinstance(token_ids, str):
            token_ids = re.findall(r"\d+", token_ids)
        token_id = str(token_ids[0])
        rows.append({
            "fixture_id": str(fx["fixture_id"]),
            "event_date": str(fx["event_date"]),
            "team_home": fx["team_home"],
            "team_away": fx["team_away"],
            "polymarket_slug": event.get("slug"),
            "primary_market_slug": market.get("slug"),
            "primary_market_id": market.get("id"),
            "condition_id": market.get("conditionId"),
            "polymarket_market_id": token_id,
            "outcome": "Yes",
            "match_score": round(float(score), 4),
            "event_title": event.get("title"),
        })
        used_fixtures.add(str(fx["fixture_id"]))
        print(
            f"[mapped] {fx['fixture_id']} "
            f"{safe(fx['team_home'])} vs {safe(fx['team_away'])} -> "
            f"{safe(event.get('title'))} score={score:.2f}"
        )
        if len(used_fixtures) >= args.target:
            break

    out = pd.DataFrame(rows).drop_duplicates("fixture_id", keep="first")
    out.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    print(f"[done] wrote {len(out)} mapped fixtures to {mapping_path}")


if __name__ == "__main__":
    main()
