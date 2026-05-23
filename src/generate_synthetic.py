"""
Generate synthetic LSports-like data for pipeline validation.

Creates realistic football match event streams with:
- Realistic event type distributions
- Score progression
- Confidence scores
- Synthetic Polymarket prices (derived from true win probability)

Run: python src/generate_synthetic.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "hyper"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SPORTS = [1]  # Football
EVENT_TYPES = [
    "POSSESSION", "ATTACK", "DANGEROUS_ATTACK",
    "SHOT_OFF_TARGET", "SHOT_ON_TARGET", "CORNER",
    "GOAL", "YELLOW_CARD", "RED_CARD", "PENALTY",
    "FREE_KICK", "THROW_IN", "OFFSIDE",
]
_raw = [0.30, 0.20, 0.15, 0.08, 0.06, 0.05, 0.02, 0.04, 0.005, 0.01, 0.04, 0.03, 0.015]
EVENT_PROBS = [p / sum(_raw) for p in _raw]

TEAMS_HOME = list(range(1, 21))
TEAMS_AWAY = list(range(21, 41))


def true_win_prob(score_diff: int, minutes_remaining: float) -> float:
    """Simple win probability based on score and time."""
    time_factor = np.sqrt(minutes_remaining / 90) if minutes_remaining > 0 else 0
    prob = 0.5 + 0.12 * score_diff * (1 - time_factor * 0.3)
    return float(np.clip(prob, 0.02, 0.98))


def simulate_match(match_id: int, n_events: int = 300, rng: np.random.Generator = None) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(match_id)

    home_team = rng.choice(TEAMS_HOME)
    away_team = rng.choice(TEAMS_AWAY)

    start_time = pd.Timestamp("2024-01-01") + pd.Timedelta(days=rng.integers(0, 365))
    score_home, score_away = 0, 0
    period = 1

    rows = []
    elapsed = 0.0
    for i in range(n_events):
        elapsed += rng.exponential(18)  # ~1 event per 18 seconds
        if elapsed > 5400:
            elapsed = 5400

        minute = elapsed / 60
        if minute > 90:
            period = 2
            minute = min(minute, 95)

        # Pick event
        event = rng.choice(EVENT_TYPES, p=EVENT_PROBS)

        # Update score
        if event == "GOAL":
            if rng.random() < 0.55:
                score_home += 1
            else:
                score_away += 1

        score_diff = score_home - score_away
        mins_remaining = max(0, 90 - minute)
        true_p = true_win_prob(score_diff, mins_remaining)

        # Polymarket price with lag and noise
        lag_noise = rng.normal(0, 0.015)
        poly_price = np.clip(true_p + lag_noise, 0.01, 0.99)

        # xT proxy (position based on event type)
        if event in ("GOAL", "SHOT_ON_TARGET", "SHOT_OFF_TARGET", "PENALTY"):
            x_norm = rng.uniform(80, 100)
            y_norm = rng.uniform(30, 70)
        elif event in ("DANGEROUS_ATTACK", "CORNER"):
            x_norm = rng.uniform(60, 95)
            y_norm = rng.uniform(20, 80)
        else:
            x_norm = rng.uniform(10, 80)
            y_norm = rng.uniform(10, 90)

        confidence = rng.beta(8, 2)  # high confidence events

        rows.append({
            "match_id": match_id,
            "sport_id": 1,
            "timestamp": start_time + pd.Timedelta(seconds=elapsed),
            "event_type": event,
            "elapsed_time": minute,
            "period": period,
            "score_home": score_home,
            "score_away": score_away,
            "team_id": home_team if rng.random() < 0.5 else away_team,
            "confidence": round(confidence, 4),
            "x_norm": round(x_norm, 1),
            "y_norm": round(y_norm, 1),
            "polymarket_price": round(poly_price, 4),
            "true_win_prob": round(true_p, 4),
            "final_result": None,  # filled after match end
        })

    # Set final result
    if score_home > score_away:
        result = "H"
    elif score_away > score_home:
        result = "A"
    else:
        result = "D"

    df = pd.DataFrame(rows)
    df["final_result"] = result
    return df


def generate_dataset(n_matches: int = 500, events_per_match: int = 300, seed: int = 42):
    print(f"Generating {n_matches} synthetic matches ({events_per_match} events each)...")
    rng = np.random.default_rng(seed)
    dfs = []
    for mid in range(n_matches):
        df = simulate_match(mid, n_events=events_per_match, rng=rng)
        dfs.append(df)
        if (mid + 1) % 100 == 0:
            print(f"  {mid+1}/{n_matches} matches...")

    full = pd.concat(dfs, ignore_index=True)
    out = DATA_DIR / "synthetic_matches.parquet"
    full.to_parquet(out, index=False)
    print(f"Saved {len(full):,} events to {out}")
    print(f"Columns: {list(full.columns)}")
    print(f"Goal rate: {(full['event_type']=='GOAL').mean():.4f}")
    return full


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--matches", type=int, default=500)
    p.add_argument("--events", type=int, default=300)
    args = p.parse_args()
    generate_dataset(n_matches=args.matches, events_per_match=args.events)
