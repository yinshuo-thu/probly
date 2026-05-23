"""
Markov Chain state transition model for sports match progression.

Builds a transition matrix from historical LSports Hyper data and computes
win probabilities per state, which serve as Polymarket implied price estimates.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Optional

from feature_engineering import build_markov_state


def _nested_dict():
    return defaultdict(int)



class MarkovPricingModel:
    """
    Discrete Markov Chain over game states.

    State = (score_diff_bucket, period, time_bucket)
    Learns P(home_win | state) and P(s'|s) from historical data.
    """

    def __init__(self):
        self.transition_counts: Dict[Tuple, Dict[Tuple, int]] = defaultdict(_nested_dict)
        self.win_counts: Dict[Tuple, int] = defaultdict(int)
        self.state_counts: Dict[Tuple, int] = defaultdict(int)
        self.transition_matrix: Optional[Dict] = None
        self.win_prob: Optional[Dict] = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "MarkovPricingModel":
        """
        Fit model from a DataFrame of match events.

        Required columns: match_id, score_home, score_away,
                          elapsed_time, period, final_result (H/D/A)
        """
        print(f"Fitting Markov model on {df['match_id'].nunique()} matches...")

        for match_id, group in df.groupby("match_id"):
            group = group.sort_values("timestamp")
            result = group["final_result"].iloc[-1]  # H/D/A
            home_win = int(result == "H")

            states = group.apply(build_markov_state, axis=1).tolist()

            for i, state in enumerate(states):
                self.state_counts[state] += 1
                self.win_counts[state] += home_win
                if i + 1 < len(states):
                    next_state = states[i + 1]
                    self.transition_counts[state][next_state] += 1

        # Normalize
        self.win_prob = {
            s: self.win_counts[s] / self.state_counts[s]
            for s in self.state_counts if self.state_counts[s] > 0
        }
        self.transition_matrix = {
            s: {
                ns: cnt / sum(self.transition_counts[s].values())
                for ns, cnt in self.transition_counts[s].items()
            }
            for s in self.transition_counts
        }
        self._fitted = True
        print(f"Markov model fitted: {len(self.state_counts)} unique states")
        return self

    def get_win_prob(self, state: Tuple, default: float = 0.5) -> float:
        """Get P(home_win | state)."""
        return self.win_prob.get(state, default)

    def get_transition_probs(self, state: Tuple) -> Dict[Tuple, float]:
        """Get P(next_state | current_state)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return self.transition_matrix.get(state, {})

    def expected_next_price(self, state: Tuple) -> float:
        """
        Compute E[P(home_win | s')] over next-state distribution.
        High delta between this and current = expected price jump.
        """
        trans = self.get_transition_probs(state)
        if not trans:
            return self.get_win_prob(state)
        return sum(p * self.get_win_prob(ns) for ns, p in trans.items())

    def price_delta(self, state: Tuple) -> float:
        """Expected price change magnitude from current state."""
        current = self.get_win_prob(state)
        expected_next = self.expected_next_price(state)
        return abs(expected_next - current)

    def state_volatility_score(self, state: Tuple) -> float:
        """
        Volatility proxy: variance of win probabilities over next-state distribution.
        High variance = unstable state = high Polymarket volatility likelihood.
        """
        trans = self.get_transition_probs(state)
        if not trans:
            return 0.0
        probs = [self.get_win_prob(ns) for ns in trans]
        weights = list(trans.values())
        mean = sum(w * p for w, p in zip(weights, probs))
        variance = sum(w * (p - mean) ** 2 for w, p in zip(weights, probs))
        return float(variance)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "MarkovPricingModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {path}")
        return model

    def summary(self) -> dict:
        return {
            "n_states": len(self.state_counts),
            "n_transitions": sum(len(v) for v in self.transition_counts.values()),
            "avg_win_prob": np.mean(list(self.win_prob.values())) if self.win_prob else None,
            "fitted": self._fitted,
        }


def build_markov_features(df: pd.DataFrame, model: MarkovPricingModel) -> pd.DataFrame:
    """Add Markov-derived features to event DataFrame."""
    df = df.copy()
    states = df.apply(build_markov_state, axis=1)
    df["markov_win_prob"] = states.apply(lambda s: model.get_win_prob(s))
    df["markov_price_delta"] = states.apply(lambda s: model.price_delta(s))
    df["markov_volatility_score"] = states.apply(lambda s: model.state_volatility_score(s))

    # Win prob change vs previous event
    df["markov_win_prob_delta"] = df["markov_win_prob"].diff().fillna(0)
    df["markov_win_prob_accel"] = df["markov_win_prob_delta"].diff().fillna(0)

    return df
