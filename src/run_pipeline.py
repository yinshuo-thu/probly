"""
End-to-end pipeline: load data → engineer features → train models → save outputs.

Usage:
  python src/run_pipeline.py --sample         # Run on small sample first
  python src/run_pipeline.py --full           # Run on full dataset
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from load_datasets import load_hyper_sample, load_trade_df
from feature_engineering import extract_match_features
from markov_model import MarkovPricingModel, build_markov_features
from volatility_predictor import VolatilityPredictor

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def synthetic_polymarket_prices(df: pd.DataFrame, markov_model: MarkovPricingModel) -> pd.DataFrame:
    """
    In absence of real Polymarket price data, simulate prices from Markov win probabilities.
    Adds Gaussian noise and lagged response to simulate market inefficiency.
    """
    states = df.apply(
        lambda r: markov_model.get_win_prob(
            (r.get("markov_score_bucket", 2), r.get("markov_period", 1), r.get("markov_time_bucket", 0))
        ), axis=1
    )
    noise = np.random.normal(0, 0.02, len(df))  # market inefficiency
    lag = states.shift(3).fillna(0.5)            # market lags ~3 events
    price = 0.7 * lag + 0.3 * states + noise
    df["polymarket_price"] = price.clip(0.01, 0.99)
    return df


def prepare_dataset(n_files: int = 5, sport_id: int = None) -> pd.DataFrame:
    """Load, clean, and engineer features for the dataset."""
    print("Loading Hyper sample...")
    df = load_hyper_sample(n_files=n_files, sport_filter=sport_id)

    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Required columns mapping
    col_map = {
        "matchid": "match_id", "sportid": "sport_id",
        "scorehome": "score_home", "scoreaway": "score_away",
        "elapsedtime": "elapsed_time", "eventtype": "event_type",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Parse timestamp
    for tc in ["timestamp", "createdat", "updatedat", "time"]:
        if tc in df.columns:
            df["timestamp"] = pd.to_datetime(df[tc], errors="coerce")
            break
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp("2024-01-01")

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["match_id", "timestamp"] if "match_id" in df.columns else ["timestamp"])

    print(f"Raw data: {df.shape}")
    print(f"Columns: {list(df.columns)[:20]}")

    # Engineer features per match (or globally if match_id unavailable)
    if "match_id" in df.columns:
        print("Extracting features per match...")
        chunks = []
        for mid, grp in df.groupby("match_id"):
            if len(grp) < 10:
                continue
            try:
                grp_feat = extract_match_features(grp)
                chunks.append(grp_feat)
            except Exception as e:
                pass
        df = pd.concat(chunks, ignore_index=True) if chunks else df
    else:
        df = extract_match_features(df)

    print(f"After feature engineering: {df.shape}")
    return df


def split_train_test(df: pd.DataFrame, test_frac: float = 0.2) -> tuple:
    """Time-based train/test split."""
    df = df.sort_values("timestamp")
    n = len(df)
    split = int(n * (1 - test_frac))
    return df.iloc[:split], df.iloc[split:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Use small sample (5 files)")
    parser.add_argument("--full", action="store_true", help="Use full dataset")
    parser.add_argument("--n-files", type=int, default=5, help="Number of parquet files to load")
    parser.add_argument("--sport-id", type=int, default=None, help="Filter to specific sport")
    args = parser.parse_args()

    n_files = 5 if args.sample else (None if args.full else args.n_files)

    # 1. Load and prepare data
    df = prepare_dataset(n_files=n_files, sport_id=args.sport_id)

    # 2. Fit Markov model (on train portion)
    df_train, df_test = split_train_test(df)

    markov = MarkovPricingModel()
    if "match_id" in df_train.columns and "final_result" in df_train.columns:
        markov.fit(df_train)
    else:
        print("No match_id/final_result columns — using synthetic Markov model")
        # Synthetic: assign win prob by score diff
        markov._fitted = True
        markov.win_prob = {(i, p, t): 0.5 + (i - 2) * 0.12 for i in range(7) for p in range(1, 4) for t in range(6)}
        markov.transition_matrix = {}
        markov.state_counts = {}

    markov.save(OUTPUTS_DIR / "markov_model.pkl")

    # 3. Add Markov features
    df_train = build_markov_features(df_train, markov)
    df_test = build_markov_features(df_test, markov)

    # 4. Add synthetic Polymarket prices if not available
    if "polymarket_price" not in df_train.columns:
        print("Generating synthetic Polymarket prices from Markov model...")
        df_train = synthetic_polymarket_prices(df_train, markov)
        df_test = synthetic_polymarket_prices(df_test, markov)

    # 5. Label volatility
    from feature_engineering import label_volatility
    df_train = label_volatility(df_train, lookahead_sec=60, threshold_pct=0.05)
    df_test = label_volatility(df_test, lookahead_sec=60, threshold_pct=0.05)

    print(f"Train volatility rate: {df_train['high_volatility_label'].mean():.3f}")
    print(f"Test  volatility rate: {df_test['high_volatility_label'].mean():.3f}")

    # 6. Save engineered datasets
    df_train.to_parquet(OUTPUTS_DIR / "train_features.parquet", index=False)
    df_test.to_parquet(OUTPUTS_DIR / "test_features.parquet", index=False)

    # 7. Train all models iteratively
    predictor = VolatilityPredictor()
    metrics = predictor.run_all(df_train, df_test)

    # 8. Save predictions for visualization
    df_test = df_test.copy()
    df_test["predicted_volatility_prob"] = predictor.predict(df_test)
    df_test.to_parquet(OUTPUTS_DIR / "test_predictions.parquet", index=False)

    # 9. Summary
    summary = {
        "n_train": len(df_train),
        "n_test": len(df_test),
        "best_model": predictor.best_name,
        "best_f1": predictor.best_f1,
        "models": [{k: v for k, v in m.items() if k not in ("report", "confusion_matrix")} for m in metrics],
    }
    with open(OUTPUTS_DIR / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPipeline complete. Best: {predictor.best_name} F1={predictor.best_f1:.4f}")
    print(f"Outputs in: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
