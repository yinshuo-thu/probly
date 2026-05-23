"""
Polymarket volatility prediction pipeline.

Trains and evaluates ML models to predict high-volatility windows.
Iterates: Logistic Regression → Random Forest → XGBoost → LSTM
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    # Rolling event features
    "event_density_30s", "event_density_60s", "event_density_120s",
    "xt_sum_30s", "xt_sum_60s", "xt_sum_120s",
    "conf_mean_30s", "conf_mean_60s", "conf_mean_120s",
    "risk_event_count_30s", "risk_event_count_60s", "risk_event_count_120s",
    # Markov features
    "markov_win_prob", "markov_price_delta", "markov_volatility_score",
    "markov_win_prob_delta", "markov_win_prob_accel",
    "markov_score_bucket", "markov_time_bucket",
    # Game state
    "score_diff", "total_goals", "minutes_remaining", "game_intensity",
    "xt_value", "event_weight", "confidence_val",
]

TARGET_COL = "high_volatility_label"


def get_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and labels from engineered DataFrame."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        print(f"Warning: missing features {missing}, filling with 0")

    X = df.reindex(columns=FEATURE_COLS, fill_value=0).values
    y = df[TARGET_COL].values if TARGET_COL in df.columns else np.zeros(len(df))
    return X.astype(np.float32), y.astype(int)


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "model": name,
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5,
        "avg_precision": average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0,
        "support_positive": int(y_test.sum()),
        "support_total": len(y_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": report,
    }
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  AUC={metrics['roc_auc']:.4f}")
    print(f"Positive class: {metrics['support_positive']}/{metrics['support_total']}")
    print("="*50)
    return metrics


class VolatilityPredictor:
    """
    Iterative volatility prediction pipeline.

    Trains multiple model families and tracks improvement.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or OUTPUTS_DIR / "models"
        self.model_dir.mkdir(exist_ok=True)
        self.history: list = []
        self.best_model = None
        self.best_f1 = 0.0
        self.best_name = None

    def train_logistic(self, X_train, y_train, X_test, y_test) -> dict:
        """v0.1 — Logistic Regression baseline."""
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000, C=1.0, random_state=42
            ))
        ])
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, "v0.1_LogisticRegression")
        self._update_best(model, metrics)
        self._save_model(model, "v01_logistic.pkl")
        return metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test) -> dict:
        """v0.2 — Random Forest."""
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=12, class_weight="balanced",
                min_samples_leaf=5, n_jobs=-1, random_state=42
            ))
        ])
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, "v0.2_RandomForest")
        self._update_best(model, metrics)
        self._save_model(model, "v02_random_forest.pkl")
        return metrics

    def train_xgboost(self, X_train, y_train, X_test, y_test) -> dict:
        """v0.3 — XGBoost."""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            print("xgboost not installed, skipping")
            return {}

        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr", random_state=42,
            use_label_encoder=False, verbosity=0,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
        metrics = evaluate(model, X_test, y_test, "v0.3_XGBoost")
        self._update_best(model, metrics)
        self._save_model(model, "v03_xgboost.pkl")

        # Feature importance
        fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        fi.to_csv(self.model_dir / "feature_importance_xgb.csv")
        print("\nTop-10 features:")
        print(fi.head(10).to_string())

        return metrics

    def train_lightgbm(self, X_train, y_train, X_test, y_test) -> dict:
        """v0.4 — LightGBM (fast, often best on tabular)."""
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            print("lightgbm not installed, skipping")
            return {}

        scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = LGBMClassifier(
            n_estimators=1000, max_depth=8, learning_rate=0.03,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale, random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[])
        metrics = evaluate(model, X_test, y_test, "v0.4_LightGBM")
        self._update_best(model, metrics)
        self._save_model(model, "v04_lightgbm.pkl")
        return metrics

    def run_all(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> list:
        """Run full model iteration sequence."""
        X_train, y_train = get_feature_matrix(df_train)
        X_test, y_test = get_feature_matrix(df_test)

        print(f"Train: {X_train.shape}, pos={y_train.sum()}/{len(y_train)}")
        print(f"Test:  {X_test.shape},  pos={y_test.sum()}/{len(y_test)}")

        all_metrics = []

        m1 = self.train_logistic(X_train, y_train, X_test, y_test)
        all_metrics.append(m1)

        m2 = self.train_random_forest(X_train, y_train, X_test, y_test)
        all_metrics.append(m2)

        m3 = self.train_xgboost(X_train, y_train, X_test, y_test)
        if m3:
            all_metrics.append(m3)

        m4 = self.train_lightgbm(X_train, y_train, X_test, y_test)
        if m4:
            all_metrics.append(m4)

        self.history = all_metrics
        self._save_metrics()

        print(f"\n{'='*50}")
        print(f"BEST MODEL: {self.best_name}  F1={self.best_f1:.4f}")
        print("="*50)

        return all_metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict volatility probability for new events."""
        X, _ = get_feature_matrix(df)
        if hasattr(self.best_model, "predict_proba"):
            return self.best_model.predict_proba(X)[:, 1]
        return self.best_model.predict(X).astype(float)

    def _update_best(self, model, metrics: dict):
        if metrics.get("f1", 0) > self.best_f1:
            self.best_f1 = metrics["f1"]
            self.best_model = model
            self.best_name = metrics["model"]

    def _save_model(self, model, fname: str):
        with open(self.model_dir / fname, "wb") as f:
            pickle.dump(model, f)

    def _save_metrics(self):
        safe = []
        for m in self.history:
            s = {k: v for k, v in m.items() if k != "report"}
            safe.append(s)
        with open(OUTPUTS_DIR / "metrics_history.json", "w") as f:
            json.dump(safe, f, indent=2)
        print(f"Metrics saved to {OUTPUTS_DIR / 'metrics_history.json'}")
