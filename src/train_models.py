"""
Train and evaluate 4 models on real_dataset_v4.parquet (falls back to v3).
Models: Logistic Regression → Random Forest → XGBoost → LightGBM
Target: high_volatility (binary: |price_move| > 3 cents in next 120s)
        OR next_60s_high_impact (causal: goal/card/penalty in next 60s) if v4
"""

import sys, json, warnings
sys.path.insert(0, '/Volumes/T7/probly/src')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    precision_score, recall_score,
)
import pickle

BASE = Path('/Volumes/T7/probly')
OUT_DIR   = BASE / 'outputs'
MODEL_DIR = OUT_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Prefer v4 dataset (richer features + causal label)
DATA_FILE = (BASE / 'outputs/real_dataset_v4.parquet'
             if (BASE / 'outputs/real_dataset_v4.parquet').exists()
             else BASE / 'outputs/real_dataset.parquet')

# v3 features (always available)
FEATURE_COLS_V3 = [
    'confidence_grade', 'conf_trend', 'xt_weight',
    'event_density_30s', 'event_density_60s', 'event_density_120s',
    'risk_event_count_30s', 'risk_event_count_60s',
    'score_diff', 'minutes_remaining', 'period_id', 'mid_price',
    'xt_x_conf', 'is_high_impact', 'is_timer_period',
    'risk_density_ratio', 'price_extremeness',
]

# v4 additional features
FEATURE_COLS_V4 = FEATURE_COLS_V3 + [
    'total_goals', 'is_leading', 'is_drawing', 'game_intensity',
    'event_density_300s', 'high_impact_60s', 'high_impact_300s',
    'xt_sum_60s', 'xt_sum_300s',
    'price_velocity', 'price_realized_vol', 'price_change_1m',
    'pressure_score',
]

TARGET = 'high_volatility'

HIGH_IMPACT = {'Score', 'PlayerGoals', 'Penalties', 'MissedPenalty',
               'RedCard', 'PlayerRedCard', 'VAR'}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'xt_x_conf' not in df.columns:
        df['xt_x_conf'] = df['xt_weight'] * df['confidence_grade']
    if 'is_high_impact' not in df.columns:
        df['is_high_impact'] = df['incident_name'].isin(HIGH_IMPACT).astype(float)
    if 'is_timer_period' not in df.columns:
        df['is_timer_period'] = df['incident_name'].isin({'Timer', 'Period'}).astype(float)
    if 'risk_density_ratio' not in df.columns:
        df['risk_density_ratio'] = df['risk_event_count_60s'] / (df['event_density_60s'] + 1)
    if 'price_extremeness' not in df.columns:
        df['price_extremeness'] = (df['mid_price'] - 0.5).abs() * 2
    return df


def load_and_split(path: Path):
    df = pd.read_parquet(path)
    is_v4 = 'price_velocity' in df.columns
    print(f"Dataset: {len(df)} rows, {df['fixture_id'].nunique()} fixtures (v{'4' if is_v4 else '3'})")
    print(f"Volatility rate: {df[TARGET].mean():.3f}")

    df = engineer_features(df)

    feat_pool = FEATURE_COLS_V4 if is_v4 else FEATURE_COLS_V3
    available_features = [c for c in feat_pool if c in df.columns]
    print(f"Features ({len(available_features)}): {available_features}")

    df = df.dropna(subset=available_features + [TARGET])

    # Chronological group split by fixture. Keep whole fixtures together to avoid
    # event-level leakage, and reserve validation for threshold selection.
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    fixture_order = (df.groupby('fixture_id')['timestamp']
                       .min()
                       .sort_values()
                       .index
                       .to_numpy())
    n = len(fixture_order)
    n_train = max(1, int(n * 0.60))
    n_val = max(1, int(n * 0.20))
    train_fixtures = set(fixture_order[:n_train])
    val_fixtures = set(fixture_order[n_train:n_train + n_val])
    test_fixtures = set(fixture_order[n_train + n_val:])

    train = df[df['fixture_id'].isin(train_fixtures)]
    val = df[df['fixture_id'].isin(val_fixtures)]
    test  = df[df['fixture_id'].isin(test_fixtures)]

    print(f"Train: {len(train)} rows ({train['fixture_id'].nunique()} fixtures)")
    print(f"Val:   {len(val)} rows ({val['fixture_id'].nunique()} fixtures)")
    print(f"Test:  {len(test)} rows ({test['fixture_id'].nunique()} fixtures)")
    print(f"Train vol rate: {train[TARGET].mean():.3f}, "
          f"Val vol rate: {val[TARGET].mean():.3f}, "
          f"Test vol rate: {test[TARGET].mean():.3f}")

    return train, val, test, available_features


def best_threshold_predict(y_prob, y_true):
    """Find threshold maximizing F1 on a validation sample."""
    thresholds = np.arange(0.1, 0.9, 0.02)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


def evaluate(name, model, X_test, y_test, scaler=None, threshold=None):
    X = scaler.transform(X_test) if scaler else X_test
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)

    if threshold is None:
        threshold = best_threshold_predict(y_prob, y_test)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        'model':     name,
        'threshold': round(float(threshold), 3),
        'f1':        round(f1_score(y_test, y_pred, zero_division=0), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc':   round(roc_auc_score(y_test, y_prob), 4),
    }
    print(f"\n{'='*50}")
    print(f"{name}  (threshold={threshold:.2f})")
    print(f"  F1={metrics['f1']:.4f}  Precision={metrics['precision']:.4f}  "
          f"Recall={metrics['recall']:.4f}  AUC={metrics['roc_auc']:.4f}")
    print(classification_report(y_test, y_pred, target_names=['stable', 'volatile']))
    return metrics, threshold


def train_lr(X_train, y_train, X_val, y_val, X_test, y_test, features):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)
    model.fit(X_tr, y_train)
    prob_val = model.predict_proba(scaler.transform(X_val))[:, 1]
    t = best_threshold_predict(prob_val, y_val)
    metrics, _ = evaluate('Logistic Regression', model, X_test, y_test, scaler, threshold=t)
    coef = dict(zip(features, model.coef_[0]))
    metrics['feature_importance'] = {k: round(float(v), 4) for k, v in
                                     sorted(coef.items(), key=lambda x: abs(x[1]), reverse=True)}
    with open(MODEL_DIR / 'lr_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'features': features, 'threshold': t}, f)
    return metrics


def train_rf(X_train, y_train, X_val, y_val, X_test, y_test, features):
    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight='balanced', n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    prob_val = model.predict_proba(X_val)[:, 1]
    t = best_threshold_predict(prob_val, y_val)
    metrics, _ = evaluate('Random Forest', model, X_test, y_test, threshold=t)
    imp = dict(zip(features, model.feature_importances_))
    metrics['feature_importance'] = {k: round(float(v), 4) for k, v in
                                     sorted(imp.items(), key=lambda x: x[1], reverse=True)}
    with open(MODEL_DIR / 'rf_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'features': features, 'threshold': t}, f)
    return metrics


def train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, features):
    try:
        from xgboost import XGBClassifier
        scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
        model = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric='logloss', n_jobs=-1, random_state=42
        )
        model.fit(X_train, y_train, verbose=False)
        prob_val = model.predict_proba(X_val)[:, 1]
        t = best_threshold_predict(prob_val, y_val)
        metrics, _ = evaluate('XGBoost', model, X_test, y_test, threshold=t)
        imp = dict(zip(features, model.feature_importances_))
        metrics['feature_importance'] = {k: round(float(v), 4) for k, v in
                                         sorted(imp.items(), key=lambda x: x[1], reverse=True)}
        with open(MODEL_DIR / 'xgb_model.pkl', 'wb') as f:
            pickle.dump({'model': model, 'features': features, 'threshold': t}, f)
        return metrics
    except ImportError:
        print("XGBoost not available, skipping")
        return None


def train_lgbm(X_train, y_train, X_val, y_val, X_test, y_test, features):
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.05,
            num_leaves=127, subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True, n_jobs=-1, random_state=42,
            verbose=-1, min_child_samples=20,
        )
        model.fit(X_train, y_train)
        prob_val = model.predict_proba(X_val)[:, 1]
        t = best_threshold_predict(prob_val, y_val)
        metrics, _ = evaluate('LightGBM', model, X_test, y_test, threshold=t)
        imp = dict(zip(features, model.feature_importances_))
        metrics['feature_importance'] = {k: round(float(v), 4) for k, v in
                                         sorted(imp.items(), key=lambda x: x[1], reverse=True)}
        with open(MODEL_DIR / 'lgbm_model.pkl', 'wb') as f:
            pickle.dump({'model': model, 'features': features, 'threshold': t}, f)
        return metrics
    except ImportError:
        print("LightGBM not available, skipping")
        return None


def main():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        print("Run build_dataset_v2.py first.")
        return

    train, val, test, features = load_and_split(DATA_FILE)

    X_train = train[features].values
    y_train = train[TARGET].values
    X_val   = val[features].values
    y_val   = val[TARGET].values
    X_test  = test[features].values
    y_test  = test[TARGET].values

    all_metrics = []

    print("\n--- Training Logistic Regression ---")
    m1 = train_lr(X_train, y_train, X_val, y_val, X_test, y_test, features)
    all_metrics.append(m1)

    print("\n--- Training Random Forest ---")
    m2 = train_rf(X_train, y_train, X_val, y_val, X_test, y_test, features)
    all_metrics.append(m2)

    print("\n--- Training XGBoost ---")
    m3 = train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, features)
    if m3: all_metrics.append(m3)

    print("\n--- Training LightGBM ---")
    m4 = train_lgbm(X_train, y_train, X_val, y_val, X_test, y_test, features)
    if m4: all_metrics.append(m4)

    # Save metrics
    metrics_out = {
        'models': all_metrics,
        'best_f1': max(m['f1'] for m in all_metrics),
        'best_model': max(all_metrics, key=lambda m: m['f1'])['model'],
        'dataset_rows': len(train) + len(test),
        'validation_rows': len(val),
        'train_vol_rate': round(float(y_train.mean()), 4),
        'val_vol_rate':   round(float(y_val.mean()), 4),
        'test_vol_rate':  round(float(y_test.mean()), 4),
        'n_features': len(features),
        'features': features,
    }

    with open(OUT_DIR / 'metrics_history.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Best model: {metrics_out['best_model']} (F1={metrics_out['best_f1']:.4f})")
    print(f"Metrics saved to outputs/metrics_history.json")

    # Save test predictions using best model
    best = max(all_metrics, key=lambda m: m['f1'])
    name_to_file = {
        'Logistic Regression': 'lr_model.pkl',
        'Random Forest':       'rf_model.pkl',
        'XGBoost':             'xgb_model.pkl',
        'LightGBM':            'lgbm_model.pkl',
    }
    model_file = MODEL_DIR / name_to_file.get(best['model'], 'rf_model.pkl')
    if model_file.exists():
        with open(model_file, 'rb') as f:
            pkg = pickle.load(f)
        mdl  = pkg['model']
        scl  = pkg.get('scaler')
        feats = pkg['features']
        t    = pkg.get('threshold', 0.5)
        X = test[feats].values
        if scl: X = scl.transform(X)
        prob = mdl.predict_proba(X)[:, 1]
        viz_cols = [
            'fixture_id', 'timestamp', 'incident_name',
            'mid_price', 'minutes_remaining', 'minutes_elapsed',
            'score_diff', 'total_goals', 'period_id',
            'xt_weight', 'confidence_grade', 'conf_trend',
            'event_density_30s', 'event_density_60s', 'event_density_120s',
            'event_density_300s', 'risk_event_count_30s', 'risk_event_count_60s',
            'high_impact_60s', 'high_impact_300s', 'xt_sum_60s', 'xt_sum_300s',
            'price_velocity', 'price_realized_vol', 'price_change_1m',
            'high_volatility', 'max_abs_move_120s', 'next_60s_high_impact',
        ]
        test_out = test[[c for c in viz_cols if c in test.columns]].copy()
        test_out['volatility_prob']    = prob
        test_out['predicted_volatile'] = (prob >= t).astype(int)
        test_out.to_parquet(OUT_DIR / 'test_predictions.parquet', index=False)
        print(f"Test predictions saved ({len(test_out)} rows).")

    return metrics_out


if __name__ == '__main__':
    main()
