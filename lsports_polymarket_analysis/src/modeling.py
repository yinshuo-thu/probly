"""
modeling.py - goal hazard / next-goal 概率建模。

提供两类方法:
A. 非参数强度基线 intensity_hazard(): 用滚动 xT 强度做单调映射得到风险代理,
   无需训练, 用于单场比赛快速展示 "进球前是否有可观测信号"。
B. 监督式 logistic 回归 fit_logistic_hazard(): 用时间步特征预测 "未来 horizon
   秒内是否进球", 适合跨多场比赛 pooled 训练。

定价用途: 模型输出的 next-goal 概率可直接转化为对 Polymarket 隐含概率的提前
修正信号 —— 当模型 hazard 升高而市场价格尚未反应时, 即为潜在 stale / edge。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_engineering import feature_matrix


def intensity_hazard(frame: pd.DataFrame, col: str = "xt_sum_300s") -> pd.DataFrame:
    """非参数强度风险代理: 把滚动 xT 强度经 logistic 压缩到 [0,1]。

    这是无需训练的基线, 用于直观展示进球前的事件强度积累。
    """
    if frame is None or frame.empty or col not in frame.columns:
        return pd.DataFrame(columns=["ts", "hazard"])
    x = frame[col].fillna(0.0).to_numpy(dtype=float)
    if x.std() < 1e-9:
        h = np.zeros_like(x)
    else:
        z = (x - x.mean()) / (x.std() + 1e-9)
        h = 1.0 / (1.0 + np.exp(-z))
    return pd.DataFrame({"ts": frame["ts"].to_numpy(), "hazard": h})


def fit_logistic_hazard(frame: pd.DataFrame, test_frac: float = 0.0) -> dict:
    """训练 logistic 回归预测 next-goal hazard。

    Args:
        frame: 一场或多场比赛拼接的特征帧 (含 label_goal_next)。
        test_frac: >0 时按时间顺序留出末尾比例做评估 (单场样本少时设 0)。

    Returns:
        dict: {model, scaler, cols, auc, n_pos, n, pred (与 frame 等长的概率)}。
        正样本不足时 model=None, 退回 intensity 基线概率。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    X, y, cols = feature_matrix(frame)
    out = {"model": None, "scaler": None, "cols": cols, "auc": np.nan,
           "n_pos": int(y.sum()) if y is not None else 0,
           "n": int(len(frame)), "pred": None}
    if y is None or y.sum() < 3 or (1 - y).sum() < 3:
        # 正/负样本太少, 无法稳健训练 -> 用强度基线代替
        base = intensity_hazard(frame)
        out["pred"] = base["hazard"].to_numpy() if not base.empty else np.zeros(len(frame))
        out["note"] = "insufficient positives; fell back to intensity baseline"
        return out

    scaler = StandardScaler()
    if test_frac and test_frac > 0:
        n_test = max(1, int(len(frame) * test_frac))
        Xtr, ytr = X[:-n_test], y[:-n_test]
        Xte, yte = X[-n_test:], y[-n_test:]
    else:
        Xtr, ytr, Xte, yte = X, y, X, y
    Xtr_s = scaler.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr_s, ytr)
    pred_all = clf.predict_proba(scaler.transform(X))[:, 1]
    try:
        if len(np.unique(yte)) > 1:
            out["auc"] = float(roc_auc_score(yte, clf.predict_proba(
                scaler.transform(Xte))[:, 1]))
    except Exception:  # noqa: BLE001
        pass
    out.update({"model": clf, "scaler": scaler, "pred": pred_all})
    return out


def advance_warning(frame: pd.DataFrame, pred: np.ndarray, goals: pd.DataFrame,
                    threshold: float = 0.5, horizon_sec: int = 120) -> pd.DataFrame:
    """评估模型提前预警能力: 每个进球前模型 hazard 首次越过阈值的提前秒数。

    Returns:
        DataFrame[goal_ts, first_alert_ts, advance_warning_sec, fired]。
    """
    if frame is None or frame.empty or goals is None or goals.empty or pred is None:
        return pd.DataFrame(columns=["goal_ts", "first_alert_ts",
                                     "advance_warning_sec", "fired"])
    ts = pd.to_datetime(frame["ts"]).reset_index(drop=True)
    p = pd.Series(pred).reset_index(drop=True)
    rows = []
    for _, g in goals.iterrows():
        gts = g["ts"]
        win = (ts > gts - pd.Timedelta(seconds=horizon_sec)) & (ts <= gts)
        alerts = ts[win & (p >= threshold)]
        if len(alerts) > 0:
            first = alerts.iloc[0]
            rows.append({"goal_ts": gts, "first_alert_ts": first,
                         "advance_warning_sec": (gts - first).total_seconds(),
                         "fired": True})
        else:
            rows.append({"goal_ts": gts, "first_alert_ts": pd.NaT,
                         "advance_warning_sec": np.nan, "fired": False})
    return pd.DataFrame(rows)
