"""
polymarket_loader.py - Polymarket 价格数据接入接口 + 赛事映射 schema。

重要说明 (诚实声明):
    本 Hugging Face 数据集 (probly/lsports-hyper-dataset) **不包含** Polymarket
    价格 / 订单簿数据。要做 "LSports vs Polymarket 反应速度" 对比, 必须额外接入
    Polymarket 历史价格 (CLOB /prices-history 或 Gamma API)。

本模块提供:
1. PolymarketPriceLoader   - 价格加载接口 (含真实 CLOB HTTP 实现, 离线自动降级)。
2. FixtureMarketMap        - fixture(LSports) -> Polymarket market 的映射 schema。
3. 占位 price schema       - 当无法获得真实价格时, 给出统一列结构供下游 join。

设计目标: 一旦拿到 market_id / condition_id, 下游 timeline / latency 代码无需改动。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None  # 离线环境降级

# 统一的价格表列结构 (无论真实或占位都遵循此 schema)
PRICE_SCHEMA = ["market_id", "timestamp", "outcome", "price", "volume",
                "best_bid", "best_ask"]

# 赛事 -> Polymarket market 映射 schema
MAPPING_SCHEMA = ["event_id", "fixture_id", "team_home", "team_away",
                  "event_start_time", "polymarket_market_id",
                  "polymarket_slug", "outcome_mapping"]


@dataclass
class FixtureMarketMap:
    """单条 fixture -> Polymarket market 映射。

    outcome_mapping: 例如 {"home": "Yes", "away": "No"} 或多 outcome 的字典。
    """
    fixture_id: str
    team_home: str
    team_away: str
    event_start_time: str
    polymarket_market_id: Optional[str] = None     # condition_id / market id
    polymarket_slug: Optional[str] = None
    outcome_mapping: dict = field(default_factory=dict)
    event_id: Optional[str] = None

    def to_row(self) -> dict:
        d = asdict(self)
        if d.get("event_id") is None:
            d["event_id"] = self.fixture_id
        return d


def empty_price_frame() -> pd.DataFrame:
    """返回符合 PRICE_SCHEMA 的空表 (占位, 供下游 join 逻辑无缝处理缺失)。"""
    return pd.DataFrame(columns=PRICE_SCHEMA)


class PolymarketPriceLoader:
    """Polymarket 价格加载器。

    优先调用真实 CLOB /prices-history 接口; 网络不可用 / 无映射 / 出错时
    返回符合 PRICE_SCHEMA 的空表, 绝不伪造数据。
    """

    def __init__(self, cfg: dict | None = None):
        pm = (cfg or {}).get("polymarket", {})
        self.clob_base = pm.get("clob_base_url", "https://clob.polymarket.com")
        self.gamma_base = pm.get("gamma_base_url",
                                 "https://gamma-api.polymarket.com")
        self.fidelity = pm.get("prices_history_fidelity", 1)
        self.timeout = pm.get("request_timeout_sec", 15)
        self.enabled = pm.get("enabled", True)

    def load_market_prices(self, market_id: str,
                           start_ts: int | None = None,
                           end_ts: int | None = None) -> pd.DataFrame:
        """加载某个 Polymarket market 的历史价格。

        Args:
            market_id: CLOB token_id / market id (来自映射表)。
            start_ts, end_ts: Unix 秒级时间范围 (可选)。

        Returns:
            DataFrame[market_id, timestamp, outcome, price, volume,
                      best_bid, best_ask]; 失败 / 离线 / 无数据 -> 空表。
        """
        if not self.enabled or requests is None or not market_id:
            return empty_price_frame()
        params = {"market": market_id, "fidelity": self.fidelity}
        if start_ts:
            params["startTs"] = int(start_ts)
        if end_ts:
            params["endTs"] = int(end_ts)
        try:
            r = requests.get(f"{self.clob_base}/prices-history", params=params,
                             timeout=self.timeout)
            r.raise_for_status()
            hist = r.json().get("history", [])
        except Exception as e:  # noqa: BLE001
            print(f"[PolymarketPriceLoader] 拉取 {market_id} 失败/不可达: {e}")
            return empty_price_frame()
        if not hist:
            return empty_price_frame()
        df = pd.DataFrame(hist)  # 通常含 {t, p}
        out = pd.DataFrame({
            "market_id": market_id,
            "timestamp": pd.to_datetime(df["t"], unit="s", utc=True),
            "outcome": "Yes",
            "price": pd.to_numeric(df["p"], errors="coerce"),
            "volume": pd.NA,
            "best_bid": pd.NA,
            "best_ask": pd.NA,
        })
        return out[PRICE_SCHEMA]

    def search_market_by_slug(self, slug: str) -> dict:
        """通过 Gamma API 按 slug 查询 market 元数据 (含 token_ids)。"""
        if not self.enabled or requests is None or not slug:
            return {}
        try:
            r = requests.get(f"{self.gamma_base}/markets",
                             params={"slug": slug}, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return data[0] if isinstance(data, list) and data else {}
        except Exception as e:  # noqa: BLE001
            print(f"[PolymarketPriceLoader] gamma 查询 {slug} 失败: {e}")
            return {}


def load_mapping_table(path: str) -> pd.DataFrame:
    """加载 fixture->market 映射表 (parquet)。不存在则返回空 schema 表。"""
    import os
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=MAPPING_SCHEMA)
    try:
        return pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=MAPPING_SCHEMA)
