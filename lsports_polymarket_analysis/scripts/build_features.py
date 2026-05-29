"""
build_features.py - 为指定比赛构建并导出特征帧 (parquet), 供建模复用。

用法:
    python scripts/build_features.py --fixture 18746260 --date 2026-05-28
"""
import _bootstrap  # noqa: F401
import argparse

from src import load_config, resolve_path
from src import data_loader as DL
from src import lsports_parser as P
from src import feature_engineering as FE


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=cfg["single_match"]["fixture_id"])
    ap.add_argument("--date", default=cfg["single_match"]["event_date"])
    args = ap.parse_args()

    res = DL.download_fixture(cfg, args.date, args.fixture)
    events = P.parse_messages(DL.load_messages(res["messages"]))
    if events.empty:
        print("事件为空"); return
    frame = FE.build_feature_frame(
        events, step_sec=cfg["analysis"]["feature_step_sec"],
        windows_sec=tuple(cfg["analysis"]["intensity_windows_sec"]),
        horizon_sec=cfg["analysis"]["hazard_horizon_sec"], fixture_id=args.fixture)
    out = resolve_path(cfg["paths"]["data_processed"]) + \
        f"/features_{args.fixture}.parquet"
    frame.to_parquet(out, index=False)
    print(f"特征帧 {frame.shape} -> {out}")
    print(f"正样本(未来{cfg['analysis']['hazard_horizon_sec']}s进球)={int(frame['label_goal_next'].sum())}")


if __name__ == "__main__":
    main()
