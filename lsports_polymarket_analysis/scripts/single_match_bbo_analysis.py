"""
single_match_bbo_analysis.py - Try to retrieve historical BBO for the key match.

This script is deliberately honest about data access:
- Official Polymarket CLOB `/book` is a current/live orderbook endpoint; closed
  markets do not expose historical BBO there.
- Historical BBO requires an archive provider. This script tries PMXT Archive
  (`PMXT_API_KEY`) and DomeAPI (`DOME_API_KEY`), then writes either a BBO
  reaction table or an explicit availability report.
"""
import _bootstrap  # noqa: F401
import argparse
import os

import pandas as pd

from src import load_config, resolve_path
from src import data_loader as DL
from src import lsports_parser as P
from src import latency_analysis as LA
from src.polymarket_loader import PolymarketPriceLoader, load_mapping_table
from src import visualization as V


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=cfg["single_match"]["fixture_id"])
    ap.add_argument("--date", default=cfg["single_match"]["event_date"])
    ap.add_argument("--lookback-sec", type=int, default=120)
    ap.add_argument("--lookahead-sec", type=int, default=600)
    args = ap.parse_args()

    res = DL.download_fixture(cfg, args.date, args.fixture)
    events = P.parse_messages(DL.load_messages(res["messages"]))
    goals = P.extract_goals(events)
    if events.empty or goals.empty:
        raise SystemExit("No events/goals available for BBO analysis.")

    mapping = load_mapping_table(resolve_path(cfg["polymarket"]["mapping_file"]))
    row = mapping[mapping["fixture_id"].astype(str) == str(args.fixture)]
    if row.empty:
        raise SystemExit("No Polymarket mapping row for fixture.")
    token_id = str(row["polymarket_market_id"].iloc[0])
    market_id = str(row["primary_market_id"].iloc[0]) \
        if "primary_market_id" in row.columns and pd.notna(row["primary_market_id"].iloc[0]) \
        else None
    market_slug = row["primary_market_slug"].iloc[0] \
        if "primary_market_slug" in row.columns else ""

    start_ts = int(events["ts"].min().timestamp()) - 300
    end_ts = int(events["ts"].max().timestamp()) + 300
    loader = PolymarketPriceLoader(cfg)
    bbo = loader.load_historical_bbo(token_id, market_id, start_ts, end_ts)

    tdir = resolve_path(cfg["paths"]["table_dir"])
    rdir = resolve_path(cfg["paths"]["report_dir"])
    fdir = resolve_path(cfg["paths"]["fig_dir"])
    out_bbo = f"{tdir}/single_match_historical_bbo_{args.fixture}.csv"
    out_reaction = f"{tdir}/single_match_bbo_reaction_{args.fixture}.csv"

    status_lines = [
        "# Historical BBO Availability Report",
        "",
        f"- fixture_id: `{args.fixture}`",
        f"- Polymarket slug: `{market_slug}`",
        f"- token_id: `{token_id}`",
        f"- PMXT_API_KEY present: `{bool(os.environ.get('PMXT_API_KEY'))}`",
        f"- DOME_API_KEY present: `{bool(os.environ.get('DOME_API_KEY'))}`",
        "",
    ]

    if bbo.empty:
        pd.DataFrame(columns=[
            "goal_ts", "base_price", "reaction_ts", "reaction_latency_sec",
            "price_jump_magnitude", "lsports_leads",
        ]).to_csv(out_reaction, index=False, encoding="utf-8-sig")
        status_lines += [
            "## Result",
            "",
            "No historical BBO snapshots were retrieved in this environment.",
            "",
            "Interpretation: official CLOB can provide current orderbook, but this closed "
            "market has no current book; historical BBO needs PMXT Archive or DomeAPI "
            "credentials. The script is wired for both via `PMXT_API_KEY` / `DOME_API_KEY`.",
        ]
    else:
        bbo.to_csv(out_bbo, index=False, encoding="utf-8-sig")
        reaction = LA.goal_price_reaction(
            goals, bbo[["timestamp", "price"]].copy(),
            jump_threshold=0.03,
            lookback_sec=args.lookback_sec,
            lookahead_sec=args.lookahead_sec,
        )
        reaction.to_csv(out_reaction, index=False, encoding="utf-8-sig")
        V.plot_price_reaction(
            bbo.rename(columns={"price": "price"}), goals,
            f"Historical BBO mid vs LSports goals | fixture {args.fixture}",
            f"{fdir}/single_match_bbo_reaction.png",
        )
        lat = reaction["reaction_latency_sec"].dropna()
        status_lines += [
            "## Result",
            "",
            f"- Historical BBO snapshots: **{len(bbo)}**",
            f"- BBO reaction table: `{out_reaction}`",
            f"- Median BBO reaction lag: **{lat.median():.1f}s**" if not lat.empty
            else "- No significant BBO reaction detected in the configured window.",
        ]

    out_report = f"{rdir}/single_match_bbo_report.md"
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(status_lines) + "\n")
    print(f"BBO report: {out_report}")


if __name__ == "__main__":
    main()
