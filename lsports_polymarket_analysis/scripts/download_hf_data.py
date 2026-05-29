"""
download_hf_data.py - 下载 docs 与 football 采样数据到 data/raw。

用法:
    python scripts/download_hf_data.py            # 下载 docs + 配置日期的采样比赛
    python scripts/download_hf_data.py --dates 2026-05-28
    python scripts/download_hf_data.py --max 20   # 每日最多下载 20 场

token: 优先环境变量 HF_TOKEN, 否则回退到 src 中的临时开发 token。
"""
import _bootstrap  # noqa: F401
import argparse

import pandas as pd

from src import load_config, resolve_path
from src import data_loader as DL


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=cfg["dates"])
    ap.add_argument("--max", type=int, default=cfg["batch"]["max_fixtures_per_date"])
    ap.add_argument("--docs", action="store_true", help="同时下载 docs 数据字典")
    args = ap.parse_args()

    if args.docs:
        for doc in ["docs/data_dictionary/README.md",
                    "docs/MODEL_SPEC_markov_pricing.md",
                    "docs/REQUIREMENTS_sports_pricing.md"]:
            p = DL.download_doc(cfg, doc)
            print(f"docs: {doc} -> {p}")

    index = []
    for d in args.dates:
        fixtures = DL.list_fixtures(cfg, d)
        if fixtures.empty:
            print(f"[{d}] 无 fixture")
            continue
        # 按 fixture_id 排序后持续尝试, 直到成功下载 N 场; 某些目录可能缺 messages.parquet。
        fixtures = fixtures.sort_values("fixture_id")
        print(f"[{d}] 计划成功下载 {args.max} 场 (最多尝试 {len(fixtures)} 个 fixture)")
        ok_count = 0
        for _, row in fixtures.iterrows():
            res = DL.download_fixture(cfg, d, row["fixture_id"])
            ok = res["messages"] is not None
            index.append({"event_date": d, "fixture_id": row["fixture_id"],
                          "downloaded": ok})
            print(f"  fixture {row['fixture_id']}: {'OK' if ok else 'FAIL'}")
            if ok:
                ok_count += 1
            if ok_count >= args.max:
                break

    idx = pd.DataFrame(index)
    out = resolve_path(cfg["paths"]["data_processed"]) + "/download_index.parquet"
    idx.to_parquet(out, index=False)
    print(f"下载索引已保存: {out}  (成功 {idx['downloaded'].sum()}/{len(idx)})")


if __name__ == "__main__":
    main()
