"""
Load LSports Hyper and Trade datasets from HuggingFace.
Run: python src/load_datasets.py
"""

import os
import sys
import argparse
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
HYPER_REPO = "probly/lsports-hyper-dataset"
TRADE_REPO = "probly/lsports-trade-dataset"


def check_login():
    from huggingface_hub import whoami
    try:
        info = whoami()
        print(f"Logged in as: {info['name']}")
        return True
    except Exception as e:
        print(f"Not logged in: {e}")
        print("Run: python3 -c \"from huggingface_hub import login; login()\"")
        return False


def download_trade(force=False):
    """Download Trade dataset (~46MB, fast)."""
    from huggingface_hub import snapshot_download
    dest = DATA_DIR / "trade"
    if dest.exists() and not force:
        print(f"Trade data already at {dest}, skipping. Use --force to re-download.")
        return dest
    print(f"Downloading LSports Trade dataset → {dest}")
    snapshot_download(
        repo_id=TRADE_REPO,
        repo_type="dataset",
        local_dir=str(dest),
        ignore_patterns=["*.md", ".gitattributes"],
    )
    print(f"Trade dataset ready at {dest}")
    return dest


def download_hyper(force=False, max_files=None):
    """Download Hyper dataset (25GB, streaming-friendly).

    Use max_files to limit download size during development.
    Set max_files=None to download everything.
    """
    from huggingface_hub import list_repo_files, hf_hub_download
    dest = DATA_DIR / "hyper"
    dest.mkdir(parents=True, exist_ok=True)

    all_files = list(list_repo_files(HYPER_REPO, repo_type="dataset"))
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    print(f"Found {len(parquet_files)} parquet files in Hyper dataset")

    if max_files:
        parquet_files = parquet_files[:max_files]
        print(f"Downloading first {max_files} files (dev mode)")

    downloaded = []
    for i, fname in enumerate(parquet_files):
        out_path = dest / fname
        if out_path.exists() and not force:
            downloaded.append(out_path)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i+1}/{len(parquet_files)}] Downloading {fname}...")
        local = hf_hub_download(
            repo_id=HYPER_REPO,
            filename=fname,
            repo_type="dataset",
            local_dir=str(dest),
        )
        downloaded.append(Path(local))

    print(f"Hyper dataset: {len(downloaded)} files ready at {dest}")
    return dest


def load_trade_df():
    """Load Trade parquet files into pandas DataFrame."""
    import pandas as pd
    trade_dir = DATA_DIR / "trade"
    files = list(trade_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {trade_dir}. Run download first.")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Trade DataFrame: {df.shape}, columns: {list(df.columns)}")
    return df


def load_hyper_sample(n_files=3, sport_filter=None):
    """Load a sample of Hyper parquet files for EDA."""
    import pandas as pd
    hyper_dir = DATA_DIR / "hyper"
    files = list(hyper_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {hyper_dir}. Run download first.")
    files = files[:n_files]
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if sport_filter:
            df = df[df.get("sport_id", df.get("SportId", pd.Series())) == sport_filter]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Hyper sample DataFrame: {df.shape}, columns: {list(df.columns)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download LSports datasets from HuggingFace")
    parser.add_argument("--trade", action="store_true", help="Download Trade dataset")
    parser.add_argument("--hyper", action="store_true", help="Download Hyper dataset")
    parser.add_argument("--hyper-max-files", type=int, default=None,
                        help="Max Hyper parquet files to download (dev mode)")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    parser.add_argument("--all", action="store_true", help="Download both datasets")
    args = parser.parse_args()

    if not check_login():
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.all or args.trade:
        download_trade(force=args.force)

    if args.all or args.hyper:
        download_hyper(force=args.force, max_files=args.hyper_max_files)

    if not any([args.trade, args.hyper, args.all]):
        print("Specify --trade, --hyper, or --all")
        print("For development: python src/load_datasets.py --hyper --hyper-max-files 5 --trade")


if __name__ == "__main__":
    main()
