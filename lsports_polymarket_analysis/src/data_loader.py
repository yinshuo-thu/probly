"""
data_loader.py - Hugging Face LSports 数据集的下载与本地缓存。

数据集结构 (probly/lsports-hyper-dataset):
    sport_id=6046__sport=Football/
        event_date=YYYY-MM-DD/
            fixture_id=<id>/
                fixtures.parquet   # 单场比赛元数据 (1 行)
                manifest.json      # 下载批次/统计信息
                messages.parquet   # 事件流 (核心), 每场数千~数万行

注意: 该数据集是 **按小时批量归档** 的快照, 不是实时流捕获。
"""
from __future__ import annotations

import json
import os
import glob
from typing import Optional

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from . import get_hf_token, resolve_path


def _api(cfg: dict) -> HfApi:
    return HfApi(token=get_hf_token(cfg))


def list_event_dates(cfg: dict) -> list[str]:
    """列出 football 分区下所有可用的 event_date (升序)。"""
    api = _api(cfg)
    repo = cfg["huggingface"]["repo_id"]
    prefix = cfg["huggingface"]["football_prefix"]
    items = api.list_repo_tree(repo, path_in_repo=prefix, repo_type="dataset",
                               recursive=False)
    dates = []
    for it in items:
        name = it.path.rsplit("/", 1)[-1]
        if name.startswith("event_date="):
            dates.append(name.split("=", 1)[1])
    return sorted(dates)


def list_fixtures(cfg: dict, event_date: str) -> pd.DataFrame:
    """列出某个 event_date 下的所有 fixture, 含 messages.parquet 大小。

    Returns:
        DataFrame[fixture_id, event_date, dir_path, messages_bytes]
    """
    token = get_hf_token(cfg)
    local = _list_cached_fixtures(cfg, event_date)
    if token is None:
        if not local.empty:
            print(f"[list_fixtures] 未配置 HF token, 使用本地缓存: {len(local)} 场")
            return local
        return pd.DataFrame(columns=["fixture_id", "event_date", "dir_path",
                                     "messages_bytes"])

    api = HfApi(token=token)
    repo = cfg["huggingface"]["repo_id"]
    prefix = cfg["huggingface"]["football_prefix"]
    base = f"{prefix}/event_date={event_date}"
    rows = []
    try:
        fixture_dirs = list(api.list_repo_tree(
            repo, path_in_repo=base, repo_type="dataset", recursive=False
        ))
    except Exception as e:  # noqa: BLE001
        print(f"[list_fixtures] 无法列出远程 {base}: {e}")
        if not local.empty:
            print(f"[list_fixtures] 回退到本地缓存: {len(local)} 场")
            return local
        return pd.DataFrame(columns=["fixture_id", "event_date", "dir_path",
                                     "messages_bytes"])
    for fd in fixture_dirs:
        dname = fd.path.rsplit("/", 1)[-1]
        if not dname.startswith("fixture_id="):
            continue
        fid = dname.split("=", 1)[1]
        rows.append({"fixture_id": fid, "event_date": event_date,
                     "dir_path": fd.path, "messages_bytes": None})
    return pd.DataFrame(rows)


def _list_cached_fixtures(cfg: dict, event_date: str) -> pd.DataFrame:
    """远程不可用时, 从 data/raw 镜像目录列出已缓存 fixture。"""
    raw_dir = resolve_path(cfg["paths"]["data_raw"])
    prefix = cfg["huggingface"]["football_prefix"]
    pattern = os.path.join(raw_dir, prefix, f"event_date={event_date}",
                           "fixture_id=*")
    rows = []
    for path in glob.glob(pattern):
        if not os.path.isdir(path):
            continue
        fid = os.path.basename(path).split("=", 1)[-1]
        msg = _find_cached_file(path, "messages.parquet")
        rows.append({
            "fixture_id": fid,
            "event_date": event_date,
            "dir_path": os.path.relpath(path, raw_dir).replace(os.sep, "/"),
            "messages_bytes": os.path.getsize(msg) if os.path.exists(msg) else None,
        })
    return pd.DataFrame(rows)


def _find_cached_file(fixture_dir: str, fname: str) -> str:
    """返回 fixture 目录中已存在的文件, 兼容早期 hf_hub_download 嵌套路径。"""
    direct = os.path.join(fixture_dir, fname)
    if os.path.exists(direct):
        return direct
    hits = glob.glob(os.path.join(fixture_dir, "**", fname), recursive=True)
    hits = [p for p in hits if ".cache" + os.sep not in p]
    return hits[0] if hits else direct


def download_fixture(cfg: dict, event_date: str, fixture_id: str | int,
                     force: bool = False) -> dict:
    """下载单场比赛的 fixtures.parquet / manifest.json / messages.parquet。

    文件缓存到 data/raw/ 下镜像目录结构。失败时返回的 dict 中对应键为 None。

    Returns:
        {"fixture_id", "event_date", "fixtures", "manifest", "messages"}
        其中 fixtures/manifest/messages 为本地文件绝对路径或 None。
    """
    repo = cfg["huggingface"]["repo_id"]
    prefix = cfg["huggingface"]["football_prefix"]
    base = f"{prefix}/event_date={event_date}/fixture_id={fixture_id}"
    raw_dir = resolve_path(cfg["paths"]["data_raw"])
    token = get_hf_token(cfg)

    out = {"fixture_id": str(fixture_id), "event_date": event_date,
           "fixtures": None, "manifest": None, "messages": None}
    targets = {"fixtures": "fixtures.parquet", "manifest": "manifest.json",
               "messages": "messages.parquet"}
    for key, fname in targets.items():
        fixture_dir = os.path.join(raw_dir, base.replace("/", os.sep))
        local = _find_cached_file(fixture_dir, fname)
        if os.path.exists(local) and not force:
            out[key] = local
            continue
        try:
            p = hf_hub_download(repo, f"{base}/{fname}", repo_type="dataset",
                                token=token,
                                local_dir=fixture_dir)
            # hf_hub_download(local_dir=...) 会把文件放在 local_dir/<filename>
            out[key] = p if os.path.exists(p) else local
        except Exception as e:  # noqa: BLE001
            print(f"[download_fixture] {base}/{fname} 下载失败: {e}")
            out[key] = None
    return out


def load_messages(path: str) -> pd.DataFrame:
    """读取一场比赛的 messages.parquet, 容错处理空文件/坏文件。"""
    if path is None or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        print(f"[load_messages] 读取失败 {path}: {e}")
        return pd.DataFrame()


def load_manifest(path: str) -> dict:
    """读取 manifest.json, 失败返回空 dict。"""
    if path is None or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[load_manifest] 读取失败 {path}: {e}")
        return {}


def load_fixture_meta(path: str) -> dict:
    """读取 fixtures.parquet 的单行元数据为 dict。"""
    if path is None or not os.path.exists(path):
        return {}
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return {}
        return df.iloc[0].to_dict()
    except Exception as e:  # noqa: BLE001
        print(f"[load_fixture_meta] 读取失败 {path}: {e}")
        return {}


def download_doc(cfg: dict, rel_path: str) -> Optional[str]:
    """下载 docs/ 下的某个文档文件, 返回本地路径。"""
    repo = cfg["huggingface"]["repo_id"]
    token = get_hf_token(cfg)
    try:
        return hf_hub_download(repo, rel_path, repo_type="dataset", token=token)
    except Exception as e:  # noqa: BLE001
        print(f"[download_doc] {rel_path} 下载失败: {e}")
        return None
