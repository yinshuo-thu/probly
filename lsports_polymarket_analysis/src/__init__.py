"""
lsports_polymarket_analysis - LSports 体育事件流对 Polymarket 定价价值的研究包。

模块划分:
- data_loader        : Hugging Face 数据下载 / 本地缓存 / 配置加载
- lsports_parser     : 解析 messages.parquet 事件流, 重建比分/进球/红黄牌
- polymarket_loader  : Polymarket CLOB 价格接口 + fixture->market 映射 schema
- timeline_builder   : 构建统一时间线 (LSports 事件 + 可选 Polymarket 价格)
- latency_analysis   : 事件发布时间 / 领先-滞后 / stale window 分析
- feature_engineering: rolling 事件强度 / 比分 / xT 风格特征
- modeling           : goal hazard / next-goal 概率模型
- visualization      : 所有图表
"""
import os
import yaml

__all__ = ["load_config", "project_root", "resolve_path", "get_hf_token"]


def project_root() -> str:
    """返回项目根目录 (lsports_polymarket_analysis/) 的绝对路径。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str | None = None) -> dict:
    """加载 config/config.yaml。

    Args:
        path: 可选的配置文件路径; 默认使用项目内 config/config.yaml。

    Returns:
        解析后的配置 dict。
    """
    if path is None:
        path = os.path.join(project_root(), "config", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel: str) -> str:
    """把 config 中相对项目根目录的路径解析为绝对路径, 并确保父目录存在。"""
    abs_path = os.path.join(project_root(), rel)
    os.makedirs(os.path.dirname(abs_path) if os.path.splitext(abs_path)[1] else abs_path,
                exist_ok=True)
    return abs_path


def get_hf_token(cfg: dict | None = None) -> str | None:
    """获取 Hugging Face token。

    查找顺序 (不把任何明文 token 写入版本库):
      1. 环境变量 (默认 HF_TOKEN);
      2. 本地未入库文件 config/.hf_token (已被 .gitignore 排除);
      3. 均无则返回 None (公开数据集仍可匿名访问)。
    """
    env_name = "HF_TOKEN"
    if cfg is not None:
        env_name = cfg.get("huggingface", {}).get("token_env", "HF_TOKEN")
    tok = os.environ.get(env_name)
    if tok:
        return tok
    local = os.path.join(project_root(), "config", ".hf_token")
    if os.path.exists(local):
        with open(local, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    return None
