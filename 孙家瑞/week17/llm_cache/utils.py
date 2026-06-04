"""llm_cache 共享工具模块"""

from urllib.parse import urlparse
import numpy as np


def _parse_redis_url(redis_url: str) -> tuple:
    """解析 redis:// URL，返回 (host, port, password)"""
    if redis_url.startswith("redis://"):
        parsed = urlparse(redis_url)
        return parsed.hostname or "localhost", parsed.port or 6379, parsed.password
    return redis_url, 6379, None


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 归一化向量，用于余弦相似度搜索"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms
