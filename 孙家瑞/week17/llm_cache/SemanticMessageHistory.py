import json

import numpy as np
import redis
from typing import Optional, List, Union, Any, Dict, Callable
try:
    from utils import _parse_redis_url, _normalize_embeddings
except ImportError:
    from utils import _parse_redis_url, _normalize_embeddings  # type: ignore

try:
    import Levenshtein as _Lev
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    _Lev = None


class SemanticMessageHistory:
    def __init__(
        self,
        name: str,
        ttl: int = 3600 * 24,
        redis_url: str = "redis://localhost:6379",
        embedding_method: Optional[Callable[[Union[str, List[str]]], np.ndarray]] = None,
        distance_threshold: float = 0.5,
    ):
        self.name = name
        self.ttl = ttl
        self.embedding_method = embedding_method
        self.distance_threshold = distance_threshold

        host, port, password = _parse_redis_url(redis_url)
        self.redis = redis.Redis(host=host, port=port, password=password, decode_responses=False)

    def _history_key(self) -> str:
        return f"semantic_history:{self.name}"

    def get_history(self) -> List[Dict[str, Any]]:
        """获取完整对话历史"""
        raw = self.redis.get(self._history_key())
        if not raw:
            return []
        data = raw.decode() if isinstance(raw, bytes) else raw
        return json.loads(data)

    def add_message(self, message: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """添加一条或多条消息"""
        if isinstance(message, dict):
            message = [message]
        msgs = self.get_history()
        msgs.extend(message)
        self.redis.setex(self._history_key(), self.ttl, json.dumps(msgs, ensure_ascii=False))

    def get_recent(
        self,
        role: Optional[Union[str, List[str]]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """获取最近 top_k 条消息，可按 role 过滤"""
        msgs = self.get_history()
        if role:
            if isinstance(role, str):
                role = [role]
            selected = [m for m in msgs if m.get("role", "") in role]
        else:
            selected = msgs
        if top_k:
            selected = selected[-top_k:]
        return selected

    def get_relevant(
        self,
        content: str,
        top_k: int = 10,
        use_vector: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        获取与 content 语义相关的消息。
        - use_vector=True: 使用向量相似度搜索（需提供 embedding_method）
        - use_vector=False: 使用 Levenshtein 字符串相似度（需安装 python-Levenshtein）
        """
        msgs = self.get_history()
        if not msgs:
            return []

        if use_vector and self.embedding_method:
            return self._search_by_vector(content, msgs, top_k)
        elif HAS_LEVENSHTEIN and _Lev is not None:
            return self._search_by_levenshtein(content, msgs, top_k)
        else:
            selected = [m for m in msgs if content.lower() in m.get("content", "").lower()]
            if top_k:
                selected = selected[-top_k:]
            return selected

    def _search_by_vector(
        self,
        query: str,
        msgs: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """基于向量相似度搜索"""
        if not self.embedding_method:
            return []

        indexed = [(i, m) for i, m in enumerate(msgs) if m.get("content")]
        if not indexed:
            return []

        texts = [m["content"] for _, m in indexed]
        embeddings = self.embedding_method(texts)
        embeddings = _normalize_embeddings(np.array(embeddings).astype(np.float32))

        query_emb = self.embedding_method(query)
        query_emb = _normalize_embeddings(np.array([query_emb]).astype(np.float32))

        sims = np.dot(query_emb, embeddings.T)[0]
        top_indices = np.argsort(sims)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if sims[idx] >= self.distance_threshold:
                results.append(indexed[idx][1])
        return results

    @staticmethod
    def _search_by_levenshtein(
        content: str,
        msgs: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """基于 Levenshtein 相似度搜索"""
        scored = []
        for m in msgs:
            m_content = m.get("content", "")
            if not m_content:
                continue
            ratio = _Lev.ratio(m_content, content)  # type: ignore
            scored.append((ratio, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """语义搜索的便捷别名"""
        return self.get_relevant(query, top_k=top_k, use_vector=self.embedding_method is not None)

    def delete_history(self, top_k: int = 10) -> None:
        """删除最近 top_k 条消息"""
        msgs = self.get_history()
        msgs = msgs[:-top_k] if top_k else []
        self.redis.setex(self._history_key(), self.ttl, json.dumps(msgs, ensure_ascii=False))

    def clear_history(self) -> int:
        """清空所有历史"""
        return self.redis.delete(self._history_key())


if __name__ == "__main__":
    history = SemanticMessageHistory(
        name="my-session",
        redis_url="redis://localhost:6379",
    )
    history.clear_history()
    history.add_message([
        {"role": "user", "content": "hello, how are you?"},
        {"role": "llm", "content": "I'm doing fine, thanks."},
        {"role": "user", "content": "what is the weather going to be today?"},
        {"role": "llm", "content": "I don't know", "metadata": {"model": "gpt-4"}},
        {"role": "user", "content": "what is the weather going to be today?"},
    ])

    print("get_history:", history.get_history())
    print("get_recent top_k=1:", history.get_recent(top_k=1))
    print("get_recent role=user:", history.get_recent(role="user", top_k=1))
    print("get_relevant 'today':", history.get_relevant("today", top_k=1, use_vector=False))
    print("get_relevant 'thanks':", history.get_relevant("thanks", top_k=1, use_vector=False))
