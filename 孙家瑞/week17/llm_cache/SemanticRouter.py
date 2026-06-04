import os
import json
import numpy as np
import redis
import faiss
import hashlib
from typing import Optional, List, Dict, Callable
try:
    from utils import _parse_redis_url, _normalize_embeddings
except ImportError:
    from utils import _parse_redis_url, _normalize_embeddings  # type: ignore


class SemanticRouter:
    def __init__(
        self,
        name: str,
        embedding_method: Callable[[List[str]], np.ndarray],
        routes: Optional[List[Dict]] = None,
        redis_url: str = "redis://localhost:6379",
        distance_threshold: float = 0.3,
        ttl: int = 3600 * 24,
    ):
        self.name = name
        self.embedding_method = embedding_method
        self.distance_threshold = distance_threshold
        self.ttl = ttl

        host, port, password = _parse_redis_url(redis_url)
        self.redis = redis.Redis(host=host, port=port, password=password, decode_responses=False)

        self._index_file = f"{self.name}.index"
        self.index: Optional[faiss.IndexFlatIP] = None
        self._route_refs: List[str] = []
        # target -> {"start_idx": int, "count": int, "metadata": dict}
        self._target_infos: Dict[str, Dict] = {}

        if os.path.exists(self._index_file):
            self.index = faiss.read_index(self._index_file)
            refs_data = self.redis.get(f"{self.name}:route_refs")
            if refs_data:
                raw = refs_data.decode() if isinstance(refs_data, bytes) else refs_data
                self._route_refs = json.loads(raw)
            infos_data = self.redis.get(f"{self.name}:target_infos")
            if infos_data:
                raw = infos_data.decode() if isinstance(infos_data, bytes) else infos_data
                self._target_infos = json.loads(raw)
        else:
            self.index = None

        if routes:
            for route in routes:
                self.add_route(
                    questions=route["questions"],
                    target=route["target"],
                    metadata=route.get("metadata"),
                )

    def add_route(
        self,
        questions: List[str],
        target: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """添加一条路由"""
        if not questions:
            return

        embeddings = self.embedding_method(questions)
        embeddings = _normalize_embeddings(np.array(embeddings).astype(np.float32))

        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])

        start_idx = len(self._route_refs)
        self.index.add(embeddings)  # type: ignore
        faiss.write_index(self.index, self._index_file)

        self._route_refs.extend(questions)

        self._target_infos[target] = {
            "start_idx": start_idx,
            "count": len(questions),
            "metadata": metadata or {},
        }

        self.redis.set(f"{self.name}:route_refs", json.dumps(self._route_refs))
        self.redis.set(f"{self.name}:target_infos", json.dumps(self._target_infos))

    def route(self, question: str) -> Optional[Dict]:
        """根据 question 返回匹配的路由信息"""
        if self.index is None:
            return None

        cache_key = f"{self.name}:cache:{hashlib.md5(question.encode()).hexdigest()}"
        cached = self.redis.get(cache_key)
        if cached:
            raw = cached.decode() if isinstance(cached, bytes) else cached
            return json.loads(raw)

        embedding = self.embedding_method([question])
        embedding = _normalize_embeddings(np.array(embedding).astype(np.float32))

        dis, ind = self.index.search(embedding, k=1)  # type: ignore

        d = float(dis[0][0])
        idx = int(ind[0][0])

        if idx < 0 or d > self.distance_threshold:
            return None

        target = self._find_target_by_index(idx)
        if target is None:
            return None

        metadata: Dict = self._target_infos.get(target, {}).get("metadata", {})

        result: Dict = {
            "target": target,
            "metadata": metadata,
            "distance": d,
        }

        self.redis.setex(cache_key, self.ttl, json.dumps(result))

        return result

    def _find_target_by_index(self, idx: int) -> Optional[str]:
        """通过索引找到对应的 target"""
        for target, info in self._target_infos.items():
            start: int = info["start_idx"]  # type: ignore
            count: int = info["count"]  # type: ignore
            if start <= idx < start + count:
                return target
        return None

    def delete_route(self, target: str) -> bool:
        """删除指定 target 的路由"""
        if target not in self._target_infos:
            return False
        del self._target_infos[target]
        self.redis.hdel(f"{self.name}:routes", target)
        self.redis.set(f"{self.name}:target_infos", json.dumps(self._target_infos))
        return True

    def clear_cache(self) -> None:
        """清空路由缓存和索引"""
        pattern = f"{self.name}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
        if os.path.exists(self._index_file):
            os.unlink(self._index_file)
        self.index = None
        self._route_refs = []
        self._target_infos = {}


if __name__ == "__main__":
    def get_embedding(texts: List[str]) -> np.ndarray:
        vectors = []
        for i, _t in enumerate(texts):
            vec = np.zeros(768, dtype=np.float32)
            vec[i % 768] = 1.0
            vectors.append(vec)
        return np.array(vectors)

    router = SemanticRouter(
        name="topic-router",
        embedding_method=get_embedding,
        redis_url="redis://localhost:6379",
        distance_threshold=0.5,
    )

    router.clear_cache()

    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon", "Hello there"],
        target="greeting",
        metadata={"type": "greeting"},
    )
    router.add_route(
        questions=["如何退货", "退货流程是什么", "我想退货"],
        target="refund",
        metadata={"type": "commerce"},
    )

    print("route 'Hi, good morning':", router.route("Hi, good morning"))
    print("route 'Hi, good morning' (cached):", router.route("Hi, good morning"))
    print("route '我想退货':", router.route("我想退货"))
