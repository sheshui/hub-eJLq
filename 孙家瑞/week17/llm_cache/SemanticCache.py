import os
import numpy as np
import redis
from typing import Optional, List, Union, Callable, Any, Dict
import faiss
try:
    from utils import _parse_redis_url, _normalize_embeddings
except ImportError:
    from utils import _parse_redis_url, _normalize_embeddings  # type: ignore


class SemanticCache:
    def __init__(
        self,
        name: str,
        embedding_method: Callable[[Union[str, List[str]]], Any],
        ttl: int = 3600 * 24,
        redis_url: str = "redis://localhost:6379",
        distance_threshold: float = 0.1,
    ):
        self.name = name
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method

        host, port, password = _parse_redis_url(redis_url)
        self.redis = redis.Redis(host=host, port=port, password=password, decode_responses=False)

        index_file = f"{self.name}.index"
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
        else:
            self.index = None

    def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]) -> Union[List, int]:
        """存储 prompt-response 对到语义缓存"""
        if isinstance(prompt, str):
            prompt = [prompt]
            response = [response]

        embedding = self.embedding_method(prompt)
        embedding = _normalize_embeddings(np.array(embedding).astype(np.float32))

        if self.index is None:
            self.index = faiss.IndexFlatIP(embedding.shape[1])

        self.index.add(embedding)  # type: ignore
        faiss.write_index(self.index, f"{self.name}.index")

        try:
            with self.redis.pipeline() as pipe:
                for q, a in zip(prompt, response):
                    pipe.setex(f"{self.name}:prompt:{q}", self.ttl, str(a))
                    pipe.lpush(f"{self.name}:list", q)
                return pipe.execute()
        except redis.RedisError:
            import traceback
            traceback.print_exc()
            return -1

    def check(self, prompt: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        检查缓存中是否有语义相似的 prompt，返回匹配的 prompt-response 对。
        返回格式: [{"prompt": ..., "response": ..., "distance": ...}, ...]
        distance 越小表示越相似（余弦距离 = 1 - 余弦相似度）
        """
        if self.index is None:
            return []

        embedding = self.embedding_method(prompt)
        embedding = _normalize_embeddings(np.array(embedding).astype(np.float32))

        dis, ind = self.index.search(embedding, k=top_k)  # type: ignore

        prompts_raw = self.redis.lrange(f"{self.name}:list", 0, -1)
        if not prompts_raw:
            return []

        results = []
        for d, idx in zip(dis[0], ind[0]):
            if idx < 0 or idx >= len(prompts_raw):
                continue
            if d > self.distance_threshold:
                continue
            q = prompts_raw[idx].decode() if isinstance(prompts_raw[idx], bytes) else prompts_raw[idx]  # type: ignore
            resp = self.redis.get(f"{self.name}:prompt:{q}")
            if resp:
                resp_data = resp.decode() if isinstance(resp, bytes) else resp
                results.append({
                    "prompt": q,
                    "response": resp_data,
                    "distance": float(d),
                })

        return results

    def call(self, prompt: str) -> Optional[List[str]]:
        """兼容旧 API：返回 response 列表"""
        results = self.check(prompt)
        if not results:
            return None
        return [r["response"] for r in results]

    def delete(self, prompt: Union[str, List[str]]) -> bool:
        """从缓存中删除指定的 prompt"""
        if isinstance(prompt, str):
            prompt = [prompt]
        try:
            with self.redis.pipeline() as pipe:
                for q in prompt:
                    pipe.delete(f"{self.name}:prompt:{q}")
                pipe.execute()
            return True
        except redis.RedisError:
            import traceback
            traceback.print_exc()
            return False

    def clear_cache(self) -> None:
        """清空所有缓存"""
        pattern = f"{self.name}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
        if os.path.exists(f"{self.name}.index"):
            os.unlink(f"{self.name}.index")
        self.index = None


if __name__ == "__main__":
    def get_embedding(text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, str):
            text = [text]
        # 模拟 768 维向量
        return np.array([np.ones(768) * (0.1 * i + 0.01) for i, _t in enumerate(text)], dtype=np.float32)

    cache = SemanticCache(
        name="semantic_cache",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="redis://localhost:6379",
        distance_threshold=0.5,
    )

    cache.clear_cache()

    cache.store(prompt="What is the capital of France?", response="Paris")
    cache.store(prompt="What is the capital of China?", response="Beijing")
    cache.store(prompt="How are you today?", response="I'm fine, thanks!")

    print("check 'What is the capital of France?':", cache.check("What is the capital of France?"))
    print("check 'What France\\'s capital?':", cache.check("What France's capital?"))
    print("call 'How are you?':", cache.call("How are you?"))
