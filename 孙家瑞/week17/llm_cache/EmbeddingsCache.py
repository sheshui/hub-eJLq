import numpy as np
import redis
from typing import Optional, List, Union
import hashlib
try:
    from .utils import _parse_redis_url
except ImportError:
    from utils import _parse_redis_url  # type: ignore


class EmbeddingsCache:
    def __init__(
        self,
        name: str,
        ttl: int = 3600 * 24,
        redis_url: str = "redis://localhost:6379",
    ):
        self.name = name
        self.ttl = ttl
        host, port, password = _parse_redis_url(redis_url)
        self.redis = redis.Redis(host=host, port=port, password=password, decode_responses=False)

    def _text_to_key(self, text: str) -> str:
        """将文本转为 Redis key"""
        t_code = hashlib.md5(text.encode()).hexdigest()
        return f"{self.name}:{t_code}"

    def store(self, text: Union[List[str], str], embedding: np.ndarray) -> Union[List, int]:
        """存储文本对应的 embedding 向量"""
        if isinstance(text, str):
            text = [text]
        if len(text) == 0:
            return []

        try:
            with self.redis.pipeline() as pipe:
                for i, t in enumerate(text):
                    key = self._text_to_key(t)
                    value = embedding[i].tobytes()
                    pipe.setex(key, self.ttl, value)
                return pipe.execute()
        except redis.RedisError:
            import traceback
            traceback.print_exc()
            return -1

    def call(self, text: Union[List[str], str]) -> Optional[List[np.ndarray]]:
        """根据文本获取缓存的 embedding 向量"""
        if isinstance(text, str):
            text = [text]
        if len(text) == 0:
            return []

        try:
            key_list = [self._text_to_key(t) for t in text]
            # 修复：mget 接收列表而非展开参数
            results = self.redis.mget(key_list)

            if not results or all(r is None for r in results):
                return None

            embeddings = []
            for result in results:
                if result is None:
                    embeddings.append(None)
                else:
                    embedding = np.frombuffer(result, dtype=np.float32)
                    embeddings.append(embedding)
            return embeddings
        except redis.RedisError:
            import traceback
            traceback.print_exc()
            return None

    def delete(self, text: Union[List[str], str]) -> Union[int, None]:
        """删除指定文本的缓存"""
        if isinstance(text, str):
            text = [text]

        try:
            key_list = [self._text_to_key(t) for t in text]
            return self.redis.delete(*key_list)
        except redis.RedisError:
            import traceback
            traceback.print_exc()
            return None

    def exists(self, text: Union[List[str], str]) -> List[bool]:
        """检查哪些文本有缓存"""
        if isinstance(text, str):
            text = [text]
        key_list = [self._text_to_key(t) for t in text]
        results = self.redis.mget(key_list)
        return [r is not None for r in results]

    def clear(self) -> None:
        """清空所有缓存（慎用）"""
        pattern = f"{self.name}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)


if __name__ == "__main__":
    embed_cache = EmbeddingsCache(
        name="embedding_cache",
        ttl=360,
        redis_url="redis://localhost:6379",
    )

    def get_embedding(_text: str) -> np.ndarray:
        return np.array(np.random.rand(768), dtype=np.float32)

    print("store:", embed_cache.store("hello world", get_embedding("hello world")))
    print("call:", embed_cache.call("hello world"))
    print("exists:", embed_cache.exists("hello world"))
    print("delete:", embed_cache.delete("hello world"))
    print("call after delete:", embed_cache.call("hello world"))
