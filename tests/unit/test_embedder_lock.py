import threading
import time
from concurrent.futures import ThreadPoolExecutor

from app.core.config import Settings
from app.retrieval.embedder import BGEM3Embedder


class SlowModel:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def encode(self, texts, **kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        return {
            "dense_vecs": [[0.1, 0.2] for _ in texts],
            "lexical_weights": [{1: 0.5} for _ in texts],
        }


def test_encode_serializes_concurrent_calls():
    embedder = BGEM3Embedder(
        Settings(embedding_device="cpu", embedding_fp16=False),
    )
    model = SlowModel()
    embedder._model = model

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: embedder.encode(["西湖"]), range(2)))

    assert model.max_active == 1
