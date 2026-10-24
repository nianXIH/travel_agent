import threading
import time
from concurrent.futures import ThreadPoolExecutor

from app.core.config import Settings
from app.retrieval.reranker import BGEReranker


class SlowRerankerModel:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def compute_score(self, pairs, **kwargs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        return [0.5 for _ in pairs]


def test_score_serializes_concurrent_calls():
    reranker = BGEReranker(Settings(reranker_device="cpu"))
    model = SlowRerankerModel()
    reranker._model = model

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: reranker.score("西湖", ["西湖简介"]), range(2)))

    assert model.max_active == 1
