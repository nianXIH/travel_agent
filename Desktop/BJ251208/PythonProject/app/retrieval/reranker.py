import os
from pathlib import Path

import numpy as np

from app.core.config import Settings


class BGEReranker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._configure_environment()
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(
                self._model_source(),
                use_fp16=self._device() == "cuda",
                devices=self._device(),
                batch_size=self._settings.reranker_batch_size,
                max_length=self._settings.reranker_max_length,
            )
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        scores = self.model.compute_score(
            pairs,
            normalize=True,
            batch_size=self._settings.reranker_batch_size,
        )
        return [float(value) for value in np.asarray(scores).reshape(-1)]

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        scores = self.score(query, [candidate["content"] for candidate in candidates])
        ranked = [
            {**candidate, "rerank_score": score}
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked[:top_n]

    def _model_source(self) -> str:
        candidate = Path(self._settings.reranker_model)
        if candidate.exists():
            return str(candidate)
        return self._settings.reranker_model

    def _device(self) -> str:
        if self._settings.reranker_device == "cuda":
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        return self._settings.reranker_device

    def _configure_environment(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", self._settings.hf_endpoint)
        os.environ.setdefault("HF_HOME", str(self._settings.hf_home_path))
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
