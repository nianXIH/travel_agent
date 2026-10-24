import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import Settings


@dataclass
class EmbeddingBatch:
    dense: list[list[float]]
    sparse: list[dict[int, float]]


class BGEM3Embedder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._encode_lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            self._configure_environment()
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self._model_source(),
                use_fp16=self._settings.embedding_fp16 and self._use_cuda(),
                device=self._device(),
            )
        return self._model

    def encode(self, texts: list[str], batch_size: int | None = None) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(dense=[], sparse=[])

        with self._encode_lock:
            output = self.model.encode(
                texts,
                batch_size=batch_size or self._settings.embedding_batch_size,
                max_length=self._settings.embedding_max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )

        dense = np.asarray(output["dense_vecs"], dtype=np.float32)
        sparse = [
            {int(token): float(weight) for token, weight in weights.items() if float(weight) > 0}
            for weights in output["lexical_weights"]
        ]
        return EmbeddingBatch(dense=dense.tolist(), sparse=sparse)

    def encode_query(self, text: str) -> tuple[list[float], dict[int, float]]:
        batch = self.encode([text], batch_size=1)
        return batch.dense[0], batch.sparse[0]

    def _model_source(self) -> str:
        candidate = Path(self._settings.embedding_model)
        if candidate.exists():
            return str(candidate)
        return self._settings.embedding_model

    def _device(self) -> str:
        if self._settings.embedding_device == "cuda":
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        return self._settings.embedding_device

    def _use_cuda(self) -> bool:
        return self._device() == "cuda"

    def _configure_environment(self) -> None:
        os.environ.setdefault("HF_ENDPOINT", self._settings.hf_endpoint)
        os.environ.setdefault("HF_HOME", str(self._settings.hf_home_path))
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
