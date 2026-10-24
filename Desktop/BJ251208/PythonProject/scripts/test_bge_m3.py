"""Load BGE-m3 on CUDA and run a dense/sparse encoding smoke test."""

from __future__ import annotations

import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel

MODEL_NAME = os.environ.get("BGE_M3_MODEL_PATH", "BAAI/bge-m3")
SENTENCES = [
    "三亚亚龙湾适合亲子度假，沙滩和海景很好。",
    "带孩子去亚龙湾玩沙看海很合适。",
    "西安兵马俑是世界文化遗产。",
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    print(f"HF_HOME={os.environ['HF_HOME']}")
    print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA unavailable")
        return 1

    started = time.perf_counter()
    model = BGEM3FlagModel(MODEL_NAME, use_fp16=True, device="cuda")
    print(f"model loaded in {time.perf_counter() - started:.1f}s")

    started = time.perf_counter()
    output = model.encode(
        SENTENCES,
        batch_size=3,
        max_length=256,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    encode_seconds = time.perf_counter() - started

    dense = np.asarray(output["dense_vecs"])
    print(f"dense shape={dense.shape} dtype={dense.dtype}")
    print(f"first encode time={encode_seconds:.2f}s (includes CUDA warmup)")

    started = time.perf_counter()
    model.encode(
        SENTENCES[:1],
        batch_size=1,
        max_length=256,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    print(f"warm encode time={time.perf_counter() - started:.3f}s")

    related = cosine(dense[0], dense[1])
    unrelated = cosine(dense[0], dense[2])
    print(f"cosine related={related:.4f}, unrelated={unrelated:.4f}")

    lexical = output["lexical_weights"]
    top_ids = sorted(lexical[0], key=lexical[0].get, reverse=True)[:8]
    tokens = model.tokenizer.convert_ids_to_tokens([int(i) for i in top_ids])
    top_weights = [(token, round(float(lexical[0][i]), 4)) for token, i in zip(tokens, top_ids)]
    print(f"sparse tokens={len(lexical[0])}, top={top_weights}")

    print(f"gpu allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    print(f"gpu reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GiB")

    if related <= unrelated:
        print("RESULT: FAIL (related similarity not higher)")
        return 1
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
