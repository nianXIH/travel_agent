"""Verify installed packages and CUDA availability."""

from __future__ import annotations

import importlib
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

REQUIRED_MODULES = [
    "fastapi",
    "uvicorn",
    "sse_starlette",
    "pydantic",
    "pydantic_settings",
    "orjson",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "tiktoken",
    "tenacity",
    "pymilvus",
    "pymongo",
    "FlagEmbedding",
    "transformers",
    "sentence_transformers",
    "sentencepiece",
    "pypdf",
    "docx",
    "loguru",
    "httpx",
]

DISTRIBUTIONS = [
    "torch",
    "fastapi",
    "uvicorn",
    "sse-starlette",
    "pydantic",
    "pydantic-settings",
    "langchain-core",
    "langchain-openai",
    "langgraph",
    "pymilvus",
    "pymongo",
    "FlagEmbedding",
    "transformers",
    "sentence-transformers",
    "sentencepiece",
    "loguru",
]


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"Interpreter: {sys.executable}")

    failed = False
    print("\n[packages]")
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:
            failed = True
            print(f"  FAIL {module}: {exc}")
        else:
            print(f"  OK   {module}")

    print("\n[versions]")
    for dist in DISTRIBUTIONS:
        try:
            print(f"  {dist}=={version(dist)}")
        except PackageNotFoundError:
            print(f"  {dist}: not installed")

    print("\n[cuda]")
    try:
        import torch
    except Exception as exc:
        print(f"  FAIL import torch: {exc}")
        return 1

    print(f"  torch: {torch.__version__}")
    print(f"  torch.version.cuda: {torch.version.cuda}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  FAIL: CUDA is not available")
        return 1

    print(f"  cudnn: {torch.backends.cudnn.version()}")
    print(f"  device count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(
            f"  device[{index}]: {props.name}, "
            f"capability={props.major}.{props.minor}, "
            f"memory={props.total_memory / 1024**3:.2f} GiB"
        )

    device = torch.device("cuda:0")
    a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    b = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    print(f"  fp16 matmul: {tuple(c.shape)}, mean={c.float().mean().item():.4f}")
    print(f"  bf16 supported: {torch.cuda.is_bf16_supported()}")

    if failed:
        print("\nRESULT: FAIL (missing modules)")
        return 1
    print("\nRESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
