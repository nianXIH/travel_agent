import hashlib
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}


@dataclass(frozen=True)
class ScannedFile:
    path: Path
    relative_path: str
    size: int
    mtime: float
    sha256: str


def scan_directory(directory: Path, suffixes: set[str] | None = None) -> list[ScannedFile]:
    if not directory.exists():
        raise FileNotFoundError(f"data directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    allowed = suffixes or SUPPORTED_SUFFIXES
    files: list[ScannedFile] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        if path.suffix.lower() not in allowed:
            continue
        stat = path.stat()
        files.append(
            ScannedFile(
                path=path,
                relative_path=path.relative_to(directory).as_posix(),
                size=stat.st_size,
                mtime=stat.st_mtime,
                sha256=file_sha256(path),
            )
        )
    return files


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()
