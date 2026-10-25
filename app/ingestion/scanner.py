import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, TENANTS_DIR

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".docx"}


@dataclass(frozen=True)
class ScannedFile:
    path: Path
    relative_path: str
    size: int
    mtime: float
    sha256: str


def scan_directory(
    directory: Path,
    suffixes: set[str] | None = None,
    *,
    exclude_dirs: set[Path] | None = None,
) -> list[ScannedFile]:
    directory = directory.resolve()
    if not directory.exists():
        raise FileNotFoundError(f"data directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    allowed = suffixes or SUPPORTED_SUFFIXES
    excluded = tuple(path.resolve() for path in (exclude_dirs or set()))
    files: list[ScannedFile] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if any(path.is_relative_to(item) for item in excluded):
            continue
        if any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        if path.suffix.lower() not in allowed:
            continue
        files.append(build_scanned_file(path, directory))
    return files


def scan_tenant_directory(settings: Settings, tenant_id: str) -> list[ScannedFile]:
    """Scan one tenant's data directory.

    The default tenant owns ``DATA_DIR`` itself, which now nests every other
    tenant under ``DATA_DIR/tenants``; that subtree is never part of ``default``.
    """
    root = settings.tenant_data_path(tenant_id)
    exclude = (
        {settings.data_path / TENANTS_DIR}
        if tenant_id == DEFAULT_TENANT_ID
        else set()
    )
    return scan_directory(root, exclude_dirs=exclude)


def build_scanned_file(path: Path, root: Path) -> ScannedFile:
    stat = path.stat()
    return ScannedFile(
        path=path,
        relative_path=path.relative_to(root).as_posix(),
        size=stat.st_size,
        mtime=stat.st_mtime,
        sha256=file_sha256(path),
    )


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()
