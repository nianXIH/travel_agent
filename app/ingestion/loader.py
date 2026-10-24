from pathlib import Path

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


class DocumentLoadError(RuntimeError):
    pass


def load_text(path: Path) -> str:
    if path.suffix.lower() not in {".md", ".markdown", ".txt"}:
        raise DocumentLoadError(f"unsupported file type: {path.suffix}")

    raw = path.read_bytes()
    for encoding in ENCODINGS:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise DocumentLoadError(f"cannot decode {path} with {ENCODINGS}")

    return text.replace("\x00", "").strip()
