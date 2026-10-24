import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
BREADCRUMB_SEPARATOR = " > "

DEFAULT_CHUNK_SIZE = 600
DEFAULT_OVERLAP = 80
MIN_CHUNK_SIZE = 10


@dataclass(frozen=True)
class Section:
    path: tuple[str, ...]
    body: str


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    section_path: str


def parse_sections(text: str) -> list[Section]:
    stack: list[tuple[int, str]] = []
    raw_sections: list[tuple[tuple[str, ...], list[str]]] = []
    current: tuple[tuple[str, ...], list[str]] | None = None
    in_metadata = False

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            heading = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            in_metadata = any(name == "元数据" for _, name in stack)
            if in_metadata:
                current = None
            else:
                current = (tuple(name for _, name in stack), [])
                raw_sections.append(current)
            continue

        if current is not None:
            current[1].append(line)

    sections: list[Section] = []
    for path, lines in raw_sections:
        body = "\n".join(lines).strip()
        if len(body) >= MIN_CHUNK_SIZE:
            sections.append(Section(path=path, body=body))
    return sections


def chunk_sections(
    sections: list[Section],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        breadcrumb = BREADCRUMB_SEPARATOR.join(section.path)
        for piece in split_text(section.body, chunk_size, overlap):
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=f"{breadcrumb}\n{piece}",
                    section_path=breadcrumb,
                )
            )
    return chunks


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    pieces: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > chunk_size:
            pieces.extend(_hard_split(paragraph, chunk_size, overlap))
        else:
            pieces.append(paragraph)

    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        candidate = f"{buffer}\n\n{piece}" if buffer else piece
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        buffer = piece
    if buffer:
        chunks.append(buffer)

    return [_add_overlap(chunks[i - 1], chunk) if i else chunk for i, chunk in enumerate(chunks)]


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    step = max(1, chunk_size - overlap)
    return [text[start : start + chunk_size] for start in range(0, len(text), step)]


def _add_overlap(previous: str, current: str) -> str:
    if len(previous) < 40:
        return current
    return previous[-40:] + current
