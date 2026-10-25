import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

TITLE_RE = re.compile(r"^#\s+(.+?)\s*$")
METADATA_HEADING_RE = re.compile(r"^##\s*元数据\s*$")
METADATA_LINE_RE = re.compile(r"^-\s*([^：:]+?)\s*[：:]\s*(.+?)\s*$")

CATEGORY_CONTENT_TYPES = {
    "景点攻略": "景点介绍",
    "线路推荐": "线路推荐",
    "交通指南": "交通指南",
    "美食推荐": "美食介绍",
    "酒店信息": "酒店信息",
}

FILENAME_SUFFIXES = {
    "景点攻略": ("景点推荐",),
    "线路推荐": ("线路推荐",),
    "交通指南": ("交通指南",),
    "美食推荐": ("美食推荐",),
    "酒店信息": ("住宿推荐",),
}


@dataclass
class DocumentMeta:
    title: str
    content_type: str
    city: str
    scenic_name: str = ""
    route_name: str = ""
    hotel_name: str = ""
    restaurant_name: str = ""
    extra: dict = field(default_factory=dict)


def parse_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        match = TITLE_RE.match(line)
        if match:
            return match.group(1)
    return fallback


def parse_metadata_block(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    in_block = False
    for line in text.splitlines():
        if METADATA_HEADING_RE.match(line):
            in_block = True
            continue
        if in_block and line.startswith("#"):
            break
        if not in_block:
            continue
        match = METADATA_LINE_RE.match(line.strip())
        if match:
            metadata[match.group(1).strip()] = match.group(2).strip()
    return metadata


def parse_document_meta(text: str, path: Path, relative_path: str) -> DocumentMeta:
    metadata = parse_metadata_block(text)
    category = path.parent.name if path.parent.name in CATEGORY_CONTENT_TYPES else ""
    fallback_title = path.stem

    content_type = (
        metadata.get("内容类型")
        or CATEGORY_CONTENT_TYPES.get(category, "")
        or "其他"
    )
    city = _normalize_city(metadata.get("城市", ""))
    if not city:
        city = _city_from_filename(path, category) or _city_from_destination(
            metadata.get("目的地", "")
        )

    title = parse_title(text, fallback_title)
    scenic_name = metadata.get("景点名称", "")
    route_name = metadata.get("线路名称", "")
    hotel_name = metadata.get("住宿名称", "")
    restaurant_name = metadata.get("餐厅名称", "")

    extra = {
        key: value
        for key, value in metadata.items()
        if key not in {"内容类型", "城市", "景点名称", "线路名称", "住宿名称", "餐厅名称"}
    }
    extra["source_category"] = category or path.parent.name

    return DocumentMeta(
        title=title,
        content_type=content_type,
        city=city,
        scenic_name=scenic_name,
        route_name=route_name,
        hotel_name=hotel_name,
        restaurant_name=restaurant_name,
        extra=extra,
    )


def make_doc_id(tenant_id: str, relative_path: str) -> str:
    digest = hashlib.sha1(f"{tenant_id}:{relative_path}".encode()).hexdigest()
    return f"doc_{digest[:16]}"


def _normalize_city(city: str) -> str:
    city = city.strip()
    if city.endswith("多城市"):
        return city[: -len("多城市")]
    return city


def _city_from_destination(destination: str) -> str:
    for separator in ("、", "，", ",", " ", "/"):
        destination = destination.replace(separator, "|")
    first = destination.split("|")[0].strip()
    return first


def _city_from_filename(path: Path, category: str) -> str:
    stem = path.stem
    for suffix in FILENAME_SUFFIXES.get(category, ()):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return ""
