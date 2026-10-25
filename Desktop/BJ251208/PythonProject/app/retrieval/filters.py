def quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_filter(
    *,
    tenant_id: str,
    city: str | None = None,
    content_type: str | None = None,
    scenic_name: str | None = None,
    route_name: str | None = None,
    hotel_name: str | None = None,
    doc_ids: list[str] | None = None,
) -> str:
    if not tenant_id:
        raise ValueError("tenant_id is required")
    expressions: list[str] = [f"tenant_id == {quote(tenant_id)}"]
    for field, value in (
        ("city", city),
        ("content_type", content_type),
        ("scenic_name", scenic_name),
        ("route_name", route_name),
        ("hotel_name", hotel_name),
    ):
        if value:
            expressions.append(f"{field} == {quote(value)}")

    if doc_ids:
        joined = ", ".join(quote(doc_id) for doc_id in doc_ids)
        expressions.append(f"doc_id in [{joined}]")

    return " and ".join(expressions)
