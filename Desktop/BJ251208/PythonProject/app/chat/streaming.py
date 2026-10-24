import json

from sse_starlette.sse import ServerSentEvent


def sse_event(event: str, data: dict) -> ServerSentEvent:
    return ServerSentEvent(
        event=event,
        data=json.dumps(data, ensure_ascii=False, default=str),
    )
