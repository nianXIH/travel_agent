class AppError(Exception):
    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str = "internal error",
        *,
        code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class NotFoundError(AppError):
    status_code = 404
    code = "NOT_FOUND"


class LLMNotConfiguredError(AppError):
    status_code = 503
    code = "LLM_NOT_CONFIGURED"

    def __init__(self) -> None:
        super().__init__("LLM API key is not configured")


class UpstreamError(AppError):
    status_code = 502
    code = "UPSTREAM_ERROR"
