import re
import secrets
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DEFAULT_TENANT_ID = "default"
DEFAULT_TENANT_NAME = "默认租户"
TENANTS_DIR = "tenants"
TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

Role = Literal["viewer", "editor", "admin"]
ROLE_RANK = {"viewer": 1, "editor": 2, "admin": 3}


class TenantCredential(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=64)
    tenant_name: str = Field(min_length=1, max_length=64)
    api_key: str = Field(min_length=16)
    role: Role = "viewer"

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, value: str) -> str:
        if TENANT_ID_RE.fullmatch(value) is None:
            raise ValueError("tenant_id must match ^[a-z0-9][a-z0-9_-]{0,63}$")
        return value


@dataclass(frozen=True)
class TenantMeta:
    tenant_id: str
    tenant_name: str


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str | None
    tenant_name: str
    role: str
    is_global_admin: bool = False
    requested_tenant_id: str | None = None
    open_mode: bool = False
    admin_verified: bool = False


class TenantRegistry:
    def __init__(
        self,
        credentials: list[TenantCredential] | None = None,
        admin_token: str = "",
    ) -> None:
        self._credentials = list(credentials or [])
        self._admin_token = admin_token
        self._validate_keys()

        names: dict[str, str] = {}
        for credential in self._credentials:
            existing = names.get(credential.tenant_id)
            if existing is not None and existing != credential.tenant_name:
                raise ValueError(
                    f"tenant {credential.tenant_id} has inconsistent tenant_name"
                )
            names[credential.tenant_id] = credential.tenant_name
        names.setdefault(DEFAULT_TENANT_ID, DEFAULT_TENANT_NAME)
        self._tenants = {
            tenant_id: TenantMeta(tenant_id=tenant_id, tenant_name=name)
            for tenant_id, name in names.items()
        }

    def _validate_keys(self) -> None:
        seen: set[str] = set()
        for credential in self._credentials:
            if credential.api_key in seen:
                raise ValueError("duplicate tenant api_key")
            seen.add(credential.api_key)
        if self._admin_token:
            admin_token = self._admin_token.encode("utf-8")
            for credential in self._credentials:
                if secrets.compare_digest(admin_token, credential.api_key.encode("utf-8")):
                    raise ValueError("admin_token must not match any tenant api_key")

    @property
    def open_mode(self) -> bool:
        return not self._credentials

    def verify_api_key(self, api_key: str | None) -> TenantCredential | None:
        provided = (api_key or "").encode("utf-8")
        matched: TenantCredential | None = None
        for credential in self._credentials:
            if secrets.compare_digest(provided, credential.api_key.encode("utf-8")):
                matched = credential
        return matched

    def verify_admin_token(self, token: str | None) -> bool:
        if not self._admin_token:
            return False
        provided = (token or "").encode("utf-8")
        return secrets.compare_digest(provided, self._admin_token.encode("utf-8"))

    def get_tenant(self, tenant_id: str) -> TenantMeta | None:
        return self._tenants.get(tenant_id)

    def list_tenants(self) -> list[TenantMeta]:
        return list(self._tenants.values())
