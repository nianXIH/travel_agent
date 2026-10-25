from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.tenancy import (
    DEFAULT_TENANT_ID,
    ROLE_RANK,
    TenantCredential,
    TenantRegistry,
)
from app.ingestion.metadata import make_doc_id


def _credential(**overrides) -> TenantCredential:
    values = {
        "tenant_id": "sanya",
        "tenant_name": "三亚运营",
        "api_key": "k" * 32,
        "role": "viewer",
    }
    values.update(overrides)
    return TenantCredential(**values)


def test_tenant_id_must_be_slug():
    for bad in ("Sanya", "三亚", "-sanya", "a" * 65):
        with pytest.raises(ValidationError):
            _credential(tenant_id=bad)


def test_api_key_min_length():
    with pytest.raises(ValidationError):
        _credential(api_key="short")


def test_registry_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate"):
        TenantRegistry([_credential(), _credential(tenant_id="xiamen", tenant_name="厦门")])


def test_registry_rejects_inconsistent_names():
    with pytest.raises(ValueError, match="tenant_name"):
        TenantRegistry(
            [
                _credential(role="viewer"),
                _credential(
                    tenant_id="sanya",
                    tenant_name="另一个名字",
                    api_key="z" * 32,
                    role="editor",
                ),
            ]
        )


def test_registry_rejects_admin_token_matching_tenant_key():
    with pytest.raises(ValueError, match="admin_token"):
        TenantRegistry([_credential()], admin_token="k" * 32)


def test_registry_always_contains_default_and_accepts_configured_name():
    registry = TenantRegistry(
        [_credential(tenant_id=DEFAULT_TENANT_ID, tenant_name="总部", role="editor")]
    )

    assert registry.open_mode is False
    assert registry.get_tenant(DEFAULT_TENANT_ID).tenant_name == "总部"
    assert {meta.tenant_id for meta in registry.list_tenants()} == {"default"}


def test_registry_uses_last_matching_key_without_early_return():
    first = _credential(tenant_id="a", tenant_name="A")
    second = _credential(tenant_id="b", tenant_name="B", api_key="z" * 32)
    registry = TenantRegistry([first, second])

    assert registry.verify_api_key("z" * 32).tenant_id == "b"
    assert registry.verify_api_key("missing") is None


def test_open_mode_and_admin_token():
    registry = TenantRegistry([], "admin-secret")

    assert registry.open_mode is True
    assert registry.verify_admin_token("admin-secret") is True
    assert registry.verify_admin_token("wrong") is False
    assert TenantRegistry().verify_admin_token("") is False


def test_role_rank_is_ordered():
    assert ROLE_RANK["viewer"] < ROLE_RANK["editor"] < ROLE_RANK["admin"]


def test_tenant_paths_are_scoped_and_reject_traversal(tmp_path: Path):
    settings = Settings(_env_file=None, data_dir=tmp_path)

    assert settings.tenant_data_path(DEFAULT_TENANT_ID) == settings.data_path
    assert settings.tenant_data_path("sanya") == (tmp_path / "tenants" / "sanya").resolve()
    assert settings.tenant_upload_path("sanya") == (
        tmp_path / "tenants" / "sanya" / "uploads"
    ).resolve()
    with pytest.raises(ValueError):
        settings.tenant_data_path("../escape")


def test_tenant_paths_tolerate_unnormalized_absolute_data_dir(tmp_path: Path):
    (tmp_path / "data").mkdir()
    settings = Settings(_env_file=None, data_dir=tmp_path / "data" / ".." / "data")

    root = settings.tenant_data_path(DEFAULT_TENANT_ID)
    tenant = settings.tenant_data_path("sanya")

    expected_root = (tmp_path / "data").resolve()
    assert root == expected_root
    assert root.is_absolute()
    assert ".." not in root.parts
    assert tenant == (expected_root / "tenants" / "sanya").resolve()
    assert tenant.is_relative_to(root)
    assert settings.tenant_upload_path("sanya") == (
        expected_root / "tenants" / "sanya" / "uploads"
    ).resolve()
    with pytest.raises(ValueError):
        settings.tenant_data_path("../escape")


def test_doc_id_is_scoped_by_tenant():
    assert make_doc_id("default", "景点攻略/杭州.md") != make_doc_id(
        "sanya", "景点攻略/杭州.md"
    )
