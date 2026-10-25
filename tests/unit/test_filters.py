import pytest

from app.retrieval.filters import build_filter


def test_filter_always_starts_with_tenant():
    expression = build_filter(tenant_id="sanya", city="三亚", content_type="景点介绍")

    assert expression.startswith('tenant_id == "sanya"')
    assert 'city == "三亚"' in expression
    assert 'content_type == "景点介绍"' in expression


def test_filter_uses_tenant_even_without_other_filters():
    assert build_filter(tenant_id="default") == 'tenant_id == "default"'


def test_filter_escapes_quotes_and_backslashes():
    expression = build_filter(tenant_id='a"b\\c')

    assert expression == 'tenant_id == "a\\"b\\\\c"'


def test_filter_rejects_empty_tenant():
    with pytest.raises(ValueError, match="tenant_id"):
        build_filter(tenant_id="")


def test_filter_keeps_doc_id_clause_after_tenant():
    expression = build_filter(tenant_id="sanya", doc_ids=["doc-1", 'doc"2'])

    assert expression == (
        'tenant_id == "sanya" and doc_id in ["doc-1", "doc\\"2"]'
    )
