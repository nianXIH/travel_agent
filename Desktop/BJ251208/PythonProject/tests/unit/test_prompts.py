from app.core.prompts import build_context


def test_build_context_labels_web_and_kb_sources():
    citations = [
        {
            "source_type": "web",
            "title": "2026 杭州西湖门票",
            "url": "https://example.com/xihu",
            "site": "example.com",
            "content": "西湖环湖区域全天开放。",
            "retrieval_sources": ["web"],
        },
        {
            "source_type": "kb",
            "source_file": "杭州西湖.md",
            "section_path": "西湖 > 门票",
            "city": "杭州",
            "content_type": "景点介绍",
            "content": "西湖景区部分景点需要预约。",
            "retrieval_sources": ["hybrid", "hyde"],
        },
    ]

    text = build_context(citations)

    assert "[1]" in text
    assert "联网检索" in text
    assert "2026 杭州西湖门票" in text
    assert "https://example.com/xihu" in text
    assert "[2]" in text
    assert "知识库（hybrid/hyde）" in text
    assert "杭州西湖.md" in text
