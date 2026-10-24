REWRITE_SYSTEM_PROMPT = """你是旅游知识库检索助手。根据对话历史，
把用户问题改写成适合检索的独立问题，并抽取过滤条件。
只输出一个 JSON 对象，不要输出解释或 Markdown。字段：
- rewritten_query: 字符串，指代消解后的完整问题；
- city: 城市名或 null。可选值：三亚、云南、厦门、张家界、成都、杭州；
- content_type: 内容类型或 null。可选值：景点介绍、线路推荐、交通指南、美食介绍、酒店信息；
- intent: 简短意图标签，如 景点推荐、住宿选择、交通方式、线路规划、美食推荐、综合问答。

只有用户明确提到城市或问题时才填写 city，不要猜测。
只有用户明确需要某一类内容时才填写 content_type：
问住宿/酒店填 酒店信息；问交通/怎么去填 交通指南；问美食/吃什么填 美食介绍；
明确问景点/门票填 景点介绍；明确问几天行程/路线填 线路推荐。
“怎么玩”“有什么推荐”“值不值得去”这类开放式问题没有明确类别时填 null。"""

QA_SYSTEM_PROMPT = """你是旅游知识库问答助手。请严格依据给定的参考资料回答，
不要编造资料中没有的信息。
要求：
1. 使用简洁、自然的中文；
2. 引用资料时在句末标注编号，如 [1]、[2]；
3. 知识库资料优先；网页资料用于补充实时信息，涉及价格、开放时间、政策等
   易变化信息时注明“据网页资料”；
4. 知识库与网页冲突时，说明差异并以知识库为主；
5. 只能引用提供的资料，不要编造链接或来源；
6. 如果资料不足以回答，明确说明知识库和联网检索中没有足够信息；
7. 不要提及“参考资料编号”以外的内部实现细节。"""

SUMMARY_SYSTEM_PROMPT = """你是会话摘要助手。请用一段不超过 200 字的中文概括对话中的目的地、
用户偏好、已给出的建议和待确认事项。只输出摘要正文。"""

HYDE_SYSTEM_PROMPT = """你是旅游知识库的检索增强助手。根据用户问题，写一段 120-200 字的
假设性旅游资料片段，用于向量检索。
要求：
1. 直接陈述事实性内容，覆盖目的地、景点、交通、门票、玩法等可能相关的信息；
2. 不要回答问题，不要使用“我认为”“可能”等不确定表述；
3. 不要编造具体价格、电话、网址等精确信息；
4. 只输出这段资料正文，不要标题、序号或解释。"""


def build_rewrite_prompt(query: str, history: list[dict], summary: str = "") -> str:
    recent = history[-6:]
    if recent:
        history_text = "\n".join(
            f"{'用户' if message['role'] == 'user' else '助手'}：{message['content'][:300]}"
            for message in recent
        )
    else:
        history_text = "无"
    summary_text = summary or "无"
    return (
        f"会话摘要：\n{summary_text}\n\n"
        f"对话历史：\n{history_text}\n\n"
        f"当前问题：{query}"
    )


def build_hyde_prompt(
    query: str,
    city: str | None = None,
    content_type: str | None = None,
) -> str:
    hints = []
    if city:
        hints.append(f"目的地：{city}")
    if content_type:
        hints.append(f"内容类型：{content_type}")
    hints.append(f"用户问题：{query}")
    return "\n".join(hints)


def build_summary_prompt(history: list[dict]) -> str:
    lines = [
        f"{'用户' if message['role'] == 'user' else '助手'}：{message['content'][:500]}"
        for message in history
        if message.get("content")
    ]
    return "对话记录：\n" + "\n".join(lines)


def build_context(citations: list[dict]) -> str:
    if not citations:
        return "无可用资料。"
    blocks = []
    for index, citation in enumerate(citations, start=1):
        if citation.get("source_type") == "web":
            title = citation.get("title") or citation.get("source_file") or "未命名网页"
            meta = " ".join(
                part
                for part in (
                    f"站点：{citation.get('site')}" if citation.get("site") else "",
                    f"链接：{citation.get('url')}" if citation.get("url") else "",
                    (
                        f"发布时间：{citation.get('published_at')}"
                        if citation.get("published_at")
                        else ""
                    ),
                )
                if part
            )
            blocks.append(
                f"[{index}] 来源类型：联网检索\n"
                f"标题：{title}\n"
                f"{meta}\n"
                f"内容：{citation.get('content', '')}"
            )
            continue

        routes = "/".join(citation.get("retrieval_sources") or ["kb"])
        location = citation.get("section_path") or citation.get("source_file", "")
        blocks.append(
            f"[{index}] 来源类型：知识库（{routes}）\n"
            f"文件：{citation.get('source_file', '')}\n"
            f"位置：{location}\n"
            f"内容：{citation.get('content', '')}"
        )
    return "\n\n".join(blocks)
