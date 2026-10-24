import re

KNOWN_CITIES = ("三亚", "云南", "厦门", "张家界", "成都", "杭州")
KNOWN_CONTENT_TYPES = ("景点介绍", "线路推荐", "交通指南", "美食介绍", "酒店信息")

CITY_ALIASES = {
    "昆明": "云南",
    "大理": "云南",
    "丽江": "云南",
    "香格里拉": "云南",
    "西双版纳": "云南",
    "玉龙雪山": "云南",
    "洱海": "云南",
    "乌镇": "杭州",
    "西湖": "杭州",
    "灵隐": "杭州",
    "千岛湖": "杭州",
    "鼓浪屿": "厦门",
    "曾厝垵": "厦门",
    "武陵源": "张家界",
    "天门山": "张家界",
    "都江堰": "成都",
    "青城山": "成都",
    "宽窄巷子": "成都",
    "亚龙湾": "三亚",
    "海棠湾": "三亚",
    "蜈支洲": "三亚",
}

CONTENT_TYPE_HINTS = (
    ("酒店信息", ("酒店", "民宿", "住宿", "客栈", "住哪", "住")),
    ("美食介绍", ("美食", "餐厅", "小吃", "吃什么", "好吃的", "菜")),
    ("交通指南", ("交通", "机场", "高铁", "火车", "航班", "地铁", "怎么去", "怎么到")),
    ("线路推荐", ("线路", "行程", "路线", "几日游", "日游", "安排几天")),
    ("景点介绍", ("景点", "景区", "门票", "打卡", "好玩", "怎么玩", "玩什么", "游玩")),
)

DURATION_RE = re.compile(r"\d+\s*天|[两三四五六七]\s*天|几天|几日")

INTENT_BY_TYPE = {
    "酒店信息": "住宿选择",
    "美食介绍": "美食推荐",
    "交通指南": "交通方式",
    "线路推荐": "线路规划",
    "景点介绍": "景点推荐",
}


def heuristic_route(query: str) -> tuple[str | None, str | None, str]:
    city = next((name for name in KNOWN_CITIES if name in query), None)
    if city is None:
        city = next(
            (target for alias, target in CITY_ALIASES.items() if alias in query),
            None,
        )

    content_type = None
    for name, keywords in CONTENT_TYPE_HINTS:
        if any(keyword in query for keyword in keywords):
            content_type = name
            break
    if content_type is None and DURATION_RE.search(query):
        content_type = "线路推荐"

    intent = INTENT_BY_TYPE.get(content_type, "综合问答")
    return city, content_type, intent
