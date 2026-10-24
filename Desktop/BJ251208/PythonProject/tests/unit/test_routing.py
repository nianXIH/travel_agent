from app.chat.routing import heuristic_route


def test_heuristic_route_recognizes_lodging_intent() -> None:
    assert heuristic_route("成都住春熙路还是宽窄巷子方便")[:2] == ("成都", "酒店信息")
    assert heuristic_route("张家界住市区还是武陵源")[:2] == ("张家界", "酒店信息")


def test_heuristic_route_recognizes_sightseeing_intent() -> None:
    assert heuristic_route("厦门鼓浪屿和环岛路怎么玩")[:2] == ("厦门", "景点介绍")
    assert heuristic_route("张家界国家森林公园怎么玩")[:2] == ("张家界", "景点介绍")


def test_heuristic_route_keeps_transport_priority() -> None:
    assert heuristic_route("张家界机场到武陵源景区怎么走")[:2] == ("张家界", "交通指南")


def test_heuristic_route_maps_city_alias_and_duration() -> None:
    assert heuristic_route("云南七日游线路怎么安排")[:2] == ("云南", "线路推荐")
    assert heuristic_route("大理住哪里方便")[:2] == ("云南", "酒店信息")
