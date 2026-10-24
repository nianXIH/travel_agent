from langgraph.graph import END, START, StateGraph

from app.chat.nodes import ChatNodes
from app.chat.state import ChatState


def build_chat_graph(nodes: ChatNodes):
    graph = StateGraph(ChatState)
    graph.add_node("load_history", nodes.load_history)
    graph.add_node("rewrite_and_route", nodes.rewrite_and_route)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate", nodes.generate)
    graph.add_node("persist", nodes.persist)

    graph.add_edge(START, "load_history")
    graph.add_edge("load_history", "rewrite_and_route")
    graph.add_edge("rewrite_and_route", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "persist")
    graph.add_edge("persist", END)
    return graph.compile()
