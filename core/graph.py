# core/graph.py

"""
THE GRAPH — This is what makes it a LangGraph application.

Key concepts:
- StateGraph: a graph where every node reads/writes shared state
- add_node: registers an agent function as a named node
- add_edge: connects nodes in sequence
- set_entry_point: where the graph starts
- set_finish_point: where the graph ends and returns

Why compile()?
compile() validates the graph structure, checks for disconnected
nodes, missing edges, and returns a runnable object.
You call .invoke() on the compiled graph to run it.
"""

from langgraph.graph import StateGraph, END
from core.state import AgentState
from agents.retriever import retriever_agent
from agents.reasoner import reasoner_agent
from agents.synthesizer import synthesizer_agent


def build_graph():
    """
    Builds and compiles the multi-agent RAG graph.
    Returns a compiled runnable graph.
    """

    # Initialize the graph with our state schema
    graph = StateGraph(AgentState)

    # ── Register each agent as a node ──
    # First argument: the name we'll use to reference this node
    # Second argument: the function to call when this node runs
    graph.add_node("retriever", retriever_agent)
    graph.add_node("reasoner", reasoner_agent)
    graph.add_node("synthesizer", synthesizer_agent)

    # ── Define the flow with edges ──
    # This is linear for now — but you could add conditional
    # edges here to branch based on state values
    graph.add_edge("retriever", "reasoner")
    graph.add_edge("reasoner", "synthesizer")
    graph.add_edge("synthesizer", END)

    # ── Set where the graph starts ──
    graph.set_entry_point("retriever")

    # ── Compile and validate the graph ──
    compiled = graph.compile()

    print("[Graph] Multi-agent RAG graph compiled successfully")
    return compiled


# Single shared instance — import this everywhere
rag_graph = build_graph()