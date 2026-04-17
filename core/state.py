# core/state.py

"""
WHAT IS THIS?
This is the shared memory object that flows through every agent in our graph.
Every agent reads from it and writes their result back into it.

WHY TypedDict?
LangGraph requires state to be a TypedDict so it knows exactly
what fields exist and can validate them at runtime.

WHY operator.add on retrieved_docs?
If two agents write to the same field, operator.add MERGES their
results instead of one overwriting the other.
"""

from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    # The original question from the user - set once, never changed
    question: str

    # Documents fetched from ChromaDB by the Retriever agent
    # operator.add means results accumulate across agents
    retrieved_docs: Annotated[list, operator.add]

    # The Reasoner agent's analysis of the retrieved docs
    reasoning: str

    # The final clean answer written by the Synthesizer agent
    final_answer: str

    # Tracks which agent just ran — useful for debugging
    current_agent: str