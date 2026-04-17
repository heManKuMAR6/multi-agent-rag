# agents/reasoner.py

"""
REASONER AGENT

What it does:
- Takes retrieved docs from state
- Uses the LLM to analyze and reason through them
- Does NOT write the final answer — only produces structured reasoning
- Writes reasoning back into state

Why chain-of-thought prompting?
Explicitly telling the LLM to think step by step produces
significantly better analysis than just asking for an answer.
This is a well-documented technique in LLM research.
"""
# agents/reasoner.py

from dotenv import load_dotenv
load_dotenv()  # ← Add this at the very top

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def reasoner_agent(state: AgentState) -> AgentState:
    """
    Reasoner Agent Node.
    Receives state with question + retrieved_docs.
    Produces structured reasoning for the Synthesizer.
    """

    print(f"\n[Reasoner Agent] Analyzing {len(state['retrieved_docs'])} documents...")

    # Join all retrieved chunks into one context block
    context = "\n\n---\n\n".join(state["retrieved_docs"])

    messages = [
        SystemMessage(content="""You are an expert analyst.
Your job is NOT to write a final answer.
Your job is to reason through the provided documents step by step.
Identify key facts, connections, and gaps relevant to the question.
Be structured and thorough."""),

        HumanMessage(content=f"""Question: {state['question']}

Retrieved Documents:
{context}

Think through this step by step:
1. What do these documents tell us about the question?
2. What are the most relevant facts?
3. Are there any gaps or contradictions?
4. What conclusions can we draw?""")
    ]

    response = llm.invoke(messages)

    print(f"[Reasoner Agent] Reasoning complete")

    return {
        "reasoning": response.content,
        "current_agent": "reasoner"
    }