# agents/synthesizer.py

"""
SYNTHESIZER AGENT

What it does:
- Takes the question, retrieved docs, and reasoner's analysis
- Writes a clean, structured, final answer for the user
- This is the LAST agent in the chain

Why does it receive reasoning AND docs?
The reasoning tells it what conclusions to draw.
The docs give it specific facts and quotes to cite.
Together they produce grounded, accurate answers.
"""
# agents/synthesizer.py

from dotenv import load_dotenv
load_dotenv()  # ← Add this at the very top

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
# Slightly higher temperature than Reasoner (0.3 vs 0)
# because writing benefits from a little creativity
# but we still want it grounded and accurate


def synthesizer_agent(state: AgentState) -> AgentState:
    """
    Synthesizer Agent Node.
    Receives full state including reasoning.
    Produces the final clean answer.
    """

    print(f"\n[Synthesizer Agent] Writing final answer...")

    # Give the LLM everything it needs
    context = "\n\n---\n\n".join(state["retrieved_docs"])

    messages = [
        SystemMessage(content="""You are an expert at communicating complex information clearly.
You will be given a question, relevant documents, and an analysis.
Your job is to write a clear, accurate, well-structured final answer.
- Be concise but complete
- Use bullet points where helpful
- Ground every claim in the provided documents
- Do not make anything up"""),

        HumanMessage(content=f"""Question: {state['question']}

Source Documents:
{context}

Analysis:
{state['reasoning']}

Write the final answer now:""")
    ]

    response = llm.invoke(messages)

    print(f"[Synthesizer Agent] Answer ready")

    return {
        "final_answer": response.content,
        "current_agent": "synthesizer"
    }