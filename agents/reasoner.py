# agents/reasoner.py

"""
REASONER AGENT

What it does:
- Takes retrieved docs from state
- Uses the LLM to analyze and reason through them
- Evaluates if context is sufficient to answer the question
- Outputs next query if more details are needed
- Writes results back into state

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
from pydantic import BaseModel, Field
from core.state import AgentState

# Define structured output schema
class ReasonerOutput(BaseModel):
    reasoning: str = Field(
        description="Detailed, step-by-step reasoning about the question using the retrieved documents. Address what the documents say, key facts, gaps/contradictions, and conclusions."
    )
    is_context_sufficient: bool = Field(
        description="Set to True if the retrieved documents contain enough factual information to fully and directly answer the question. Set to False if crucial information is missing."
    )
    next_query: str = Field(
        description="If is_context_sufficient is False, provide a refined query to search the vector database for the missing information. If is_context_sufficient is True, set this to an empty string."
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(ReasonerOutput)


def reasoner_agent(state: AgentState) -> AgentState:
    """
    Reasoner Agent Node.
    Receives state with question + retrieved_docs.
    Produces structured reasoning, sufficiency assessment, and next query suggestion.
    """

    print(f"\n[Reasoner Agent] Analyzing {len(state.get('retrieved_docs', []))} documents...")

    # Join all retrieved chunks into one context block
    retrieved_docs = state.get("retrieved_docs", [])
    if retrieved_docs:
        context = "\n\n---\n\n".join(retrieved_docs)
    else:
        context = "No documents retrieved yet."

    messages = [
        SystemMessage(content="""You are an expert analyst.
Your job is to reason through the provided documents step by step to determine if they contain sufficient information to answer the user's question.
Be extremely rigorous: if the documents do not directly contain the answer or parts of the answer, mark context as insufficient."""),

        HumanMessage(content=f"""Question: {state['question']}

Retrieved Documents:
{context}

Think through this step by step:
1. What do these documents tell us about the question?
2. What are the most relevant facts?
3. Are there any gaps or contradictions?
4. What conclusions can we draw?

If the information is insufficient, formulate a precise search query for the missing facts.""")
    ]

    response = structured_llm.invoke(messages)

    print(f"[Reasoner Agent] Reasoning complete. Context sufficient: {response.is_context_sufficient}")
    if not response.is_context_sufficient:
        print(f"[Reasoner Agent] Refined next query: '{response.next_query}'")

    return {
        "reasoning": response.reasoning,
        "is_context_sufficient": response.is_context_sufficient,
        "next_query": response.next_query,
        "current_agent": "reasoner"
    }