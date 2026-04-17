# agents/retriever.py

"""
RETRIEVER AGENT

What it does:
- Takes the user's question from state
- Searches ChromaDB for the top 3 most relevant document chunks
- Writes results back into state under 'retrieved_docs'

Why top 3?
Enough context without overwhelming the Reasoner agent with noise.
In production you'd make this configurable.
"""

# agents/retriever.py

from dotenv import load_dotenv
load_dotenv()  # ← This must be FIRST before any OpenAI imports

from chromadb import PersistentClient
from langchain_openai import OpenAIEmbeddings
from core.state import AgentState

chroma_client = PersistentClient(path="./chroma_db")
embeddings = OpenAIEmbeddings()


def retriever_agent(state: AgentState) -> AgentState:
    """
    Retriever Agent Node.
    LangGraph calls this function and passes the current state.
    We return only the fields we're updating.
    """

    print(f"\n[Retriever Agent] Searching for: {state['question']}")

    # Get or create the collection (like a table in a database)
    collection = chroma_client.get_or_create_collection(name="knowledge_base")

    # Convert the question to a vector embedding for semantic search
    query_embedding = embeddings.embed_query(state["question"])

    # Search ChromaDB for top 3 most semantically similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, collection.count()) if collection.count() > 0 else 1
    )

    # Extract the actual text from results
    docs = results["documents"][0] if results["documents"] else ["No relevant documents found."]

    print(f"[Retriever Agent] Found {len(docs)} relevant chunks")

    # Return only what this agent updates
    return {
        "retrieved_docs": docs,
        "current_agent": "retriever"
    }