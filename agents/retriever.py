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
    # Determine which query to search with
    query = state.get("next_query")
    if not query:
        query = state["question"]

    print(f"\n[Retriever Agent] Searching for: '{query}'")

    # Get or create the collection (like a table in a database)
    collection = chroma_client.get_or_create_collection(name="knowledge_base")

    # Convert the query to a vector embedding for semantic search
    query_embedding = embeddings.embed_query(query)

    # Search ChromaDB for top 3 most semantically similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, collection.count()) if collection.count() > 0 else 1
    )

    # Extract the actual text from results
    docs = results["documents"][0] if results["documents"] else []

    # Filter out already retrieved documents to avoid duplicate context
    existing_docs = state.get("retrieved_docs", [])
    new_docs = [doc for doc in docs if doc not in existing_docs]

    print(f"[Retriever Agent] Found {len(docs)} chunks. Adding {len(new_docs)} new unique chunks.")

    current_loop = state.get("loop_count", 0)

    # Return only what this agent updates
    return {
        "retrieved_docs": new_docs,
        "current_agent": "retriever",
        "loop_count": current_loop + 1
    }