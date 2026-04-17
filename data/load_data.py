# data/load_data.py

"""
DATA LOADER — One-time script to populate ChromaDB

What it does:
- Takes a list of text documents
- Converts each to a vector embedding using OpenAI
- Stores them in ChromaDB with unique IDs

Why chunk documents?
LLMs have context limits. Breaking documents into smaller
chunks means we retrieve only the RELEVANT parts, not entire
documents. This keeps the LLM focused and reduces cost.

Why unique IDs?
ChromaDB requires a unique ID per document chunk.
If you reload, it won't create duplicates.
"""

import os
from chromadb import PersistentClient
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Sample documents about AI/ML topics
# In production: load from PDFs, websites, databases
DOCUMENTS = [
    """
    LangGraph is a framework for building stateful, multi-actor applications with LLMs.
    It extends LangChain with the ability to coordinate multiple agents in a graph structure.
    Each node in the graph is an agent or function. Edges define the flow between nodes.
    LangGraph supports conditional edges, allowing dynamic routing based on state values.
    It is particularly useful for building complex agentic workflows that require loops,
    branching, and persistent memory across steps.
    """,

    """
    Retrieval Augmented Generation (RAG) is a technique that enhances LLM responses
    by retrieving relevant external documents before generating an answer.
    Instead of relying solely on the LLM's training data, RAG grounds responses
    in real, up-to-date information. The pipeline has two stages: retrieval and generation.
    Vector databases like ChromaDB store documents as embeddings for semantic search.
    RAG significantly reduces hallucination and improves factual accuracy.
    """,

    """
    Federated Learning is a machine learning approach where models are trained
    across multiple decentralized devices or servers holding local data samples.
    The key advantage is privacy — raw data never leaves the local node.
    Only model updates (gradients) are shared with a central coordinator.
    This is especially valuable in healthcare and finance where data privacy is critical.
    Federated learning enables collaborative model training without centralizing sensitive data.
    """,

    """
    LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique for LLMs.
    Instead of updating all model weights during fine-tuning, LoRA injects small trainable
    matrices into the transformer layers. This reduces trainable parameters by up to 10000x
    while maintaining model quality. LoRA is widely used for fine-tuning large models
    like LLaMA and GPT on task-specific data without requiring massive GPU resources.
    Combined with supervised fine-tuning (SFT), LoRA enables efficient model alignment.
    """,

    """
    Vector databases store data as high-dimensional vectors called embeddings.
    Unlike traditional databases that match exact values, vector databases find
    semantically similar content using distance metrics like cosine similarity.
    ChromaDB is a popular open-source vector database that runs locally or in the cloud.
    It supports metadata filtering, persistent storage, and multiple embedding models.
    Vector databases are the backbone of modern RAG systems and semantic search applications.
    """,

    """
    Agentic AI systems are AI applications where LLMs autonomously plan, reason,
    and execute multi-step tasks using tools. Unlike simple chatbots, agents can
    browse the web, write and execute code, query databases, and call external APIs.
    The ReAct pattern (Reasoning + Acting) is a popular framework where the agent
    alternates between reasoning about what to do and taking actions.
    LangChain and LangGraph are the most widely used frameworks for building agentic systems.
    """,
]


def load_documents():
    """
    Loads documents into ChromaDB.
    Safe to run multiple times — checks for existing docs first.
    """

    print("[Data Loader] Starting document ingestion...")

    # Connect to ChromaDB
    client = PersistentClient(path="./chroma_db")
    embeddings = OpenAIEmbeddings()

    # Get or create collection
    collection = client.get_or_create_collection(name="knowledge_base")

    # Check if already loaded
    if collection.count() > 0:
        print(f"[Data Loader] Collection already has {collection.count()} documents. Skipping.")
        return

    # Process each document
    for i, doc in enumerate(DOCUMENTS):
        doc_clean = doc.strip()

        # Convert text to embedding vector
        embedding = embeddings.embed_documents([doc_clean])[0]

        # Store in ChromaDB
        collection.add(
            ids=[f"doc_{i}"],
            embeddings=[embedding],
            documents=[doc_clean]
        )
        print(f"[Data Loader] Loaded document {i+1}/{len(DOCUMENTS)}")

    print(f"[Data Loader] Done. {collection.count()} documents in knowledge base.")


if __name__ == "__main__":
    load_documents()