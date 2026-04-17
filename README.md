# Multi-Agent RAG Orchestration System

A production-grade agentic AI system built with **LangGraph** and **LangChain** that orchestrates multiple specialized AI agents to answer complex questions from a document knowledge base.

## Architecture
User Question
↓
[Supervisor Graph — LangGraph]
↓
[Retriever Agent] → Semantic search via ChromaDB
↓
[Reasoner Agent]  → Chain-of-thought analysis
↓
[Synthesizer Agent] → Clean structured answer
↓
REST API Response (FastAPI)

## Tech Stack

- **LangGraph** — Multi-agent orchestration and graph state management
- **LangChain** — LLM abstractions and tool wrappers
- **ChromaDB** — Vector database for semantic document retrieval
- **OpenAI GPT-4o-mini** — LLM backbone for reasoning and synthesis
- **FastAPI** — Production REST API layer
- **Python 3.14** — Core language

## Key Features

- Multi-agent architecture with separation of concerns (retrieval, reasoning, synthesis)
- Semantic search using vector embeddings — finds meaning, not just keywords
- Persistent ChromaDB vector store — survives restarts
- Chain-of-thought reasoning before final answer synthesis
- Full REST API with auto-generated Swagger docs
- Modular design — each agent is independently swappable

## Project Structure
multi_agent_rag/
├── agents/
│   ├── retriever.py      # Semantic search agent
│   ├── reasoner.py       # Chain-of-thought analysis agent
│   └── synthesizer.py    # Final answer synthesis agent
├── core/
│   ├── state.py          # Shared AgentState (LangGraph state schema)
│   └── graph.py          # LangGraph graph definition and compilation
├── data/
│   └── load_data.py      # Document ingestion into ChromaDB
├── api/
│   └── app.py            # FastAPI REST API
├── .env                  # API keys (never committed)
├── .gitignore
└── requirements.txt

## Setup

```bash
# Clone the repo
git clone https://github.com/heManKuMAR6/multi_agent_rag.git
cd multi_agent_rag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Load documents into ChromaDB
python3 data/load_data.py

# Start the API
python3 -m uvicorn api.app:app --reload --port 8000
```

## Usage

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LangGraph and how does it work?"}'
```

Interactive API docs available at: `http://localhost:8000/docs`

## Sample Response

```json
{
  "question": "What is LangGraph and how does it work?",
  "answer": "LangGraph is a framework for building stateful multi-actor applications...",
  "reasoning": "Step-by-step analysis of retrieved documents...",
  "num_docs_retrieved": 3
}
```