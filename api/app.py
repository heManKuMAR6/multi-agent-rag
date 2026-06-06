# api/app.py

"""
FASTAPI APPLICATION — Exposes the multi-agent RAG system as a REST API

Key concepts:
- BaseModel: Pydantic model that validates incoming request data
- @app.post: registers a POST endpoint
- @app.get: registers a GET endpoint
- The /ask endpoint runs the full LangGraph pipeline
- The /health endpoint is standard in every production service

Why Pydantic models?
They automatically validate incoming data and return clear
error messages if the request format is wrong.
No manual validation code needed.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.graph import rag_graph
from data.load_data import load_documents
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent RAG System",
    description="A LangGraph-powered multi-agent system for intelligent document Q&A",
    version="1.0.0"
)

# Load documents into ChromaDB on startup
@app.on_event("startup")
async def startup_event():
    print("[API] Loading documents into ChromaDB...")
    load_documents()
    print("[API] System ready")


# ── Request and Response Models ──

class QuestionRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is LangGraph and how does it work?"
            }
        }

class AnswerResponse(BaseModel):
    question: str
    answer: str
    reasoning: str
    num_docs_retrieved: int


# ── Endpoints ──

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Every production service has this.
    Load balancers and monitoring tools ping this to verify the service is alive.
    """
    return {"status": "healthy", "service": "multi-agent-rag"}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint — runs the full multi-agent pipeline.

    Flow:
    1. Receives question via HTTP POST
    2. Initializes LangGraph state
    3. Runs: Retriever → Reasoner → Synthesizer
    4. Returns structured response
    """

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    print(f"\n[API] Received question: {request.question}")

    # Initialize state — this is what gets passed into the graph
    initial_state = {
        "question": request.question,
        "retrieved_docs": [],
        "reasoning": "",
        "final_answer": "",
        "current_agent": "",
        "loop_count": 0,
        "next_query": "",
        "is_context_sufficient": False
    }

    # Run the full LangGraph pipeline
    result = rag_graph.invoke(initial_state)

    return AnswerResponse(
        question=request.question,
        answer=result["final_answer"],
        reasoning=result["reasoning"],
        num_docs_retrieved=len(result["retrieved_docs"])
    )


if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)