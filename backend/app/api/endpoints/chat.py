from fastapi import APIRouter
from pydantic import BaseModel
from app.services.rag_service import RAGService
from app.services.llm.transformers_llm import TransformersLLM
from app.db.neo4j_client import Neo4jClient
from app.services.retriever.neo4j_retriever import Neo4jRetriever
from app.core.config import settings

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    sources: list

# instantiate components (in production use DI container)
neo4j = Neo4jClient()
retriever = Neo4jRetriever(neo4j)
llm = TransformersLLM(settings.LLM_MODEL_ID)
rag = RAGService(llm, retriever)

@router.post("/chat")
def chat(req: ChatRequest):
    out = rag.answer(req.query, k=req.top_k)
    return {"answer": out["answer"], "sources": out["sources"], "docs": out["docs"]}
