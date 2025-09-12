from app.services.llm.base_llm import BaseLLM
from app.services.retriever.base_retriever import BaseRetriever
from app.core.config import settings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.llms import HuggingFacePipeline
from typing import List

class RAGService:
    def __init__(self, llm: BaseLLM, retriever: BaseRetriever):
        self.llm = llm
        self.retriever = retriever

    def answer(self, query: str, k: int = None) -> dict:
        # Use retriever to get docs
        docs = self.retriever.get_relevant_documents(query, k=k)
        # build prompt manually or use LangChain's RetrievalQA pattern
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""Bạn là trợ lý tiếng Việt. Dưới đây là các đoạn tham khảo trích từ dữ liệu:
        
{context}

Câu hỏi: {query}

Trả lời ngắn gọn, rõ ràng, trích nguồn nếu có.
"""
        ans = self.llm.generate(prompt, max_tokens=256)
        sources = [d.metadata.get("id") for d in docs]
        return {"answer": ans, "sources": sources, "docs": [{"id":d.metadata.get("id"), "score": d.metadata.get("score")} for d in docs]}
