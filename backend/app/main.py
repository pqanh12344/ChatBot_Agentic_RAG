from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import chat as chat_router
from app.core.config import settings

app = FastAPI(title=settings.APP_NAME)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(chat_router.router, prefix="/api")

@app.get("/")
def root():
    return {"msg": "agentic_rag backend"}
