from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "agentic_rag"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Kaggle
    KAGGLE_DATA_DIR: str = "data/raw"

    # Neo4j
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "test"

    # Embedding model (sentence-transformers)
    EMBED_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBED_DIM: int = 768

    # LLM (transformers model id) - choose small model for CPU
    LLM_MODEL_ID: str = "ehartford/WizardLM-7B"  # replace with a small model you can run or use HF endpoint

    # RAG param
    TOP_K: int = 5

    class Config:
        env_file = ".env"

settings = Settings()
