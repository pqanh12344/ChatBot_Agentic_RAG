# ChatBot_Agentic_RAG

agentic_rag_project/
├── frontend/                     # React + Vite frontend
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/
│   │   │   │   ├── Layout.tsx
│   │   │   │   ├── Header.tsx
│   │   │   │   ├── Sidebar.tsx
│   │   │   │   └── LoadingSpinner.tsx
│   │   │   ├── chat/
│   │   │   │   ├── ChatInterface.tsx
│   │   │   │   ├── MessageList.tsx
│   │   │   │   ├── MessageInput.tsx
│   │   │   │   └── MessageBubble.tsx
│   │   │   └── dashboard/
│   │   │       ├── Dashboard.tsx
│   │   │       ├── DocumentManager.tsx
│   │   │       └── Analytics.tsx
│   │   ├── services/
│   │   │   ├── api.ts
│   │   │   ├── websocket.ts
│   │   │   └── types.ts
│   │   ├── hooks/
│   │   │   ├── useChat.ts
│   │   │   ├── useWebSocket.ts
│   │   │   └── useDocuments.ts
│   │   ├── utils/
│   │   │   ├── constants.ts
│   │   │   ├── helpers.ts
│   │   │   └── validators.ts
│   │   ├── styles/
│   │   │   ├── globals.css
│   │   │   └── components.css
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── Dockerfile
├── backend/                      # Python Backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── core/                 # Core Layer
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   ├── exceptions.py
│   │   │   ├── logging.py
│   │   │   ├── security.py
│   │   │   └── database.py
│   │   ├── models/               # Domain Models Layer
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── document.py
│   │   │   ├── chat.py
│   │   │   ├── user.py
│   │   │   └── embedding.py
│   │   ├── schemas/              # Pydantic Schemas Layer
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── document.py
│   │   │   ├── chat.py
│   │   │   ├── user.py
│   │   │   └── response.py
│   │   ├── repositories/         # Data Access Layer
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── document_repository.py
│   │   │   ├── chat_repository.py
│   │   │   ├── user_repository.py
│   │   │   └── vector_repository.py
│   │   ├── services/             # Business Logic Layer
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── document_service.py
│   │   │   ├── chat_service.py
│   │   │   ├── embedding_service.py
│   │   │   ├── llm_service.py
│   │   │   ├── rag_service.py
│   │   │   └── agent_service.py
│   │   ├── api/                  # API Layer
│   │   │   ├── __init__.py
│   │   │   ├── deps.py
│   │   │   ├── middleware.py
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       ├── auth.py
│   │   │       ├── chat.py
│   │   │       ├── documents.py
│   │   │       ├── health.py
│   │   │       └── websocket.py
│   │   ├── agents/               # Agent System Layer
│   │   │   ├── __init__.py
│   │   │   ├── base_agent.py
│   │   │   ├── rag_agent.py
│   │   │   ├── document_agent.py
│   │   │   ├── search_agent.py
│   │   │   └── orchestrator.py
│   │   ├── utils/                # Utilities Layer
│   │   │   ├── __init__.py
│   │   │   ├── text_processing.py
│   │   │   ├── file_handler.py
│   │   │   ├── embeddings.py
│   │   │   ├── chunking.py
│   │   │   └── validators.py
│   │   └── workers/              # Background Workers Layer
│   │       ├── __init__.py
│   │       ├── document_processor.py
│   │       ├── embedding_worker.py
│   │       └── indexing_worker.py
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── embeddings/
│   ├── scripts/
│   │   ├── setup_database.py
│   │   ├── process_vietnamese_data.py
│   │   ├── create_embeddings.py
│   │   └── migrate_data.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── unit/
│   │   ├── integration/
│   │   └── e2e/
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── pyproject.toml
│   ├── Dockerfile
│   └── alembic/
├── docker/
│   ├── docker-compose.yml
│   ├── docker-compose.prod.yml
│   ├── nginx/
│   │   ├── nginx.conf
│   │   └── Dockerfile
│   └── postgres/
│       └── init.sql
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd.yml
│       └── test.yml
├── docs/
│   ├── API.md
│   ├── DEPLOYMENT.md
│   ├── ARCHITECTURE.md
│   └── USER_GUIDE.md
├── .env.example
├── .gitignore
├── README.md
└── Makefile

