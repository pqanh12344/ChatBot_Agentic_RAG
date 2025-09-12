# ChatBot_Agentic_RAG

agentic_rag_project/ <br>
├── frontend/ <br>                     # React + Vite frontend
│   ├── public/ <br>
│   ├── src/ <br>
│   │   ├── components/ <br>
│   │   │   ├── common/ <br>
│   │   │   │   ├── Layout.tsx <br>
│   │   │   │   ├── Header.tsx <br>
│   │   │   │   ├── Sidebar.tsx <br>
│   │   │   │   └── LoadingSpinner.tsx <br>
│   │   │   ├── chat/ <br>
│   │   │   │   ├── ChatInterface.tsx <br>
│   │   │   │   ├── MessageList.tsx <br>
│   │   │   │   ├── MessageInput.tsx <br>
│   │   │   │   └── MessageBubble.tsx <br>
│   │   │   └── dashboard/ <br>
│   │   │       ├── Dashboard.tsx <br>
│   │   │       ├── DocumentManager.tsx
│   │   │       └── Analytics.tsx <br>
│   │   ├── services/ <br>
│   │   │   ├── api.ts <br>
│   │   │   ├── websocket.ts <br>
│   │   │   └── types.ts <br>
│   │   ├── hooks/ <br>
│   │   │   ├── useChat.ts <br>
│   │   │   ├── useWebSocket.ts <br>
│   │   │   └── useDocuments.ts <br>
│   │   ├── utils/ <br>
│   │   │   ├── constants.ts <br>
│   │   │   ├── helpers.ts <br>
│   │   │   └── validators.ts <br>
│   │   ├── styles/ <br>
│   │   │   ├── globals.css <br>
│   │   │   └── components.css <br>
│   │   ├── App.tsx <br>
│   │   └── main.tsx <br>
│   ├── package.json <br>
│   ├── vite.config.ts <br>
│   ├── tailwind.config.js <br>
│   └── Dockerfile <br>
├── backend/ <br>                      # Python Backend
│   ├── app/ <br>
│   │   ├── __init__.py <br>
│   │   ├── main.py <br>
│   │   ├── core/ <br>                 # Core Layer
│   │   │   ├── __init__.py <br>
│   │   │   ├── config.py <br>
│   │   │   ├── exceptions.py <br>
│   │   │   ├── logging.py <br>
│   │   │   ├── security.py <br>
│   │   │   └── database.py <br>
│   │   ├── models/ <br>               # Domain Models Layer
│   │   │   ├── __init__.py <br>
│   │   │   ├── base.py <br>
│   │   │   ├── document.py <br>
│   │   │   ├── chat.py <br>
│   │   │   ├── user.py <br>
│   │   │   └── embedding.py <br>
│   │   ├── schemas/ <br>              # Pydantic Schemas Layer
│   │   │   ├── __init__.py <br>
│   │   │   ├── base.py <br>
│   │   │   ├── document.py <br>
│   │   │   ├── chat.py <br>
│   │   │   ├── user.py <br>
│   │   │   └── response.py <br>
│   │   ├── repositories/ <br>         # Data Access Layer
│   │   │   ├── __init__.py <br>
│   │   │   ├── base.py <br>
│   │   │   ├── document_repository.py <br>
│   │   │   ├── chat_repository.py <br>
│   │   │   ├── user_repository.py <br>
│   │   │   └── vector_repository.py <br>
│   │   ├── services/ <br>             # Business Logic Layer
│   │   │   ├── __init__.py <br>
│   │   │   ├── base.py <br>
│   │   │   ├── document_service.py <br>
│   │   │   ├── chat_service.py <br>
│   │   │   ├── embedding_service.py <br>
│   │   │   ├── llm_service.py <br>
│   │   │   ├── rag_service.py <br>
│   │   │   └── agent_service.py <br>
│   │   ├── api/ <br>                  # API Layer
│   │   │   ├── __init__.py <br>
│   │   │   ├── deps.py <br>
│   │   │   ├── middleware.py <br>
│   │   │   └── v1/ <br>
│   │   │       ├── __init__.py <br>
│   │   │       ├── auth.py <br>
│   │   │       ├── chat.py <br>
│   │   │       ├── documents.py <br>
│   │   │       ├── health.py <br>
│   │   │       └── websocket.py <br>
│   │   ├── agents/ <br>               # Agent System Layer
│   │   │   ├── __init__.py <br>
│   │   │   ├── base_agent.py <br>
│   │   │   ├── rag_agent.py <br>
│   │   │   ├── document_agent.py <br>
│   │   │   ├── search_agent.py <br>
│   │   │   └── orchestrator.py <br>
│   │   ├── utils/ <br>                # Utilities Layer
│   │   │   ├── __init__.py <br>
│   │   │   ├── text_processing.py <br>
│   │   │   ├── file_handler.py <br>
│   │   │   ├── embeddings.py <br>
│   │   │   ├── chunking.py <br>
│   │   │   └── validators.py <br>
│   │   └── workers/ <br>              # Background Workers Layer
│   │       ├── __init__.py <br>
│   │       ├── document_processor.py <br>
│   │       ├── embedding_worker.py <br>
│   │       └── indexing_worker.py <br>
│   ├── data/ <br>
│   │   ├── raw/ <br>
│   │   ├── processed/ <br>
│   │   └── embeddings/ <br>
│   ├── scripts/ <br>
│   │   ├── setup_database.py <br>
│   │   ├── process_vietnamese_data.py <br>
│   │   ├── create_embeddings.py <br>
│   │   └── migrate_data.py <br>
│   ├── tests/ <br>
│   │   ├── __init__.py <br>
│   │   ├── conftest.py <br>
│   │   ├── unit/ <br>
│   │   ├── integration/ <br>
│   │   └── e2e/ <br>
│   ├── requirements.txt <br>
│   ├── requirements-dev.txt <br>
│   ├── pyproject.toml <br>
│   ├── Dockerfile <br>
│   └── alembic/ <br>
├── docker/ <br>
│   ├── docker-compose.yml <br>
│   ├── docker-compose.prod.yml <br>
│   ├── nginx/ <br>
│   │   ├── nginx.conf <br>
│   │   └── Dockerfile <br>
│   └── postgres/ <br>
│       └── init.sql <br>
├── .github/ <br>
│   └── workflows/ <br>
│       ├── ci.yml <br>
│       ├── cd.yml <br>
│       └── test.yml <br>
├── docs/ <br>
│   ├── API.md <br>
│   ├── DEPLOYMENT.md <br>
│   ├── ARCHITECTURE.md <br>
│   └── USER_GUIDE.md <br>
├── .env.example <br>
├── .gitignore <br>
├── README.md <br>
└── Makefile <br>

