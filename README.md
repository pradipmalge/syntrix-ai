# SyntrixAI

**SyntrixAI** is an enterprise-oriented AI application demo built to
explore the engineering patterns required to build, operate, and scale
LLM-powered applications. It currently runs locally and combines a
provider-based LLM layer, conversational APIs, document ingestion,
retrieval-augmented generation (RAG), vector search, observability, and
an initial security/guardrails layer.

> **Project status:** Active development / technical demo. This
> repository demonstrates architecture and implementation patterns; it
> is not yet production-ready. Review the limitations and security notes
> before exposing it to users or deploying it publicly.

## Contents

-   [Overview](#overview)
-   [Implemented capabilities](#implemented-capabilities)
-   [Technology stack](#technology-stack)
-   [Architecture at a glance](#architecture-at-a-glance)
-   [Prerequisites](#prerequisites)
-   [Local setup](#local-setup)
-   [Run the application](#run-the-application)
-   [API documentation](#api-documentation)
-   [RAG workflow](#rag-workflow)
-   [Observability](#observability)
-   [Tests](#tests)
-   [Configuration](#configuration)
-   [Known limitations and security
    considerations](#known-limitations-and-security-considerations)
-   [Roadmap](#roadmap)
-   [Suggested demo flow](#suggested-demo-flow)

## Overview

SyntrixAI is being developed as a modular-monolith foundation for an
enterprise AI platform. The design separates API routes, application
services, domain contracts, and infrastructure adapters so that
providers and storage implementations can evolve without coupling the
whole application to one vendor.

The current local development setup uses:

-   Ollama to run a local chat model and embedding model.
-   PostgreSQL to persist conversations, messages, and document
    metadata.
-   Qdrant to store and search document vectors.
-   FastAPI to expose HTTP APIs.

## Implemented capabilities

### Application foundation

-   FastAPI application and API routing.
-   Centralized configuration and structured logging.
-   Shared exception handling and dependency injection.
-   Asynchronous database access with SQLAlchemy.
-   Alembic database migrations.

### LLM integration

-   Provider abstraction and provider factory.
-   Ollama chat integration.
-   Asynchronous generation and streaming interfaces.
-   Configurable model, temperature, token limit, and timeout.
-   Retry and provider-error handling foundations.

### Conversations

-   Conversation and message persistence in PostgreSQL.
-   APIs to create, retrieve, list, and delete conversations.
-   Message history retrieval.
-   Chat generation and streaming service foundations.

### Retrieval-augmented generation (RAG)

-   PDF, DOCX, and TXT document loaders.
-   Text chunking with overlap.
-   Ollama embedding integration using `nomic-embed-text`.
-   Qdrant vector storage and similarity search.
-   Tenant and knowledge-base filters in vector retrieval.
-   Document ingestion, status tracking, and retrieval APIs.
-   RAG answer generation with source context.
-   Optional conversation context and follow-up question rewriting.

### Observability

-   Prometheus metrics endpoint.
-   OpenTelemetry tracing foundations for the application, LLM service,
    and provider calls.
-   Trace identifiers in application logs.
-   Local Jaeger and OpenTelemetry Collector setup.

### Initial security and guardrails

-   Input validation for empty, oversized, null-byte, and disallowed
    control-character input.
-   Heuristic prompt-injection detection.
-   Prompt-injection checks integrated into the LLM and RAG request
    routes.

## Technology stack

  Area                  Technologies
  --------------------- ------------------------------------------------
  Language              Python
  API                   FastAPI, Pydantic
  LLM                   Ollama, LangChain integration
  Chat model            `qwen3:4b` (local development default)
  Embeddings            `nomic-embed-text` (768 dimensions)
  Relational database   PostgreSQL, SQLAlchemy, Alembic
  Vector database       Qdrant
  HTTP client           HTTPX
  Metrics               Prometheus
  Tracing               OpenTelemetry, Jaeger, OpenTelemetry Collector
  Containers            Docker, Docker Compose
  Tests                 pytest

## Architecture at a glance

The application follows a layered structure:

``` text
src/syntrix_ai/
├── api/                 # Application-level API routes and dependencies
├── application/         # Application services and orchestration
├── core/                # Configuration, logging, and shared concerns
├── domain/              # Core models and contracts
├── infrastructure/      # Database and external provider adapters
├── rag/
│   ├── api/             # Document, retrieval, and RAG answer endpoints
│   ├── application/     # Ingestion, retrieval, and answer services
│   ├── domain/          # RAG models, interfaces, and exceptions
│   └── infrastructure/  # Loaders, chunker, embeddings, Qdrant, persistence
└── guardrails/
    └── application/     # Input validation and prompt-injection checks
```

The exact repository may contain additional modules as development
continues.

## Prerequisites

The local setup assumes Windows with PowerShell, but the services can
also be run on other operating systems with equivalent commands.

Install or prepare:

1.  Python 3.12 or later (use a version supported by the dependencies in
    your lock file).
2.  `uv` for Python environment and dependency management.
3.  Docker Desktop with virtualization enabled.
4.  Ollama.
5.  Git.

You will need enough memory to run the local chat model, PostgreSQL,
Qdrant, and the application together. Model performance depends on your
hardware.

## Local setup

### 1. Clone the repository

``` powershell
git clone <YOUR_REPOSITORY_URL>
cd syntrix-ai
```

If the repository is already cloned, open PowerShell in the project
root.

### 2. Install dependencies

``` powershell
uv sync
```

If your project uses a different dependency group or optional extras,
install the groups declared by the repository's `pyproject.toml`.

### 3. Configure environment variables

Create a `.env` file in the project root. Use the variable names defined
in the project's settings module and `.env.example`, if present.

Typical local values include:

``` dotenv
DATABASE_URL=postgresql+asyncpg://syntrixai:syntrixai_dev@localhost:5432/syntrixai

OLLAMA_BASE_URL=http://localhost:11434
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:4b
LLM_REASONING=false

EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=syntrix_documents
```

**Important:** These are representative development settings. Confirm
the exact environment variable names against the current `Settings`
class and `.env.example` in your checkout. Do not commit real
credentials or production secrets.

### 4. Start Ollama and pull models

Ensure the Ollama service is running, then pull the chat and embedding
models:

``` powershell
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

Verify the models are available:

``` powershell
ollama list
```

The embedding model configured here produces 768-dimensional vectors.
The Qdrant collection dimension must match the embedding model output.

### 5. Start infrastructure services

Start the services defined in the repository's Docker Compose file:

``` powershell
docker compose up -d
```

The local setup uses PostgreSQL and Qdrant. Depending on the current
Compose configuration, observability services such as Jaeger and the
OpenTelemetry Collector may also be included.

Check container status:

``` powershell
docker compose ps
```

Useful local endpoints:

-   PostgreSQL: `localhost:5432`
-   Qdrant API: `http://localhost:6333`
-   Qdrant dashboard: `http://localhost:6333/dashboard`
-   Ollama API: `http://localhost:11434`

### 6. Apply database migrations

Run Alembic migrations from the project root:

``` powershell
uv run alembic upgrade head
```

If Alembic configuration is not at the repository root, use the path
configured by the project.

### 7. Run the API

``` powershell
uv run uvicorn syntrix_ai.main:app --reload --app-dir src
```

The API should be available at:

-   API base: `http://127.0.0.1:8000`
-   Swagger UI: `http://127.0.0.1:8000/docs`
-   OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

## Run the application

A typical local startup sequence is:

1.  Start Docker Desktop.
2.  Start Ollama.
3.  Start PostgreSQL and Qdrant using Docker Compose.
4.  Apply database migrations.
5.  Start the FastAPI application.
6.  Open Swagger UI and exercise the endpoints.

Keep the terminal running the API open so that application logs and
errors are visible.

## API documentation

Swagger UI at `/docs` is the source of truth for the endpoints exposed
by the current checkout.

The implemented API areas include:

  -----------------------------------------------------------------------
  Area                                Purpose
  ----------------------------------- -----------------------------------
  LLM                                 Submit prompts for model generation

  Conversations                       Create and manage conversations and
                                      messages

  Documents                           Upload, list, inspect, and delete
                                      knowledge-base documents

  Retrieval                           Search indexed chunks in a
                                      knowledge base

  RAG answers                         Ask questions using retrieved
                                      document context

  Metrics                             Expose Prometheus metrics
  -----------------------------------------------------------------------

The RAG routes include the following patterns:

``` text
POST /knowledge-bases/{knowledge_base_id}/search
POST /knowledge-bases/{knowledge_base_id}/ask
```

Document upload and conversation route paths should be checked in
Swagger because route prefixes can change during development.

## RAG workflow

The current document question-answering flow is:

``` text
Document upload
      |
      v
Document loader (PDF / DOCX / TXT)
      |
      v
Text extraction
      |
      v
Text chunking
      |
      v
Embedding generation (Ollama)
      |
      v
Vector storage (Qdrant)
      |
      v
User question
      |
      v
Query embedding
      |
      v
Tenant + knowledge-base filtered vector search
      |
      v
Retrieved chunks and source context
      |
      v
LLM answer generation
      |
      v
Answer and source metadata
```

When a conversation ID is supplied to the RAG answer endpoint, the
application can load prior conversation messages, rewrite a follow-up
question into a standalone retrieval query, and persist the user and
assistant messages.

## Observability

### Prometheus metrics

The application exposes a Prometheus-compatible metrics endpoint at:

``` text
http://127.0.0.1:8000/metrics
```

### Jaeger traces

When the local tracing stack is enabled, open:

``` text
http://localhost:16686
```

Use Jaeger to inspect traces emitted by the application and LLM request
flow.

**Current instrumentation note:** The development environment has
reported a compatibility warning between the installed SQLAlchemy
version and the SQLAlchemy OpenTelemetry instrumentation package.
Automatic SQLAlchemy spans may therefore be missing. Do not assume
database spans are present until this compatibility issue is resolved
and verified.

## Tests

Run the test suite with:

``` powershell
uv run pytest
```

Run a specific test file:

``` powershell
uv run pytest tests/unit/guardrails/test_input_validation_service.py -q
```

Run tests with verbose output:

``` powershell
uv run pytest -v
```

The repository includes unit and integration tests for areas such as
input validation, prompt-injection detection, document loaders,
chunking, embeddings, vector storage, retrieval, and persistence. The
exact test inventory evolves as implementation continues.

## Configuration

Configuration is centralized in the application's settings module and
loaded from environment variables / `.env`.

Review these settings before running locally:

-   Database URL and connection settings.
-   LLM provider, model, timeout, temperature, and token limit.
-   Ollama base URL.
-   Embedding model and vector dimension.
-   Qdrant URL and collection name.
-   Chunk size and overlap.
-   Logging and tracing configuration.

Do not place API keys, passwords, tokens, or other secrets in source
control.

## Known limitations and security considerations

This is an evolving technical demo. Important gaps include:

-   **Tenant identity:** Some development APIs accept `tenant_id` from
    the request. Production systems must derive tenant identity from an
    authenticated and authorized principal, not trust caller-supplied
    tenant IDs.
-   **Authorization and ownership:** Conversation and document access
    needs complete tenant ownership checks and authorization enforcement
    before production use.
-   **Prompt-injection detection:** The current detector is heuristic
    and pattern-based. It is not a complete prompt-injection defense and
    can produce false positives or miss attacks.
-   **Indirect prompt injection:** Retrieved document content is not yet
    comprehensively inspected as untrusted input.
-   **PII protection:** PII detection and masking are planned, not yet
    implemented.
-   **RAG grounding:** Source attribution, citation validation, and
    answer-grounding checks need further hardening.
-   **Model output handling:** Local model reasoning/thinking output
    (for example, `<think>` content) may require filtering before
    presenting responses to end users.
-   **Document extraction:** PDF OCR for scanned documents is not
    implemented.
-   **Retrieval quality:** Hybrid retrieval, reranking, and retrieval
    evaluation remain future enhancements.
-   **Database tracing:** SQLAlchemy automatic tracing has a known
    dependency compatibility warning; verify before relying on DB spans.
-   **Operational readiness:** Production-grade secrets management, rate
    limiting, quotas, audit coverage, deployment hardening,
    backup/restore, and load testing remain to be completed.

Do not expose the development instance publicly without implementing
authentication, authorization, tenant isolation, rate limiting, secret
management, and appropriate content/security controls.

## Roadmap

Planned areas of development include:

1.  Complete security and guardrails, including PII detection and
    masking.
2.  Strengthen tenant authentication, authorization, quotas, and audit
    logging.
3.  Add MCP client/server integration.
4.  Add agent workflows and orchestration with LangGraph.
5.  Add evaluation and regression testing for RAG and LLM behavior.
6.  Improve observability, tracing, and operational dashboards.
7.  Add advanced retrieval capabilities such as hybrid search and
    reranking.
8.  Add background processing and distributed workload patterns.
9.  Containerize and deploy to cloud infrastructure with CI/CD.
10. Improve developer experience and frontend workflows.

## Suggested demo flow

For a concise technical walkthrough:

1.  Start the local services and show the health of PostgreSQL, Qdrant,
    and Ollama.
2.  Open Swagger UI and create a conversation.
3.  Send a prompt through the LLM endpoint and show the provider
    abstraction.
4.  Upload a PDF, DOCX, or TXT file to a knowledge base.
5.  Explain the ingestion pipeline: loader, chunker, embedding provider,
    and Qdrant.
6.  Search the knowledge base and inspect retrieved chunks and
    similarity scores.
7.  Ask a question through the RAG endpoint and inspect the answer and
    source metadata.
8.  Ask a follow-up question using the same conversation ID.
9.  Show `/metrics` and Jaeger traces.
10. Demonstrate input validation and prompt-injection checks, explaining
    that these are initial guardrails rather than complete security
    controls.

------------------------------------------------------------------------

## Contributing

This project is under active development. Before submitting changes:

1.  Keep domain contracts independent of infrastructure implementations.
2.  Prefer dependency injection for external services.
3.  Add or update unit tests for application and domain behavior.
4.  Add integration tests for database, vector-store, and provider
    interactions where appropriate.
5.  Update this README and the detailed technical documentation when
    architecture or setup changes.

## License

Add the license and usage terms applicable to this repository before
distributing it.
