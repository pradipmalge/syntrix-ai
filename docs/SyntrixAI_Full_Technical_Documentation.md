**SYNTRIXAI**

Enterprise AI Platform -- Full Technical Documentation

**Setup • Architecture • Concepts • Implementation walkthrough • Testing
• Operations**

Version 1.0 \| 27 November 2026

> *Scope and accuracy: This guide documents the implementation discussed
> in the SyntrixAI project. It distinguishes implemented capabilities
> from planned and deferred work. Verify exact filenames, environment
> aliases, service names and route prefixes against the current
> repository. This is a demo application and is not yet
> production-ready.*

# 1. Product overview

SyntrixAI is an enterprise-oriented AI platform demo. Its current
foundation covers a FastAPI application, model-provider abstraction,
local Ollama inference, PostgreSQL persistence, document ingestion,
embeddings, Qdrant semantic retrieval, RAG answer generation,
conversation-aware RAG, metrics/tracing foundations and first-layer
guardrails.

The architecture is a **modular monolith**: domain contracts,
application use cases, infrastructure adapters and API routes live in
one deployable Python application. PostgreSQL, Qdrant, Ollama and
observability services run as external processes or containers.

## 1.1 Current technology stack

  -----------------------------------------------------------------------
  **Layer**        **Technology**           **Purpose**
  ---------------- ------------------------ -----------------------------
  API              Python, FastAPI,         HTTP endpoints, schemas,
                   Pydantic                 dependency injection

  LLM              LangChain integration,   Model interaction
                   ChatOllama               

  Local inference  Ollama, qwen3:4b         Chat generation

  Embeddings       Ollama, nomic-embed-text 768-dimensional text vectors

  Relational data  PostgreSQL, SQLAlchemy   Conversations, messages,
                   async, Alembic           document records

  Vector search    Qdrant                   Chunk vectors and metadata

  Metrics          Prometheus client        Application and LLM metrics

  Tracing          OpenTelemetry,           Request and provider tracing
                   Collector, Jaeger        

  Local            Docker Desktop / Compose Supporting services
  infrastructure                            

  Python tooling   uv, pytest               Environment, dependencies and
                                            tests
  -----------------------------------------------------------------------

## 1.2 Implemented capabilities

-   FastAPI application, routing, configuration, logging and dependency
    injection.

-   LLM service and provider abstraction; Ollama is the current
    provider.

-   Asynchronous LLM generation and streaming interfaces.

-   Conversation and message persistence in PostgreSQL.

-   PDF, DOCX and TXT document loaders.

-   Recursive chunking with configured size and overlap.

-   Ollama embeddings using \`nomic-embed-text\` (768 dimensions).

-   Qdrant vector collection, upsert and tenant/knowledge-base-filtered
    search.

-   Document ingestion pipeline and document status persistence.

-   Semantic retrieval API and RAG answer API.

-   Optional conversation context and follow-up question rewriting for
    RAG.

-   Prometheus metrics and OpenTelemetry tracing foundations.

-   Input validation and rule-based prompt-injection
    detection/enforcement integrated into LLM and RAG ask endpoints.

# 2. Architecture

## 2.1 High-level flow

Client / Swagger

\|

v

FastAPI API layer

\|\-- LLM routes

\|\-- Conversation routes

\|\-- Document routes

\|\-- Retrieval routes

\|\-- RAG answer routes

\|

v

Application services

\|\-- LLMService

\|\-- ConversationService / ContextService

\|\-- DocumentIngestionService

\|\-- RetrievalService

\|\-- QuestionRewriterService

\|\-- RAGAnswerService

\|\-- Guardrail services

\|

+\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--+\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--+

\| \| \|

v v v

Provider adapter Persistence adapter Vector adapter

Ollama / ChatOllama SQLAlchemy/PostgreSQL Qdrant client

## 2.2 Architectural layers

  ---------------------------------------------------------------------------
  **Layer**        **Responsibility**           **Examples**
  ---------------- ---------------------------- -----------------------------
  API              HTTP boundary,               FastAPI routers
                   request/response schemas,    
                   status codes                 

  Application      Coordinates business         Ingestion, retrieval, RAG,
                   workflows                    conversation services

  Domain           Core concepts, interfaces    Document, DocumentChunk,
                   and exceptions               EmbeddingProvider,
                                                VectorStore

  Infrastructure   External systems and         Ollama, Qdrant, SQLAlchemy,
                   persistence adapters         loaders
  ---------------------------------------------------------------------------

**Why modular monolith first?** It keeps local development and
deployment simple while preserving boundaries. A service should be
extracted only when scaling, ownership or deployment requirements
justify distributed-system complexity.

## 2.3 Logical repository structure

syntrix-ai/

\|\-- pyproject.toml

\|\-- uv.lock

\|\-- .env

\|\-- docker-compose.yml

\|\-- alembic.ini

\|\-- migrations/

\|\-- src/syntrix_ai/

\| \|\-- main.py

\| \|\-- api/

\| \| \|\-- dependencies.py

\| \| \|\-- router.py

\| \| \|\-- routes/

\| \|\-- core/

\| \| \|\-- config.py

\| \| \|\-- logging.py

\| \|\-- domain/

\| \|\-- application/

\| \| \|\-- services/

\| \|\-- infrastructure/

\| \| \|\-- persistence/

\| \|\-- rag/

\| \| \|\-- domain/

\| \| \|\-- application/

\| \| \|\-- infrastructure/

\| \| \| \|\-- loaders/

\| \| \| \|\-- chunking/

\| \| \| \|\-- embeddings/

\| \| \| \|\-- vectorstores/

\| \| \| \|\-- persistence/

\| \| \|\-- api/

\| \| \|\-- dependencies.py

\| \| \|\-- schemas.py

\| \| \|\-- routes/

\| \|\-- guardrails/

\| \|\-- application/services/

\| \|\-- input_validation_service.py

\| \|\-- prompt_injection_detector.py

\| \|\-- prompt_injection_guard_service.py

\|\-- tests/

\|\-- unit/

\|\-- integration/

This is a logical map, not an exhaustive filesystem listing. The actual
repository may contain more modules.

# 3. Local setup

## 3.1 Prerequisites

-   Windows 10/11 development workstation.

-   Python environment compatible with the dependency lock. The project
    has been exercised with Python 3.14.6; check library compatibility
    if changing versions.

-   \`uv\` for Python package management.

-   Docker Desktop and WSL 2 / hardware virtualization.

-   Ollama for local model serving.

-   Git.

-   PowerShell.

Docker Desktop may fail to start if AMD-V/SVM or Intel VT-x is disabled
in firmware. Enable hardware virtualization and required Windows
virtualization features, restart Windows, then verify Docker Desktop is
running.

## 3.2 Clone and install

git clone \<your-syntrix-ai-repository-url\>

cd syntrix-ai

uv sync

uv run python \--version

If already cloned, open the existing project directory. The working
directory used in this project was D:\\AI-WS\\SYNTRIX-AI\\syntrix-ai.

uv sync installs the dependencies declared by the project and lock file.
Prefer uv run \<command\> to consistently execute inside the project
environment.

## 3.3 Environment configuration

The following values are known from the current setup. Preserve any
other settings already present in .env; exact environment aliases and
defaults are defined by src/syntrix_ai/core/config.py.

DATABASE_URL=postgresql+asyncpg://syntrixai:syntrixai_dev@localhost:5432/syntrixai

OLLAMA_BASE_URL=http://localhost:11434

LLM_REASONING=false

EMBEDDING_MODEL=nomic-embed-text

EMBEDDING_DIMENSION=768

EMBEDDING_TIMEOUT_SECONDS=60

QDRANT_URL=http://localhost:6333

QDRANT_COLLECTION_NAME=syntrix_documents

RAG_CHUNK_SIZE=1000

RAG_CHUNK_OVERLAP=200

Do not commit .env, passwords, API keys or production secrets.

## 3.4 Start infrastructure

Start Docker Desktop, then from the repository root:

docker compose up -d

docker compose ps

The Compose file is the source of truth for service names and ports.
Qdrant is configured on ports 6333 (HTTP) and 6334 (gRPC), with a
persistent Docker volume.

Invoke-WebRequest http://localhost:6333/healthz

Start-Process http://localhost:6333/dashboard

PostgreSQL is expected at localhost:5432 with database syntrixai; the
development connection string is shown above. If service names differ,
inspect docker compose ps and docker compose logs \<service\>.

## 3.5 Install and verify Ollama models

Install Ollama and ensure its API is available at
http://localhost:11434.

ollama pull qwen3:4b

ollama pull nomic-embed-text

ollama list

qwen3:4b is the chat model. nomic-embed-text is the embedding model. The
embedding dimension configured in the application and Qdrant collection
is 768; query and document vectors must have matching dimensions.

## 3.6 Apply database migrations

uv run alembic upgrade head

uv run alembic current

The migration head observed during the latest project work was
2fe9d6b100c5. Use the current repository migration state if it has
changed.

## 3.7 Run the API

uv run uvicorn syntrix_ai.main:app \--reload \--app-dir src

Open Swagger UI at http://127.0.0.1:8000/docs. The API routes have been
used under the /api/v1 prefix; confirm the exact prefix in the running
Swagger document.

## 3.8 Setup verification checklist

  ---------------------------------------------------------------------
  **Dependency**           **Verification**
  ------------------------ --------------------------------------------
  API                      Swagger \`/docs\` loads

  PostgreSQL               Application connects using DATABASE_URL

  Alembic                  \`uv run alembic current\` reports current
                           revision

  Ollama                   API responds and both models appear in
                           \`ollama list\`

  Qdrant                   \`/healthz\` responds; dashboard opens

  Metrics                  \`/metrics\` is available if registered

  Jaeger                   \`http://localhost:16686\` opens when
                           tracing stack is running
  ---------------------------------------------------------------------

# 4. Core concepts and implementation

## 4.1 LLM provider abstraction

An LLM provider abstraction keeps application workflows independent of a
specific model vendor. The application sends a normalized LLMRequest
containing messages and generation settings. LLMService delegates to the
selected provider adapter and returns an LLMResponse with content,
provider/model identity, token usage and metadata.

-   \`LLMMessage\`: system, user or assistant message.

-   \`LLMRequest\`: message list, optional model, temperature and token
    limit.

-   \`LLMResponse\`: generated content, model, provider, token usage and
    metadata.

-   \`OllamaProvider\`: current adapter based on ChatOllama.

This allows other providers or a model gateway to be introduced without
rewriting RAG and conversation use cases.

## 4.2 Conversations and messages

Conversation and Message are relational entities in PostgreSQL. A
conversation groups messages; each message stores role, content,
provider/model details, token usage and timestamp. Repositories
encapsulate persistence operations and ConversationService coordinates
use cases.

ConversationContextService loads recent messages and maps them into LLM
messages. The existing conversation chat service supports normal chat
and streaming flows.

## 4.3 Document ingestion

Ingestion converts a source file into searchable chunks and vectors.
PostgreSQL stores document metadata and status; Qdrant stores chunk
vectors and payloads.

Upload

-\> create document (PENDING)

-\> mark PROCESSING

-\> select loader

-\> extract text

-\> split into chunks

-\> generate embeddings

-\> upsert into Qdrant

-\> mark INDEXED

-\> on error, mark FAILED where possible

DocumentIngestionService coordinates the pipeline. A loader registry
selects the file parser, the chunker splits text, the embedding adapter
creates vectors and the vector store writes points into Qdrant.

### Loaders

  ----------------------------------------------------------------------
  **Loader**   **Behavior**                     **Limitation**
  ------------ -------------------------------- ------------------------
  PDF          pypdf extracts text page-by-page No OCR for scanned PDFs
               and retains page metadata        

  DOCX         python-docx extracts paragraphs  Complex layouts may need
               and tables                       richer extraction

  TXT          Handles UTF-8 BOM and UTF-16     Other encodings may need
               fallback                         explicit support
  ----------------------------------------------------------------------

### Chunking

RecursiveTextChunker splits text into bounded chunks with overlap.
Current settings are 1,000 characters and 200 characters overlap.
Overlap helps preserve context across chunk boundaries. Deterministic
UUID5 IDs are used for chunks. Offsets refer to normalized extracted
text, not exact offsets in original PDF bytes or page layout.

### Embeddings

An embedding model maps text into a numeric vector. SyntrixAI calls
Ollama\'s embedding API with nomic-embed-text and expects 768
dimensions. The same model and compatible preprocessing must be used for
document and query embeddings.

### Qdrant

Qdrant stores chunk vectors with payload fields such as chunk_id,
document_id, tenant_id, knowledge_base_id, content, chunk_index and
metadata. Search applies tenant and knowledge-base filters. The
collection uses cosine similarity and payload indexes for common filter
fields.

## 4.4 Retrieval

The RetrievalService validates the query, tenant ID, knowledge-base ID
and top-k value. It embeds the query and calls the vector store with the
tenant/knowledge-base filters, result limit and optional score
threshold. Results contain the chunk and similarity score.

## 4.5 Retrieval-Augmented Generation (RAG)

RAG retrieves relevant document chunks and supplies them to the LLM as
evidence. Retrieved chunks are labeled \[S1\], \[S2\], and so on. The
answer prompt instructs the model to use the reference material and the
API returns the answer plus structured source metadata.

Question

-\> query embedding

-\> Qdrant retrieval

-\> source labels \[S1\]\...\[Sn\]

-\> grounded prompt

-\> LLM generation

-\> answer + sources

RAG grounds answers in application data but does not guarantee
correctness. Retrieval can miss evidence, the model can misinterpret
content and citation consistency must be validated.

## 4.6 Conversation-aware RAG

The RAG ask request optionally accepts conversation_id. When supplied,
the service loads recent history, rewrites the follow-up into a
standalone search query, persists the new user message, retrieves
chunks, generates an answer using conversation history and retrieved
evidence, and persists the assistant response.

Question + conversation_id

-\> load recent history

-\> rewrite follow-up as standalone query

-\> persist user message

-\> retrieve chunks

-\> generate answer using history + references

-\> persist assistant response

-\> return answer + sources

The existing PostgreSQL conversation/message tables are reused; no
separate memory database was introduced. Without a conversation ID, the
request is stateless. The query rewrite itself was not separately
inspected in the latest manual test because that verification was
explicitly skipped.

## 4.7 Guardrails implemented

  -------------------------------------------------------------------------------------------
  **Component**                 **Responsibility**                        **Current state**
  ----------------------------- ----------------------------------------- -------------------
  InputValidationService        Type check, trim, reject empty, enforce   Implemented and
                                max length, reject null bytes and         integrated into LLM
                                disallowed control chars; allow           and RAG ask routes
                                tabs/newlines                             

  PromptInjectionDetector       Rule-based detection for instruction      Implemented and
                                override, role manipulation, prompt       unit tested
                                extraction and instruction bypass         

  PromptInjectionGuardService   Converts detection result into an         Implemented and
                                allow/block decision and raises a         unit tested
                                dedicated exception when blocked          

  LLM route                     Validates every message and inspects it   Integrated
                                before provider invocation                

  RAG ask route                 Validates and inspects the question       Integrated
                                before RAG execution                      
  -------------------------------------------------------------------------------------------

The API boundaries configure a 12,000-character maximum. The detector is
heuristic, not a semantic security classifier. It can miss novel attacks
and may produce false positives. The LLM route currently inspects every
message role, which should be reviewed during hardening. The RAG route
currently checks the incoming question, not malicious instructions
embedded in retrieved documents.

# 5. API walkthrough

Confirm exact prefixes and schemas in Swagger. The following routes are
based on the project implementation discussed.

  ------------------------------------------------------------------------------------------------
  **Endpoint**                                           **Purpose**
  ------------------------------------------------------ -----------------------------------------
  \`POST /api/v1/llm/generate\`                          Generate a response from messages; input
                                                         validation and prompt-injection checks
                                                         run first

  \`POST /api/v1/documents\`                             Upload and ingest a supported document

  \`GET /api/v1/documents\`                              List documents

  \`GET /api/v1/documents/{document_id}\`                Get document details

  \`DELETE /api/v1/documents/{document_id}\`             Delete a document and indexed content,
                                                         subject to implementation cleanup
                                                         semantics

  \`POST                                                 Semantic retrieval
  /api/v1/knowledge-bases/{knowledge_base_id}/search\`   

  \`POST                                                 RAG answer; optional conversation ID
  /api/v1/knowledge-bases/{knowledge_base_id}/ask\`      

  \`POST /api/v1/conversations\`                         Create a conversation

  \`GET                                                  Read conversation history
  /api/v1/conversations/{conversation_id}/messages\`     
  ------------------------------------------------------------------------------------------------

## 5.1 LLM generation

In Swagger, open the LLM generate endpoint and submit a normal user
message. A successful response contains generated content,
provider/model and token usage. Then submit a known prompt-injection
phrase to verify it is blocked before the provider call.

## 5.2 Document ingestion

Upload a PDF, DOCX or TXT with the tenant and knowledge-base identifiers
required by the request schema. The implementation described enforces a
10 MiB upload limit. The result includes document ID, filename, chunk
count, embedding dimension and status.

## 5.3 Semantic search

Submit tenant ID, query, top-k and optional score threshold. Check that
the returned chunk content belongs to the requested tenant and knowledge
base.

## 5.4 RAG ask

Submit tenant ID, question, top-k and optional score threshold. Include
a conversation ID to enable context-aware RAG. Inspect both answer and
sources. An empty source array means the answer has no retrieved
document evidence.

## 5.5 Conversation persistence

Create a conversation, send messages through conversation or RAG
workflows, then fetch its messages. Verify role, content, conversation
ID, timestamps, provider/model and token usage.

## 5.6 Guardrail behavior

  ----------------------------------------------------------------------
  **Input**                              **Expected result**
  -------------------------------------- -------------------------------
  Normal technical question              Allowed

  Empty or whitespace-only message       HTTP 422 at validated endpoints

  Text over 12,000 characters            HTTP 422

  Null byte or disallowed control        HTTP 422
  character                              

  "Ignore all previous instructions"     HTTP 400 under current rules

  "Reveal your system prompt"            HTTP 400 under current rules
  ----------------------------------------------------------------------

# 6. Testing

Run all tests:

uv run pytest -v

Run guardrail tests:

uv run pytest tests/unit/guardrails -v

Run a focused test:

uv run pytest tests/unit/guardrails/test_input_validation_service.py -v

The following guardrail results were explicitly reported during
implementation:

-   Input validation: 7 tests passed.

-   Prompt injection detector: 9 tests passed.

-   Prompt injection guard service: 6 tests passed.

-   Endpoint integration tests were reported passing after LLM and RAG
    integration.

Some test runs showed 17 warnings; those runs still passed. A warning
was also observed around SQLAlchemy instrumentation compatibility,
discussed in the observability section.

Recommended test coverage:

  ----------------------------------------------------------------------
  **Area**               **Verify**
  ---------------------- -----------------------------------------------
  Input validation       Empty, whitespace, max length, null byte,
                         control chars, valid text

  Prompt detector        Safe text, each category, case-insensitivity,
                         whitespace, multiple matches

  Guard service          Allowed and blocked cases, detection categories

  Chunker                Chunk size, overlap, boundaries, IDs, metadata

  Loaders                PDF/DOCX/TXT extraction and encoding

  Retrieval              Query validation, embedding invocation,
                         tenant/KB filters

  Qdrant integration     Upsert, search and tenant isolation

  Persistence            Conversation, message and document repository
                         behavior

  RAG integration        Ingestion → retrieval → answer and
                         conversation-aware flow
  ----------------------------------------------------------------------

# 7. Observability and operations

## 7.1 Metrics

Prometheus client instrumentation and a /metrics endpoint have been
added. The application records request-level and LLM-related metrics,
including token usage where available. Use metrics to inspect request
volume, latency, errors and token consumption.

## 7.2 Tracing

OpenTelemetry tracing is present for FastAPI and key operations
including LLMService, ConversationChatService and OllamaProvider. An
OTel Collector and Jaeger are used locally. Jaeger UI is available at
http://localhost:16686 when the stack is running.

## 7.3 Trace-aware logs

A logging filter adds trace_id, span_id and sampled fields so log
messages can be correlated with traces.

**Known issue:** SQLAlchemy instrumentation reported that the installed
instrumentor supports SQLAlchemy versions below 2.1, while the
environment had SQLAlchemy 2.1.9. Automatic SQLAlchemy spans were
therefore not active. Application tracing and log context still worked.
Do not assume database spans are present until compatibility is
resolved.

## 7.4 Operational habits

-   Check API health and Swagger first.

-   Confirm Ollama is running and required models exist.

-   Confirm PostgreSQL and migrations are healthy.

-   Confirm Qdrant health and vector dimension.

-   Use application logs and Jaeger traces to isolate failures.

-   Do not log full prompts, secrets, personal data or confidential
    retrieved content.

# 8. Data and tenancy

## 8.1 PostgreSQL

The development connection string is:

postgresql+asyncpg://syntrixai:syntrixai_dev@localhost:5432/syntrixai

The application uses an async SQLAlchemy engine and session factory.
Repositories encapsulate persistence.

## 8.2 Qdrant collection

-   Collection: \`syntrix_documents\`

-   Vector size: 768

-   Distance: cosine

-   Payload: chunk, document, tenant, knowledge base, content and
    metadata fields.

## 8.3 Document statuses

  ----------------------------------------------------------------------
  **Status**              **Meaning**
  ----------------------- ----------------------------------------------
  PENDING                 Record created; processing has not started

  PROCESSING              Extraction, chunking, embedding or indexing is
                          underway

  INDEXED                 Successfully indexed

  FAILED                  An ingestion stage failed

  DELETED                 Document marked/handled as deleted according
                          to repository behavior
  ----------------------------------------------------------------------

## 8.4 Tenant isolation caveat

Document and vector search include tenant and knowledge-base filters.
However, tenant ID is currently supplied by the caller rather than
derived from an authenticated identity. Conversations do not yet have
tenant/user ownership columns. A production application must derive
tenant identity from trusted authentication and authorize every
read/write operation. Caller-supplied IDs are not an authorization
mechanism.

# 9. Known limitations and deferred work

These items were intentionally deferred and must not be described as
completed:

  -----------------------------------------------------------------------
  **Area**           **State**
  ------------------ ----------------------------------------------------
  PII detection and  Not started. Next planned task: email, phone, Indian
  masking            PAN, Aadhaar-like values and IP addresses

  Indirect prompt    Retrieved document and tool-output attacks need
  injection          separate controls

  Prompt-injection   Current regex rules are a first layer only
  robustness         

  Citation           A previous response cited \`\[S1\]\` when no sources
  correctness        were returned; deferred

  \`\<think\>\`      Ollama output included reasoning artifacts in
  handling           response/persisted content; filtering deferred

  Hybrid retrieval   Deferred
  and reranking      

  OCR                Scanned PDFs are not handled by current PDF text
                     extraction

  Chunk provenance   Exact PDF page/source offset mapping needs
                     improvement

  Conversation       A failure after saving a user message can leave an
  failure            incomplete turn; recovery/transaction design
  consistency        deferred

  Conversation       Tenant/user ownership and authorization are not
  ownership          complete

  Authentication and Production identity, API keys/JWT and role
  RBAC               authorization are future work

  SQLAlchemy tracing Instrumentor compatibility needs resolution

  Production         Kubernetes, cloud infrastructure, secrets, CI/CD and
  deployment         production hardening are future work
  -----------------------------------------------------------------------

**Production warning:** The current implementation is a technical demo
foundation, not a production-ready multi-tenant platform. In particular,
caller-controlled tenant IDs and missing conversation ownership must be
fixed before use with untrusted tenants.

# 10. Roadmap

  --------------------------------------------------------------------------
  **Phase**   **Focus**            **Current state**
  ----------- -------------------- -----------------------------------------
  1           Project foundation,  Foundation implemented
              configuration,       
              FastAPI structure    

  2           LLM integration and  Ollama implementation present; more
              provider abstraction providers planned

  3           Persistence and      Conversations/messages implemented;
              conversation APIs    ownership hardening pending

  4           Reliability and      Metrics/tracing foundations present;
              observability        additional resilience and tracing work
                                   remains

  5           Knowledge ingestion  Core ingestion, retrieval, answer
              and RAG              generation and conversation-aware RAG
                                   implemented; quality hardening deferred

  6           Security and         Input validation and first-layer
              guardrails           prompt-injection controls implemented;
                                   PII is next

  7           MCP integration      Planned: client/server, tools, resources,
                                   prompts and authentication

  8           Agent workflows      Planned: LangGraph state, memory,
                                   checkpointing, human-in-the-loop and
                                   multi-agent patterns

  9           Evaluation           Planned: evaluation datasets, RAG/agent
                                   evaluation, regression and quality
                                   metrics

  10          Scale and deployment Planned: hardened multi-tenancy,
                                   Redis/Kafka/workers, Kubernetes, cloud
                                   and CI/CD
  --------------------------------------------------------------------------

## 10.1 Next planned implementation step

Development was paused immediately before **Phase 6, Step 3 -- PII
Detection & Protection**. The proposed scope is a reusable detector for
email addresses, phone numbers, Indian PAN, Aadhaar-like values and IP
addresses, followed by masking support and unit tests.

## 10.2 Deferred RAG hardening sequence

1.  Prevent reasoning artifacts from being returned or persisted.

2.  Make citations consistent with actual retrieved sources.

3.  Handle partial conversation failures and persistence consistency.

4.  Run end-to-end tests with high-quality answerable documents.

5.  Enforce tenant ownership and access controls.

# 11. Troubleshooting

  ------------------------------------------------------------------------
  **Symptom**      **Likely cause**          **Check**
  ---------------- ------------------------- -----------------------------
  Docker reports   Firmware virtualization   Enable AMD SVM/AMD-V or Intel
  virtualization   or Windows features       VT-x; enable WSL2/Virtual
  disabled         disabled                  Machine Platform; restart

  Compose cannot   Command run outside       \`Get-Location\`,
  find config      project root              \`Get-ChildItem\`

  PostgreSQL       Container stopped, port   \`docker compose ps\`, logs,
  connection       mismatch or URL incorrect DATABASE_URL
  refused                                    

  Alembic          DB unavailable or         Verify DB health and URL
  connection       configuration mismatch    
  failure                                    

  Ollama           Ollama service stopped or \`ollama list\`; call local
  connection error wrong base URL            tags endpoint

  Embedding        Model or Qdrant vector    Verify configured 768
  dimension        size differs              dimension and collection
  mismatch                                   

  Qdrant           Container or port issue   Check \`docker compose ps\`
  unavailable                                and \`/healthz\`

  RAG returns no   No relevant chunks, wrong Inspect document status,
  sources          tenant/KB, high threshold points, filters and query
                   or weak retrieval         

  Citation exists  Model emitted an          Known deferred
  but source list  unsupported citation      citation-hardening issue
  is empty                                   

  Tests show       Dependency/runtime        Distinguish warnings from
  warnings         compatibility warnings    test failures; inspect
                                             SQLAlchemy warning

  PowerShell       Bash-style stdin          Use PowerShell piping or run
  rejects \`\<\`   redirection used in       command in Bash
  redirection      PowerShell                
  ------------------------------------------------------------------------

Useful commands:

Get-Location

Get-ChildItem

Get-ChildItem src/syntrix_ai/guardrails -Recurse

docker compose up -d

docker compose ps

uv run uvicorn syntrix_ai.main:app \--reload \--app-dir src

uv run pytest -v

uv run pytest tests/unit/guardrails -v

# 12. Suggested live demo walkthrough

6.  Start Docker services, Ollama and FastAPI. Open Swagger.

7.  Generate a normal LLM response.

8.  Send a known prompt-injection phrase and demonstrate the block.

9.  Upload a sample PDF or DOCX to a knowledge base.

10. Show the document status and chunk count.

11. Run semantic search and inspect chunk text and scores.

12. Ask an answerable RAG question and show citations and sources.

13. Create a conversation and ask a follow-up using its conversation ID.

14. Fetch conversation messages to demonstrate persistence.

15. Open Jaeger and show a request trace; show metrics if enabled.

Use a small document with clear, testable facts. Verify it is indexed
before the demo. This avoids confusing weak retrieval quality with a
broken ingestion pipeline.

# 13. Engineering decisions -- interview talking points

**Why provider abstraction?** It isolates vendor-specific behavior and
allows local or hosted models to be swapped without changing business
workflows.

**Why PostgreSQL and Qdrant?** PostgreSQL handles transactional entities
and conversation history. Qdrant handles vector similarity search over
document chunks.

**Why chunking and overlap?** Models have bounded input sizes. Chunking
creates retrievable units; overlap reduces context loss at chunk
boundaries.

**Why tenant filters?** Similarity search is not authorization. Tenant
and knowledge-base filters scope retrieval, but production tenant
identity must come from trusted authentication.

**Why question rewriting?** Follow-ups often omit context. Rewriting
turns them into standalone retrieval queries while preserving the
original question for answer generation.

**Why a modular monolith?** It keeps the demo deployable and debuggable
while preserving boundaries that can be extracted later if needed.

**What demonstrates AI platform engineering?** Model integration,
provider abstraction, embedding and retrieval pipelines, persistence,
async APIs, observability, security entry points and a roadmap for
evaluation, agents and deployment.

# 14. Pre-demo readiness checklist

-   [ ] Dependencies install using \`uv sync\`.

-   [ ] Docker services are healthy.

-   [ ] PostgreSQL is reachable and migrations are current.

-   [ ] Ollama is running and both models are installed.

-   [ ] Qdrant is healthy and uses the expected vector dimension.

-   [ ] FastAPI starts and Swagger loads.

-   [ ] Full pytest suite passes.

-   [ ] Sample document is indexed and contains relevant chunks.

-   [ ] Normal LLM request succeeds.

-   [ ] Normal RAG question returns relevant sources.

-   [ ] Prompt-injection examples are blocked by current rules.

-   [ ] No secrets or personal data appear in logs or screenshots.

-   [ ] Deferred features are not presented as implemented.

# Appendix A. Glossary

  ---------------------------------------------------------------------
  **Term**              **Meaning**
  --------------------- -----------------------------------------------
  LLM                   Large Language Model; generates text from
                        prompts and context

  Inference             Running a trained model to produce output

  Embedding             Numeric vector representation of text or data

  Vector database       Database optimized for vector similarity search

  RAG                   Retrieval-Augmented Generation; retrieves
                        evidence and supplies it to an LLM

  Chunk                 Segment of a document used for embedding and
                        retrieval

  Top-k                 Number of highest-ranked retrieval results

  Score threshold       Cutoff used to exclude low-scoring search
                        results

  Tenant isolation      Preventing one customer from accessing another
                        customer\'s data

  Prompt injection      Attempt to manipulate model behavior using
                        malicious instructions

  PII                   Personally Identifiable Information

  MCP                   Model Context Protocol for connecting AI
                        applications to tools and context

  Agent                 AI-driven workflow that can select actions or
                        tools based on goals and state

  OpenTelemetry         Framework for traces, metrics and logs

  Jaeger                Distributed tracing UI/backend

  Idempotency           Repeating an operation has the same effect as
                        performing it once

  Circuit breaker       Temporarily stops calls to a failing dependency
  ---------------------------------------------------------------------

# Appendix B. Source-of-truth modules

Use the actual repository as the final authority when continuing
development:

  -------------------------------------------------------------------------------------
  **Concern**        **Module**
  ------------------ ------------------------------------------------------------------
  Configuration      \`src/syntrix_ai/core/config.py\` and \`.env\`

  Startup and        \`src/syntrix_ai/main.py\`; \`src/syntrix_ai/api/router.py\`
  routing            

  LLM route/service  \`src/syntrix_ai/api/routes/llm.py\`;
                     \`src/syntrix_ai/application/services/llm_service.py\`

  Database           \`src/syntrix_ai/infrastructure/persistence/database.py\`;
                     \`base.py\`

  Conversation       \`src/syntrix_ai/application/services/conversation_service.py\`;
  services           \`conversation_context_service.py\`

  RAG contracts      \`src/syntrix_ai/rag/domain/models.py\`; \`interfaces.py\`

  Ingestion and      \`src/syntrix_ai/rag/application/services/\`
  retrieval          

  RAG API            \`src/syntrix_ai/rag/api/routes/\`; \`schemas.py\`;
                     \`dependencies.py\`

  Vector store and   \`src/syntrix_ai/rag/infrastructure/vectorstores/\`;
  embeddings         \`embeddings/\`

  Guardrails         \`src/syntrix_ai/guardrails/application/services/\`

  Migrations         \`migrations/\`; \`alembic.ini\`

  Tests              \`tests/unit/\`; \`tests/integration/\`
  -------------------------------------------------------------------------------------

# Appendix C. Security and implementation cautions

-   Treat retrieved documents and tool outputs as untrusted data, not
    instructions.

-   Never expose internal reasoning artifacts in API responses or
    persisted messages.

-   Validate source identifiers before emitting citations; models can
    invent citation labels.

-   A tenant ID in a JSON body is not proof of identity or
    authorization.

-   Avoid logging prompts, PII, credentials and confidential retrieved
    content.

-   Keep model timeouts, concurrency limits, retries, token budgets and
    rate limits explicit.

-   Use unit tests and integration tests against real local
    dependencies.
