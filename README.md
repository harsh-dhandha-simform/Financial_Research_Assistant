# Financial Research Assistant

A multi-agent, supervisor-led financial analysis pipeline designed to autonomously ingest, chunk, index, and analyze SEC filings (10-K, 10-Q, 8-K, DEF 14A), corporate reports, and web-based financial news. It is orchestrated via LangGraph, enabling asynchronous parallel execution of specialized agents, structured LLM-bound tools, and a comprehensive synthesis process.

## 🚀 Key Features

- **Supervisor-led Orchestration (LangGraph)**:
  - **Dynamic Routing**: An LLM-based supervisor analyzes user queries to dynamically determine which specialist agents need to run.
  - **Parallel Execution**: Uses `asyncio.gather` under the hood to run selected specialist agents concurrently, bringing total processing time down to the slowest agent rather than their sum.
  - **State Preservation**: Persists intermediate agent outputs using a local SQLite-backed LangGraph checkpointer.
- **Autonomous Specialist Agents (ReAct Loop)**:
  - **Metrics Agent**: Automatically retrieves SEC filings to extract raw tabular financial figures (e.g., EBITDA margins, cash flow details, EPS) and parses material events.
  - **Risk Agent**: Targets risk-factor sections and groups issues by category (e.g., Operational, Financial, Legal, Competitor) and severity.
  - **News Agent**: Sources real-time corporate updates via Tavily and evaluates overall market sentiment using sentiment score mapping.
  - **Synthesis Agent**: Reviews metrics, risk, and news outputs, performs cross-section context retrieval, and formats a unified investment thesis (Buy/Hold/Sell) with confidence scores.
- **Hierarchical Parent-Child Chunking**:
  - Leverages **Chonkie** chunkers to implement a hierarchical retrieval pipeline.
  - Detects standard SEC item boundaries (MD&A, Risk Factors, Business description, Financial Statements) through multi-line regular expression pattern matching.
  - **Parent Chunks** (~2,000 tokens): Chunked using `RecursiveChunker` to preserve paragraph structures and tabular row relationships.
  - **Child Chunks** (~500 tokens): Chunked using `SentenceChunker` to ensure no sentence is cut mid-way, optimizing embedding alignment.
- **Hybrid Retrieval System**:
  - **Dense Retrieval**: Utilizes Qdrant (HNSW index, Cosine similarity) to store child-chunk dense vector representations.
  - **Sparse Retrieval**: Utilizes BM25 lexical indexing for exact-keyword queries (e.g., specific ticker acronyms, regulatory terms, financial quantities).
  - **Reciprocal Rank Fusion (RRF)**: Merges sparse and dense search lists to score document relevancy.
  - **Context Expansion**: At retrieval time, the parent chunk text is automatically fetched and provided to the LLM to provide richer context without retrieving redundant small snippets.
- **Robust Model Routing & Graceful Fallbacks**:
  - Per-agent primary and secondary models mapped in a central configuration.
  - Handles rate limits (HTTP 429), timeouts (>10s), and schema validation failures (Pydantic parsing issues) by automatically falling back to secondary providers.

---

## 🛠️ Tech Stack & Core Libraries

- **Agent Framework**: `langgraph`, `langchain`
- **Retrieval Engine & Vector Store**: `qdrant-client`, `rank-bm25`
- **Text Extraction & Chunking**: `chonkie`, `pymupdf` (PyMuPDF), `requests`
- **LLM Integrations**: `openai`, `google-genai`, `langchain-anthropic`, `langchain-google-genai`, `langchain-huggingface`
- **Tracing & Monitoring**: `langfuse`
- **Dependency & Env Management**: `uv`, `python-dotenv`, `pydantic` (v2)

---

## 📋 System Architecture & Routing Table

### LLM Agent Configuration Routing
Each agent is assigned a primary model and a fallback model in `agents/base.py`:

| Agent | Primary Model | Primary Endpoint / Provider | Fallback Model | Fallback Endpoint / Provider |
| :--- | :--- | :--- | :--- | :--- |
| **Supervisor** | `openai/gpt-oss-120b` | OpenRouter | `Qwen/Qwen2.5-72B-Instruct:novita` | HuggingFace Inference API |
| **Metrics** | `meta-llama/Llama-3.1-8B-Instruct:scaleway` | HuggingFace Inference API | `llama-3.1-8b-instant` | Groq |
| **Risk** | `Qwen/Qwen2.5-72B-Instruct:novita` | HuggingFace Inference API | `meta-llama/llama-3.3-70b-instruct:free` | OpenRouter |
| **News** | `gemini-3-flash-preview` | Google AI | `gemini-2.5-flash` | Google AI |
| **Synthesis** | `gpt-oss-120b` | Cerebras | `qwen/qwen3-32b` | Groq |

### Embedding Model Fallback Chain
Embeddings are computed using inference endpoints first to avoid local memory footprints. The system steps through the list until one API key validates:

1. **Voyage voyage-4** (1024 dims)
2. **BAAI/bge-m3** (1024 dims via HF Inference API)
3. **Qwen3-Embedding-0.6B** (1024 dims via local sentence-transformers fallback)
4. **nomic-embed-text-v1.5** (768 dims via Nomic API)
5. **OpenAI text-embedding-3-large** (1024 dims)
6. **OpenAI text-embedding-3-small** (1024 dims)

---

## 📂 Project Directory Structure

```
├── agents/                  # Specialist agent implementations
│   ├── base.py              # LLM factory, fallback handling, and ReAct loop execution
│   ├── metrics_agent.py     # Metrics analyst agent code
│   ├── news_agent.py        # Real-time news and sentiment analyst
│   ├── risk_agent.py        # SEC filing risk assessor agent
│   └── synthesis_agent.py   # Synthesis thesis compiler agent
├── retrieval/               # Hybrid retrieval engines
│   ├── embedding.py         # Embedding fallback chain initialization
│   ├── bm25_retriever.py    # BM25 sparse indexer & searcher
│   ├── qdrant_store.py      # Qdrant client vector store wrappers
│   └── hybrid_retriever.py  # RRF and parent context expansion merger
├── chunking/                # Hierarchical text parser
│   └── chunker.py           # Recursive & Sentence Chonkie configurations
├── ingestion/               # SEC filing loaders
│   ├── pdf_reader.py        # PyMuPDF parser for PDF data
│   └── jina_reader.py       # Jina Reader API client for URL text extraction
├── tools/                   # Executable tools for agents
│   ├── financial.py         # Tabular data extractor
│   ├── retriever.py         # In-filing search retriever wrapper
│   ├── risk.py              # Risk categorizer tool
│   ├── search.py            # Tavily Search API wrapper
│   ├── sentiment.py         # Text sentiment scoring tool
│   └── state_reader.py      # Inter-agent output data fetchers
├── graph/                   # LangGraph topology
│   ├── state.py             # ResearchState attributes schema
│   └── workflow.py          # Node logic and workflow compiler
├── schemas/                 # Strict validation interfaces
│   ├── agents.py            # Supervisor decisions & Agent response Pydantic models
│   ├── chunks.py            # Chunk hierarchies metadata validation
│   ├── retrieval.py         # Retrieved collections definitions
│   └── ingestion.py         # Ingestion document structures
├── test/                    # Suite of execution validation files
├── config.py                # Environment check & Pydantic config validation
├── callbacks.py             # Langfuse tracing middleware factory
└── main.py                  # Scaffolding diagnostic check script
```

---

## ⚙️ Installation & Setup

Ensure `uv` is installed on your system.

### 1. Synchronize Dependencies
This creates a local `.venv` environment and installs all dependencies specified in `pyproject.toml` and `uv.lock`.
```bash
uv sync
```

### 2. Configure Credentials
Duplicate `.env.example` (if present) or create a new `.env` file in the project root:
```env
# Langfuse Tracing
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com

# LLM Providers
OPENROUTER_API_KEY=your-openrouter-key
HF_TOKEN=your-huggingface-token
JINA_API_KEY=jina_...
GOOGLE_API_KEY=AIzaSy...
GROQ_API_KEY=gsk_...
CEREBRAS_API_KEY=csk-sec...
TAVILY_API_KEY=tvly-...

# Vector Store
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-qdrant-api-key # Optional
```

---

## ⚡ Execution Commands

### Diagnostic Verification
Ensure all required environment keys, Langfuse callback handler initializations, and package imports resolve correctly before starting agent runs:
```bash
uv run python main.py
```

### Running Embedding Tests
Test the fallback chain sequence and dense search upsertions into Qdrant using the HNSW index configurations:
```bash
uv run python test/embedding_test.py
uv run python test/embedding_test_2.py
```

### Running Retrieval Verification
Verify the BM25 sparse indexer, dense vector search, and RRF (Reciprocal Rank Fusion) hybrid outputs:
```bash
uv run python test/retrieval_test.py
```

---

## 📈 Detailed Architecture Workflow

```
               [ User Input Request ]
                         │
                         ▼
             [ supervisor_node (LLM) ]
                         │
         ┌───────────────┼───────────────┐
         │ (Parallel asyncio)            │
         ▼                               ▼
 [ metrics_agent ]               [ risk_agent ]                  [ news_agent ]
  ├── RAG Retriever               ├── RAG Retriever               ├── Tavily Web Search
  └── Table Extractor             └── Risk Classifier             └── Sentiment Scorer
         │                               │                               │
         └───────────────┬───────────────┘                               │
                         ▼                                               ▼
                         ├───────────────────────────────────────────────┘
                         ▼
             [ synthesis_node (LLM) ] ──> [ Unified Recommendation / Thesis ]
```

1. **User Request**: The system receives a query regarding a specific stock ticker and target context (e.g., *"Analyze Apple's financial performance and supply chain risks in FY24"*).
2. **Supervisor Routing**: The supervisor node selects which of the specialized agents need to execute and formulates optimized search queries for each.
3. **Concurrent Processing**: The chosen agents start up in parallel. They autonomously query the vector databases, scan web news, format financial charts, and score sentiment.
4. **Synthesis Compilation**: Once all agents complete their ReAct cycles, the synthesis agent fetches their structured outputs, reads any additional cross-filing context as necessary, and outputs an actionable recommendation.
