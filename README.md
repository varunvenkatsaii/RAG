# 🔍 Hybrid RAG Pipeline
### Recursive Chunking | Hybrid Retrieval | RRF | Cross-Encoder Reranking | FastAPI

An end-to-end, framework-agnostic **Retrieval-Augmented Generation (RAG) pipeline** designed to retrieve highly relevant context from multiple document formats.

The system supports document ingestion, metadata extraction, recursive chunking, hybrid retrieval, and neural reranking through a modular architecture. It combines dense vector search with sparse keyword retrieval to capture both semantic relationships and exact keyword matches.

Built with Python, ChromaDB, Hugging Face, and FastAPI, the pipeline exposes retrieval capabilities through an API that can be integrated with any LLM, chatbot, or agentic workflow.

**The core idea:** Combine the semantic understanding of dense retrieval with the precision of keyword search, then use a Cross-Encoder to identify the most relevant context.

---

## ✨ Key Features

### 1. Unified Document Ingestion
A common parsing and metadata extraction interface for multiple document formats.

- PDF documents
- Microsoft Word (DOCX)
- PowerPoint (PPT)
- Plain text (TXT)
- Structured metadata extraction for source tracking and retrieval.

### 2. Recursive Text Chunking
Uses LangChain's `RecursiveCharacterTextSplitter` to divide documents into manageable chunks while preserving meaningful text boundaries.

- Prioritizes paragraph and sentence boundaries before smaller text segments.
- Configurable chunk size and overlap.
- Maintains document metadata across generated chunks.
- Reduces unnecessary fragmentation of contextual information.

### 3. Hybrid Retrieval
Combines two complementary retrieval strategies:

**Dense Retrieval – ChromaDB**
- Uses IBM Granite Embedding English R2.
- Captures semantic relationships between queries and document chunks.
- Retrieves context even when query terminology differs from the source document.

**Sparse Retrieval – BM25**
- Uses keyword-based relevance scoring.
- Captures exact terminology, technical terms, and identifiers.
- Complements semantic retrieval by identifying lexical matches.

### 4. Reciprocal Rank Fusion (RRF)
Combines dense and sparse retrieval results using Reciprocal Rank Fusion.

- Merges independently ranked candidate lists.
- Rewards documents appearing prominently across retrieval strategies.
- Avoids the need to normalize scores from different retrieval systems.
- Produces a unified candidate ranking for downstream reranking.

### 5. Cross-Encoder Reranking
Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to evaluate query-document relevance.

Unlike bi-encoder retrieval, the Cross-Encoder jointly processes the query and candidate chunk to generate a relevance score.

- Evaluates the fused candidate set.
- Reorders candidates based on query-specific relevance.
- Filters out less relevant context before returning results.

### 6. Asynchronous Execution
Uses Python's `asyncio` to support asynchronous ingestion and retrieval workflows.

Dense and sparse retrieval operations can execute concurrently, reducing unnecessary sequential processing.

### 7. FastAPI Integration
Exposes the complete retrieval pipeline through REST API endpoints.

- Document ingestion.
- Query-based retrieval.
- Configurable top-k results.
- Source metadata in retrieved context.
- Integration with external LLMs and agent frameworks.

### 8. Framework-Agnostic Design
The retrieval pipeline operates independently of the generation framework.

It can serve as a standalone retrieval layer for LangChain, LangGraph, LlamaIndex, custom agents, or direct LLM API integrations.

---

## 🏗️ System Architecture

The pipeline follows a modular ingestion and retrieval architecture.

### Document Ingestion

```text
       ┌──────────────────────────────┐
       │          Documents           │
       │                              │
       │     PDF | DOCX | PPT | TXT   │
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │  Unified Parser & Metadata   │
       │         Extraction           │
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │      Recursive Chunking      │
       │                              │
       │  Configurable Size & Overlap │
       └──────────────┬───────────────┘
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
    ┌────────────────┐  ┌────────────────┐
    │ Dense Embedding│  │   BM25 Index   │
    │                │  │                │
    │ IBM Granite R2 │  │ Keyword Search │
    └───────┬────────┘  └───────┬────────┘
            │                   │
            ▼                   │
    ┌────────────────┐          │
    │    ChromaDB    │          │
    └───────┬────────┘          │
            │                   │
            └─────────┬─────────┘
                      │
                      ▼
             ┌─────────────────┐
             │   RRF Fusion    │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Cross-Encoder   │
             │   Reranking     │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  Top-K Context  │
             │     Chunks      │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │    FastAPI      │
             │   REST API      │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │   LLM / Agent   │
             └─────────────────┘
```

---

## 🔄 How Retrieval Works

The retrieval pipeline consists of four major stages:

**1. Parallel Candidate Retrieval**

The incoming query is processed by two retrieval systems concurrently:

- ChromaDB performs semantic similarity search using dense embeddings.
- BM25 performs lexical retrieval using keyword relevance.

**2. Reciprocal Rank Fusion**

The two ranked result lists are combined using:

\[
RRF(d)=\sum_{r \in R}\frac{1}{k+rank_r(d)}
\]

Where:

- \(d\) represents a retrieved document chunk.
- \(R\) represents the individual retrieval result lists.
- \(rank_r(d)\) represents the rank of the chunk in a result list.
- \(k\) is a smoothing constant.

Chunks ranked highly across multiple retrieval systems receive stronger combined rankings.

**3. Cross-Encoder Reranking**

The fused candidates are passed through the Cross-Encoder, which evaluates each query-chunk pair jointly and generates relevance scores.

This provides a more query-specific ranking than relying exclusively on embedding similarity.

**4. Context Generation**

The highest-ranked chunks are returned with their associated metadata, providing the downstream LLM or agent with relevant and traceable context.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python | Core implementation |
| Orchestration | LangChain | Retrieval components |
| Parsing | Unified Parser | Multi-format document ingestion |
| Chunking | RecursiveCharacterTextSplitter | Context-aware text splitting |
| Embeddings | IBM Granite Embedding English R2 | Dense vector representations |
| Embedding Runtime | Hugging Face | Local embedding inference |
| Vector Database | ChromaDB | Dense retrieval |
| Sparse Retrieval | BM25 | Keyword-based retrieval |
| Fusion | Reciprocal Rank Fusion | Combining ranked results |
| Reranking | MS MARCO MiniLM L-6-v2 | Query-context relevance scoring |
| Concurrency | Asyncio | Asynchronous execution |
| API | FastAPI | Retrieval service |

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.10+
- Git
- Sufficient disk space to cache Hugging Face models during the first run.

### 1. Clone the Repository

```bash
git clone https://github.com/varunvenkatsaii/RAG.git
cd RAG
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI Server

```bash
uvicorn api:app --reload
```

> Adjust `main:app` according to your actual application entry point.

Once the server starts, access the interactive API documentation:

**http://localhost:8000/docs**


## ⚙️ Configuration

The following parameters can be configured to control chunking and retrieval behavior.

| Parameter | Description | Example |
|---|---|---|
| `CHUNK_SIZE` | Maximum characters per chunk | `1000` |
| `CHUNK_OVERLAP` | Shared characters between adjacent chunks | `200` |
| `TOP_K_DENSE` | Candidates retrieved from ChromaDB | `20` |
| `TOP_K_BM25` | Candidates retrieved from BM25 | `20` |
| `TOP_K_FINAL` | Final chunks returned after reranking | `5` |


> Update this section with the actual repository structure.

---

## 🔮 Roadmap

- [ ] **Retrieval Evaluation Framework** – Benchmark dense, sparse, and hybrid retrieval using Recall@K and MRR.
- [ ] **Persistent BM25 Index** – Support persistence and incremental updates of the sparse retrieval index.
- [ ] **Extended Document Support** – Add more document formats and improve parsing capabilities.
- [ ] **Retrieval Optimization** – Explore query expansion, metadata filtering, and improved candidate selection.
- [ ] **Production Deployment** – Improve API deployment, logging, monitoring, and scalability.

---

## 🎯 Project Objective

Build a reusable retrieval infrastructure that separates context retrieval from LLM generation.

By combining dense retrieval, keyword search, rank fusion, and neural reranking, the system aims to deliver relevant, traceable context to downstream applications without coupling retrieval to a specific LLM or agent framework.

**From document ingestion to intelligent context retrieval.**

---
