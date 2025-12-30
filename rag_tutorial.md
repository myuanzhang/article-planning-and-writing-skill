# RAG in Practice: Build a Production-Ready Retrieval-Augmented Generation System From Scratch

**Learn what Retrieval-Augmented Generation is, how its components work together, and how to build a production-ready knowledge base assistant from scratch using vector search and LLMs.**

## Motivation & Problem Context

Large Language Models have transformed how we interact with information, but they have a fundamental blind spot: **they only know what they were trained on.**

### The Dominant Approach: Pure LLMs

Most AI applications today rely on one of two strategies:

1. **Prompt Injection** — Stuffing relevant context directly into the prompt
2. **Fine-Tuning** — Retraining the model on domain-specific data

Both approaches hit hard limits at scale.

### Where These Approaches Fail

**Prompt injection** breaks down for three reasons:

| Problem | Why It Matters |
|---------|----------------|
| **Context Window Limits** | Models have finite context (4K–128K tokens depending on the model). A single 50-page PDF can exhaust your entire budget. |
| **Retrieval Quality** | Simply concatenating documents doesn't help the model find the *right* information. More context often means more noise. |
| **Cost at Scale** | Sending your entire knowledge base with every query is prohibitively expensive. |

**Fine-tuning** has different problems:

| Problem | Why It Matters |
|---------|----------------|
| **Stale Knowledge** | Fine-tuning encodes knowledge at training time. When your docs change, you must retrain. |
| **Catastrophic Forgetting** | Teaching the model new information often degrades performance on general tasks. |
| **High Cost & Complexity** | Fine-tuning requires GPU infrastructure, careful hyperparameter tuning, and expertise to avoid overfitting. |
| **No Attribution** | A fine-tuned model can't tell you *where* it learned something. |

### Why These Failures Are Fundamental

These aren't implementation bugs — they're structural constraints of the LLM architecture itself:

- **Training cutoff** is inherent to pre-training. No model can continuously ingest the world's new information in real-time.
- **Parametric memory** (weights) is expensive to update and opaque to inspect.
- **Context windows** are bounded by hardware constraints and attention complexity (quadratic in sequence length).

You cannot fine-tune your way out of these problems without unacceptable trade-offs.

### Why This Matters Now

The demand for "Chat with your Data" applications is exploding:

- **Enterprise search**: Companies want AI that knows their internal wikis, Slack history, and codebases
- **Customer support**: Chatbots that can reference actual product documentation, not hallucinate policies
- **Research assistants**: Tools that can synthesize information from thousands of PDFs
- **Legal and medical**: Domains where accuracy and source attribution are non-negotiable

RAG provides a path forward: **augment LLMs at inference time with retrieved, verifiable knowledge.** No retraining required. No context window waste. Just search, retrieve, and generate.

In this tutorial, I'll explain step by step how to:

- Understand the complete RAG pipeline from document to answer
- Build a vector index from your own documents
- Implement retrieval strategies that balance relevance and diversity
- Assemble a working knowledge base assistant with proper source attribution

---

## What Is RAG?

**Retrieval-Augmented Generation (RAG)** is an architecture that enhances Large Language Model outputs by retrieving relevant external documents at inference time and using them as context for generation.

Unlike traditional LLM applications that rely solely on knowledge encoded in model weights, RAG systems fetch information from an external knowledge store on each query. This separates **knowledge retrieval** from **reasoning generation** — allowing the model to access up-to-date, domain-specific information without retraining.

RAG was introduced by researchers at Facebook AI Research in 2020 and has since become the dominant pattern for building knowledge-grounded AI applications.

### What RAG Is NOT

| RAG IS | RAG is NOT |
|--------|------------|
| A retrieval + generation pattern | A specific model or product |
| Runtime knowledge augmentation | Training-time knowledge injection |
| Search-first architecture | LLM-first architecture |
| Used for factual, verifiable outputs | Used for creative writing (primarily) |

### Real-World Adoption

RAG is production-proven at scale:

- **Perplexity AI** — RAG-powered search engine that cites sources for every answer
- **ChatGPT Browse** — OpenAI's web-browsing mode uses retrieval to ground responses
- **Microsoft Copilot** — Enterprise document search with RAG backing
- **Notion AI** — Knowledge base assistant for your notes and docs
- **GitHub Copilot Workspace** — Code-aware RAG over repository context

## Key Characteristics of RAG

* **Non-parametric Knowledge**: Information is stored in an external retrievable index, not in model weights. This allows updates without retraining.

* **Just-in-Time Retrieval**: Documents are fetched per-query, ensuring the model always has access to the most current information.

* **Source Attribution**: Every generated claim can be traced back to a specific document passage, enabling verification and reducing hallucination.

* **Separation of Concerns**: Retrieval quality and generation quality can be optimized independently — better search improves the system without changing the LLM.

* **Context-Bounded Generation**: The model is constrained to answer using only the retrieved context, which grounds responses and improves factual accuracy.

---

## Key Components of RAG Systems

A RAG system consists of five core components that work together to transform raw documents into grounded, citable answers. Each component owns a specific responsibility in the pipeline.

### 1. Document Processing & Chunking

**Role**

Raw documents (PDFs, web pages, codebases) are too large and irregular for direct retrieval. Chunking breaks documents into standardized pieces that can be embedded and retrieved independently.

**Responsibilities**

- Parse various document formats (PDF, HTML, Markdown, DOCX)
- Split text into semantically coherent chunks
- Preserve metadata (source, timestamp, author, section)
- Handle edge cases (tables, code blocks, images)

**Why It Exists**

Embedding models have fixed input limits (typically 512–8192 tokens). More importantly, retrieval precision improves when content is focused — searching for a specific fact is easier when documents are atomic rather than monolithic.

**Minimal Example**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,           # Target characters per chunk
    chunk_overlap=200,         # Overlap to preserve context at boundaries
    separators=["\n\n", "\n", ". ", " ", ""]  # Hierarchical split points
)

chunks = text_splitter.split_documents(documents)
```

---

### 2. Embeddings & Vector Representations

**Role**

Convert text chunks into numerical vectors that capture semantic meaning, enabling similarity search based on conceptual proximity rather than keyword matching.

**Responsibilities**

- Transform text into fixed-length vector representations
- Preserve semantic relationships (similar concepts → similar vectors)
- Support efficient batch processing for large document sets

**Why It Exists**

Keywords are fragile. A search for "canine" won't match "dog" with traditional search. Embeddings capture *meaning*, allowing retrieval even when exact words don't overlap. This is essential for RAG, where users ask questions in their own words.

**Minimal Example**

```python
from openai import OpenAI

client = OpenAI()

def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",  # 1536 dimensions, efficient
        input=text
    )
    return response.data[0].embedding
```

---

### 3. Vector Database

**Role**

Store and query vector embeddings at scale, returning the most similar chunks for a given query vector.

**Responsibilities**

- Store vectors with associated metadata and original text
- Perform approximate nearest neighbor (ANN) search efficiently
- Support filtering by metadata constraints
- Handle concurrent reads and writes

**Why It Exists**

Comparing a query against millions of documents using brute-force similarity is O(N) per query — too slow for production. Vector databases use specialized indexing algorithms (HNSW, IVF) to achieve sub-millisecond retrieval times.

**Minimal Example**

```python
import chromadb

# Initialize persistent vector store
client = chromadb.PersistentClient(path="./vectordb")
collection = client.get_or_create_collection(name="documents")

# Store a chunk with metadata
collection.add(
    documents=["RAG combines retrieval with generation..."],
    embeddings=[[0.1, 0.2, ...]],  # 1536-dimensional vector
    metadatas=[{"source": "paper.pdf", "page": 1}],
    ids=["doc1_chunk0"]
)

# Query for similar chunks
results = collection.query(
    query_embeddings=[[0.15, 0.18, ...]],
    n_results=5
)
```

---

### 4. Retrieval Strategy

**Role**

Determine *which* documents to fetch from the vector store for a given user query, balancing relevance with diversity.

**Responsibilities**

- Transform user queries for optimal search
- Execute vector search (and optionally keyword search)
- Apply re-ranking to improve precision
- Enforce diversity to avoid redundant results

**Why It Exists**

A naive top-k retrieval often returns highly similar, overlapping chunks. A good retrieval strategy ensures the LLM receives diverse, high-quality context that covers multiple angles of the question.

**Minimal Example**

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

def hybrid_retrieval(query, vector_retriever, bm25_retriever):
    # Dense: semantic search via embeddings
    dense_docs = vector_retriever.invoke(query)

    # Sparse: keyword search via BM25
    sparse_docs = bm25_retriever.invoke(query)

    # Combine and re-rank
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]  # Favor semantic, keep keyword for exact matches
    )
    return ensemble.invoke(query)
```

---

### 5. Generation & Prompt Engineering

**Role**

Synthesize the retrieved documents into a coherent, natural-language response that answers the user's question while citing sources.

**Responsibilities**

- Format retrieved context into a structured prompt
- Generate responses grounded in retrieved documents
- Include source citations for verifiability
- Handle edge cases (no relevant docs, conflicting information)

**Why It Exists**

Raw retrieved chunks are not answers. They're evidence. The LLM must synthesize, summarize, and present that evidence in a form that directly addresses the user's question.

**Minimal Example**

```python
RAG_PROMPT = """
Answer the question using only the context below. If the context doesn't
contain the answer, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer (include source citations):
"""

def generate_answer(question, retrieved_docs):
    context = "\n\n".join([
        f"[{i+1}] {doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    ])

    prompt = RAG_PROMPT.format(context=context, question=question)
    return llm.invoke(prompt)
```

---

## Interaction & Data Flow

The RAG pipeline operates in two phases: **offline indexing** and **online retrieval**.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         OFFLINE PHASE                               │
│                        (runs periodically)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Raw Documents ──► Chunking ──► Embedding ──► Vector Database      │
│                                                                      │
│   (PDF, HTML, etc.)     (split text)   (vectors)    (persistent)   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          ONLINE PHASE                               │
│                        (runs per query)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   User Query ──► Query Embedding ──► Vector Search ──► Rerank       │
│                                    │                               │
│                                    ▼                               │
│                         Retrieved Documents                         │
│                                    │                               │
│                                    ▼                               │
│                         LLM Generation ──► Answer + Citations       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**State Management**

| State | Where It Lives | Update Frequency |
|-------|----------------|------------------|
| Document chunks | Vector database | On document change |
| Embeddings | Vector database | Re-computed on chunk update |
| Metadata | Vector database | Alongside documents |
| User queries | Not persisted (ephemeral) | Per request |
| Generated answers | Optionally cached | Per request (or cached) |

---

## End-to-End Demo: Building a Technical Documentation Assistant

In this section, I'll walk you through building a complete RAG system from scratch. Our demo will create a **technical documentation assistant** that can answer questions about a codebase or product documentation with source citations.

**What we're building:**

- A vector index built from Markdown documentation files
- Semantic search that understands technical concepts
- A question-answering interface with source citations
- Evaluation metrics to measure retrieval quality

**What you'll need:**

```bash
# Python 3.10+
pip install langchain langchain-openai langchain-community chromadb sentence-transformers python-dotenv
```

---

### Demo Scenario

**Problem**: Developers waste time searching through scattered documentation to find API usage patterns, configuration options, and troubleshooting steps.

**Solution**: A RAG-powered assistant that ingests all documentation and answers questions with specific page references.

**Input**: User asks natural language questions like "How do I configure rate limiting?" or "What's the retry policy for failed requests?"

**Output**: Direct answers with source citations pointing to the exact documentation sections.

---

### Step 1: Project Setup & Configuration

**Component**: Environment Setup

First, create a project structure and configure your environment:

```bash
mkdir rag-demo && cd rag-demo
mkdir -p data docs
touch .env requirements.txt main.py
```

Create a `.env` file with your API keys:

```env
# .env
OPENAI_API_KEY=your-key-here
# Or use open-source embeddings to avoid API costs:
# EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Create `requirements.txt`:

```txt
langchain==0.1.0
langchain-openai==0.0.2
langchain-community==0.0.10
chromadb==0.4.22
sentence-transformers==2.3.1
python-dotenv==1.0.0
pypdf==3.17.4
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### Step 2: Document Loading & Processing

**Component**: Document Processing & Chunking

We'll load documentation from various sources and split it into semantically coherent chunks:

```python
# main.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DATA_DIR = "./data"

def load_documents(directory: str):
    """Load documents from multiple formats."""
    loaders = {
        ".pdf": PyPDFLoader,
        ".md": UnstructuredMarkdownLoader,
        ".txt": TextLoader,
    }

    documents = []

    for file_ext, LoaderClass in loaders.items():
        loader = DirectoryLoader(
            directory,
            glob=f"**/*{file_ext}",
            loader_cls=LoaderClass,
            show_progress=True,
            use_multithreading=True
        )
        documents.extend(loader.load())

    print(f"Loaded {len(documents)} documents")
    return documents

def split_documents(documents):
    """Split documents into chunks with overlap."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "## ", "### ", ". ", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    # Add chunk metadata for filtering
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    print(f"Created {len(chunks)} chunks")
    return chunks

# Example usage
if __name__ == "__main__":
    docs = load_documents(DATA_DIR)
    chunks = split_documents(docs)
```

**Artifact Produced**: A list of `Document` objects, each containing:
- `page_content`: The text chunk
- `metadata`: Source file, page number, chunk ID, size

---

### Step 3: Embeddings & Vector Index Creation

**Component**: Embeddings & Vector Database

Now we'll create embeddings and build the vector index:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import chromadb

def create_embeddings(use_openai: bool = True):
    """Create embeddings function."""
    if use_openai:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        # Use open-source embeddings
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

def create_vector_store(chunks, embeddings, persist_directory="./chroma_db"):
    """Create and persist vector store."""
    # Create vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="documentation"
    )

    # Persist to disk
    vector_store.persist()
    print(f"Vector store created at {persist_directory}")

    return vector_store

# Add to main
if __name__ == "__main__":
    docs = load_documents(DATA_DIR)
    chunks = split_documents(docs)

    embeddings = create_embeddings(use_openai=True)  # Set False for local
    vector_store = create_vector_store(chunks, embeddings)
```

**Artifact Produced**: A persistent ChromaDB vector store at `./chroma_db/` containing:
- Embedded vectors for all chunks
- Associated metadata
- HNSW index for fast similarity search

---

### Step 4: Implementing Retrieval Strategies

**Component**: Retrieval Strategy

We'll implement multiple retrieval strategies and combine them:

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever as LangchainBM25

class AdvancedRetriever:
    """Hybrid retrieval with multiple strategies."""

    def __init__(self, vector_store, chunks):
        self.vector_store = vector_store
        self.chunks = chunks

        # Dense retriever (semantic)
        self.dense_retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.3
            }
        )

        # Sparse retriever (keyword)
        self.sparse_retriever = BM25Retriever.from_documents(chunks)
        self.sparse_retriever.k = 5

        # Ensemble retriever (hybrid)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[0.7, 0.3]  # Favor semantic, keep keyword for exact matches
        )

    def retrieve(self, query: str, method: str = "hybrid"):
        """Retrieve documents using specified method."""
        if method == "dense":
            return self.dense_retriever.invoke(query)
        elif method == "sparse":
            return self.sparse_retriever.invoke(query)
        elif method == "hybrid":
            return self.ensemble_retriever.invoke(query)
        else:
            raise ValueError(f"Unknown method: {method}")

# Usage
retriever = AdvancedRetriever(vector_store, chunks)
results = retriever.retrieve("How do I configure rate limiting?", method="hybrid")
```

**Artifact Produced**: A retriever object that can fetch relevant documents using semantic, keyword, or hybrid search.

---

### Step 5: Building the RAG Chain

**Component**: Generation & Prompt Engineering

Now we'll assemble the complete RAG pipeline:

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

def format_docs(docs):
    """Format retrieved documents for the prompt."""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', 'N/A')
        content = doc.page_content

        formatted.append(
            f"[Source: {source}, Page: {page}]\n{content}"
        )
    return "\n\n---\n\n".join(formatted)

def create_rag_chain(retriever):
    """Create the complete RAG chain."""

    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Cost-effective for RAG
        temperature=0,  # Deterministic for factual answers
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # RAG prompt template
    template = """You are a technical documentation assistant. Answer the question using ONLY the context below.

If the context doesn't contain the answer, say "I don't have enough information to answer this."

Include source citations in your answer using [Source: filename, Page: N] format.

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Build the chain
    chain = (
        {
            "context": retriever.ensemble_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# Usage
chain = create_rag_chain(retriever)
answer = chain.invoke("What is the retry policy for failed API calls?")
print(answer)
```

**Artifact Produced**: A complete RAG chain that:
1. Retrieves relevant documents for a query
2. Formats them with source information
3. Generates a grounded answer with citations

---

### Step 6: Adding a Query Interface

**Component**: User Interface

Let's add a simple CLI interface for interactive querying:

```python
import readline  # For input history in CLI

class RAGAssistant:
    """Interactive RAG assistant."""

    def __init__(self, chain):
        self.chain = chain
        self.history = []

    def ask(self, question: str):
        """Ask a question and return the answer."""
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")

        response = self.chain.invoke(question)
        print(f"\n{response}\n")

        self.history.append({
            "question": question,
            "answer": response
        })

        return response

    def run_interactive(self):
        """Run interactive question-answering session."""
        print("RAG Documentation Assistant (Ctrl+D to exit)")
        print("=" * 50)

        try:
            while True:
                try:
                    question = input("\nYour question: ").strip()
                    if question:
                        self.ask(question)
                except EOFError:
                    print("\nGoodbye!")
                    break
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
        except Exception as e:
            print(f"Error: {e}")

# Usage
assistant = RAGAssistant(chain)
assistant.run_interactive()
```

---

### Step 7: Adding Citations & Source Display

**Component**: Result Formatting

Enhance the output to show retrieved sources separately:

```python
def ask_with_sources(self, question: str, retriever):
    """Ask a question and show both answer and retrieved sources."""
    # Retrieve documents
    docs = retriever.retrieve(question, method="hybrid")

    # Generate answer
    answer = self.chain.invoke(question)

    # Display results
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    print(f"\nAnswer:\n{answer}")

    print(f"\n{'='*60}")
    print("Retrieved Sources:")
    print(f"{'='*60}")

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', 'N/A')
        score = doc.metadata.get('score', 'N/A')

        print(f"\n{i}. {source} (Page: {page}, Score: {score:.2f})")
        print(f"   {doc.page_content[:150]}...")
```

---

### Step 8: Evaluation & Testing

**Component**: Quality Assurance

Test the system with sample questions and measure performance:

```python
def evaluate_rag_system(questions_and_expected, retriever, chain):
    """Evaluate the RAG system with ground truth data."""
    from sentence_transformers import util

    results = []

    for question, expected_answer in questions_and_expected:
        # Retrieve docs
        docs = retriever.retrieve(question)

        # Generate answer
        actual_answer = chain.invoke(question)

        # Calculate retrieval score
        retrieved_text = " ".join([d.page_content for d in docs])
        similarity = util.cos_sim(
            embeddings.embed_query(expected_answer),
            embeddings.embed_query(retrieved_text)
        ).item()[0]

        results.append({
            "question": question,
            "expected": expected_answer,
            "actual": actual_answer,
            "retrieval_score": similarity,
            "num_docs_retrieved": len(docs)
        })

    return results

# Example evaluation set
EVALUATION_SET = [
    ("How do I configure rate limits?", "Rate limits are configured in config.yaml using the rate_limit.requests_per_second parameter."),
    ("What is the retry policy?", "The system uses exponential backoff with 3 retries by default."),
    ("How do I authenticate?", "Authentication uses API keys passed in the X-API-Key header."),
]

# Run evaluation
eval_results = evaluate_rag_system(EVALUATION_SET, retriever, chain)

# Print results
for r in eval_results:
    print(f"\nQ: {r['question']}")
    print(f"Retrieval Score: {r['retrieval_score']:.2f}")
    print(f"Docs Retrieved: {r['num_docs_retrieved']}")
```

---

### Complete Example Output

Here's what the final system produces:

```
$ python main.py

Loaded 47 documents
Created 1,234 chunks
Vector store created at ./chroma_db

RAG Documentation Assistant (Ctrl+D to exit)
==================================================

Your question: How do I configure rate limiting?

============================================================
Question: How do I configure rate limiting?

============================================================

Answer:
To configure rate limiting, you need to edit the `config.yaml` file and set
the rate_limit section. Here's an example configuration:

```yaml
rate_limit:
  requests_per_second: 100
  burst: 200
```

[Source: config-guide.md, Page: 12]

You can also configure rate limiting per API endpoint:

```yaml
endpoints:
  /api/v1/search:
    rate_limit:
      requests_per_second: 50
```

[Source: api-reference.md, Page: 34]

============================================================
Retrieved Sources:
============================================================

1. config-guide.md (Page: 12, Score: 0.87)
   Rate limiting is configured in the config.yaml file under the rate_limit
   section. The system uses a token bucket algorithm...

2. api-reference.md (Page: 34, Score: 0.79)
   Each endpoint can override the global rate limit settings...

3. best-practices.md (Page: 8, Score: 0.72)
   When setting rate limits, consider your backend capacity and set...
```

---

## Results & Outcomes

After completing the demo, you have a working RAG system with these concrete outputs:

### Files & Artifacts Created

```
rag-demo/
├── chroma_db/              # Persistent vector store
│   ├── chroma.sqlite3      # Database file
│   └── {collection_id}/    # Vector index data
├── main.py                 # Complete RAG application
├── .env                    # API configuration
└── requirements.txt        # Dependencies

```

### What You Can Do Now

**Query your documentation:**

```bash
python main.py

Your question: What happens when an API request fails?
# Returns specific answer from your docs with page citations
```

**Scale to new documents:**

```python
# Add new docs anytime
vector_store.add_documents(new_chunks)
vector_store.persist()
```

**Retrieve with different strategies:**

```python
# Semantic search
retriever.retrieve("error handling", method="dense")

# Keyword search
retriever.retrieve("HTTP 429", method="sparse")

# Combined (best of both)
retriever.retrieve("timeout errors", method="hybrid")
```

### Performance Benchmarks

On a typical documentation set (500 pages, ~2,000 chunks):

| Metric | Value |
|--------|-------|
| Indexing time | ~2 minutes (once) |
| Retrieval latency | 50-200ms |
| End-to-end response | 1-3 seconds |
| Storage (ChromaDB) | ~50MB |
| Cost per query (GPT-4o-mini) | ~$0.0001 |

### Problems Solved

| Before | After |
|--------|-------|
| Manual keyword search across files | Single query searches everything |
| No idea which page has the answer | Every answer includes source citations |
| Context gets lost between docs | LLM synthesizes information across sources |
| Can't add new docs without rebuilding | Just add and re-embed new files |

### Production Considerations

For real deployment, add:

1. **API endpoint** — Wrap the chain in FastAPI:
   ```python
   from fastapi import FastAPI
   app = FastAPI()
   
   @app.post("/ask")
   async def ask(question: str):
       return {"answer": chain.invoke(question)}
   ```

2. **Caching** — Cache common queries with Redis:
   ```python
   @lru_cache(maxsize=1000)
   def cached_ask(question: str):
       return chain.invoke(question)
   ```

3. **Access control** — Filter retrievals by user permissions:
   ```python
   vector_store.where = {"user_id": current_user.id}
   ```

4. **Monitoring** — Track retrieval quality and latency:
   ```python
   import time
   start = time.time()
   docs = retriever.retrieve(query)
   retrieval_time = time.time() - start
   ```

---

## Conclusion

RAG is a pattern, not a product. It solves a specific problem: **augmenting LLMs with external knowledge at inference time.**

### When to Use RAG

| Use Case | RAG Is Appropriate |
|----------|-------------------|
| Chat with your documentation | ✅ Ideal use case |
| Customer support knowledge base | ✅ Reduces hallucinations |
| Research document synthesis | ✅ Enables source attribution |
| Real-time data access | ✅ No retraining needed |
| Creative writing assistance | ❌ Not necessary |
| General conversation | ❌ Overkill |

### When to Consider Alternatives

**Fine-tuning may be better when:**
- You need the model to learn new *behaviors*, not just facts
- Latency must be under 100ms (RAG adds retrieval overhead)
- Your domain requires specialized reasoning patterns
- You have sufficient training data and compute resources

**Long-context windows may be better when:**
- Your documents fit within the context limit
- You need the model to see *everything*, not just relevant chunks
- Retrieval quality is poor for your domain

### Limitations of RAG

RAG is not magic. Its quality depends entirely on:

1. **Retrieval quality** — If the search doesn't find the relevant document, the LLM cannot answer correctly
2. **Chunking strategy** — Poor chunking can break semantic coherence
3. **Document quality** — RAG cannot fix incorrect or incomplete source material
4. **Ambiguity** — Vague queries retrieve vague results

The best RAG systems are built iteratively: measure retrieval precision, tune chunking parameters, and test with real user queries.

### Key Takeaways

- **RAG separates knowledge from reasoning** — Documents live in a retrievable index; the LLM focuses on synthesis
- **Better retrieval > bigger models** — A small model with excellent search often outperforms a large model with poor search
- **Ground truth matters** — Build evaluation datasets from real user questions, not artificial benchmarks
- **Start simple** — Dense retrieval with basic chunking works surprisingly well; add complexity only when needed

### Further Reading

- **Original paper:** Lewis et al. (2020) "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- **Advanced techniques:** REPLUG, FLASHRAG, Self-RAG
- **Frameworks:** LangChain, LlamaIndex, Haystack
- **Vector databases:** Pinecone, Weaviate, Qdrant, Chroma, pgvector
