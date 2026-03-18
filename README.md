# Mini RAG for Construction Marketplace Documents

This project implements a simple retrieval-augmented generation pipeline for answering questions from internal construction marketplace documents. It indexes the provided files locally, retrieves the most relevant chunks with FAISS, and generates answers that are constrained to the retrieved context.

## Stack

- Backend: FastAPI
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector search: `FAISS`
- LLM: OpenRouter
- Frontend: custom HTML, CSS, and vanilla JavaScript interface

## Project Structure

```text
app/                  FastAPI app and RAG pipeline
data/raw/             Place the provided source documents here
data/processed/       Saved chunk metadata and FAISS index
frontend/             Custom chatbot UI
scripts/build_index.py
tests/eval_questions.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set:
- `OPENROUTER_API_KEY` with your OpenRouter API key
- optionally `OPENROUTER_MODEL` if you want a different hosted model

4. Place the assessment documents into `data/raw/`.
   Supported formats: `.pdf`, `.txt`, `.md`

## Run

Build the local FAISS index:

```bash
python scripts/build_index.py
```

Start the app:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## How It Works

### Document Chunking

The ingestion pipeline reads local files from `data/raw/` and splits them into overlapping word-based chunks. Default settings:

- Chunk size: `900`
- Overlap: `150`

These values can be changed in `.env`.

### Retrieval

Each chunk is embedded with `all-MiniLM-L6-v2`. The embeddings are normalized and stored in a FAISS inner-product index, which acts as cosine similarity search after normalization. At query time, the system embeds the user question and retrieves the top `k` chunks.

### Grounded Answer Generation

The generation prompt explicitly tells the model to:

- answer only from the retrieved context
- avoid outside knowledge
- say when the answer is not present in the documents

The UI always displays the retrieved chunks alongside the final answer so the result is explainable and auditable.

## Recommended Model Choice

- Embedding model: `all-MiniLM-L6-v2`
  Reason: small, fast, and strong enough for short internal policy and FAQ documents.
- LLM: OpenRouter free model
  Reason: simple hosted setup for deployment and no local inference dependency.

## Evaluation

Use `tests/eval_questions.md` as a starting point. After you add the real source documents, test 8 to 15 document-derived questions and record:

- whether the correct chunks were retrieved
- whether the answer contained unsupported claims
- whether the answer was complete and clear

Summarize those findings in this README for the final submission.

## Deployment Notes

The app serves the frontend directly from FastAPI, which keeps deployment simple for the assignment. You can deploy this to any platform that supports a Python web service.

## Current Limitation

The Google Drive assessment files referenced in the PDF are not bundled in this repository. The app is ready to ingest them once they are added to `data/raw/`.
