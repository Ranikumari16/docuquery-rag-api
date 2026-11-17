# DocuQuery — Semantic RAG Retrieval Backend

A lightweight Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, **ChromaDB**, and **Sentence-Transformers** to index unstructured documents and retrieve semantic context for LLM applications.

## Features
- **Automated Text Chunking:** Splits raw documents into overlapping word chunks to preserve semantic context.
- **Vector Embeddings:** Encodes text chunks into dense vectors using HuggingFace's `all-MiniLM-L6-v2` transformer model.
- **Semantic Search API:** Exposes RESTful `/ingest` and `/query` endpoints with automatic Swagger UI documentation (`/docs`).

## Tech Stack
- **Backend:** Python, FastAPI, Uvicorn, Pydantic
- **AI / Vector Store:** ChromaDB, Sentence-Transformers

## Quick Start
```bash
pip install -r requirements.txt
uvicorn main:app --reload