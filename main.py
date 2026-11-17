from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import uuid

app = FastAPI(
    title="DocuQuery RAG API",
    description="Semantic document retrieval and context-grounded QA backend",
    version="1.0.0"
)

# Initialize local vector store and lightweight embedding model
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="knowledge_base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


class DocumentInput(BaseModel):
    title: str
    content: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 2


def chunk_text(text: str, chunk_size: int = 300) -> list[str]:
    """Splits long documents into smaller overlapping chunks for better vector retrieval."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 50):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


@app.get("/")
def health_check():
    return {"status": "online", "service": "DocuQuery RAG Pipeline"}


@app.post("/ingest")
def ingest_document(doc: DocumentInput):
    """Chunks text, generates dense vector embeddings, and stores them in ChromaDB."""
    chunks = chunk_text(doc.content)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document content cannot be empty.")

    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{doc.title}_{uuid.uuid4().hex[:6]}" for _ in chunks]
    metadatas = [{"source": doc.title} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    return {
        "message": "Document indexed successfully",
        "title": doc.title,
        "chunks_indexed": len(chunks)
    }


@app.post("/query")
def retrieve_context(req: QueryRequest):
    """Performs semantic similarity search to retrieve relevant context for LLM prompting."""
    query_embedding = embedder.encode([req.question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=req.top_k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        raise HTTPException(status_code=404, detail="No relevant context found in knowledge base.")

    context_blocks = [
        {"source": meta["source"], "matched_text": doc}
        for doc, meta in zip(documents, metadatas)
    ]

    return {
        "question": req.question,
        "retrieved_context": context_blocks,
        "llm_prompt_ready": f"Context: {' '.join(documents)}\nQuestion: {req.question}"
    }