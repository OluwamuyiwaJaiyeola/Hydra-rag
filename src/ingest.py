import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, HUGGINGFACE_MODEL

def load_regulations(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def generate_embeddings(chunks: list, model: SentenceTransformer) -> list:
    return model.encode(chunks, show_progress_bar=True).tolist()

def upsert_to_pinecone(index, regulation: dict, chunks: list, embeddings: list):
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"reg_{regulation['regulation_id']}_chunk_{i}"
        metadata = {
            "regulation_id": regulation["regulation_id"],
            "title": regulation["title"],
            "jurisdiction": regulation["jurisdiction"],
            "category": regulation["category"],
            "chunk_text": chunk
        }
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        })
    index.upsert(vectors=vectors)
    print(f"Upserted {len(vectors)} chunks for: {regulation['title']}")

def main():
    print("Loading regulations...")
    regulations = load_regulations("data/hydra_regulations.json")
    print(f"Loaded {len(regulations)} regulations")

    print(f"Loading embedding model: {HUGGINGFACE_MODEL}")
    model = SentenceTransformer(HUGGINGFACE_MODEL)

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    print("Starting ingestion...")
    total_chunks = 0
    for regulation in regulations:
        chunks = chunk_text(regulation["full_text"])
        embeddings = generate_embeddings(chunks, model)
        upsert_to_pinecone(index, regulation, chunks, embeddings)
        total_chunks += len(chunks)
        time.sleep(0.5)

    print(f"\nIngestion complete.")
    print(f"Total regulations processed: {len(regulations)}")
    print(f"Total chunks upserted: {total_chunks}")

    stats = index.describe_index_stats()
    print(f"Pinecone index vector count: {stats['total_vector_count']}")

if __name__ == "__main__":
    main()