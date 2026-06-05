import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from src.config import EMBEDDING_DIM

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hydra-regulations")
HUGGINGFACE_MODEL   = os.getenv("HUGGINGFACE_MODEL", "BAAI/bge-base-en-v1.5")
INDEX_RECORDS_PATH = Path("data/hydra_index_records.json")


def load_index_records(path: Path) -> list:
    with open(path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} index records from {path}")
    n_chunks = len({r['chunk_id'] for r in data})
    logger.info(f"  distinct clauses (chunk_ids): {n_chunks}")
    return data


def generate_embeddings(texts: list, model: SentenceTransformer) -> list:
    """
    Passages are embedded as plain text (no prefix). BGE only requires the
    instruction prefix on the QUERY side, which is applied in rag_chain.py.
    """
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32
    )
    return embeddings.tolist()


def build_vectors(records: list, model: SentenceTransformer) -> list:
    texts = [r["text"] for r in records]
    embeddings = generate_embeddings(texts, model)

    vectors = []
    for r, emb in zip(records, embeddings):
        vectors.append({
            "id": r["vector_id"],
            "values": emb,
            "metadata": {
                "chunk_id":      r["chunk_id"],
                "regulation_id": float(r["regulation_id"]),
                "title":         r["title"],
                "jurisdiction":  r["jurisdiction"],
                "category":      r["category"],
                "family":        r["family"],
                "article_ref":   r["article_ref"],
                "scope":         r["scope"],
                "chunk_text":    r["text"][:500],
            }
        })
    logger.info(f"Total vectors built: {len(vectors)}")
    return vectors


def upsert_to_pinecone(vectors: list, index) -> None:
    batch_size = 100
    total = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        total += len(batch)
        logger.info(f"Upserted {total}/{len(vectors)} vectors")
    logger.info("Upsert complete.")


def main():
    logger.info("=== Hydra Analytics: Data Ingestion Pipeline ===")

    logger.info(f"Loading embedding model: {HUGGINGFACE_MODEL}")
    model = SentenceTransformer(HUGGINGFACE_MODEL)

    records = load_index_records(INDEX_RECORDS_PATH)

    logger.info("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(f"Creating index: {PINECONE_INDEX_NAME} (dim={EMBEDDING_DIM})")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        logger.info(f"Index exists: {PINECONE_INDEX_NAME}")

    index = pc.Index(PINECONE_INDEX_NAME)

    logger.info("Checking index before clear...")
    stats = index.describe_index_stats()
    if stats["total_vector_count"] > 0:
        index.delete(delete_all=True)
        logger.info("Index cleared.")
    else:
        logger.info("Index is empty, skipping clear.")

    logger.info("Building vectors...")
    vectors = build_vectors(records, model)

    logger.info("Upserting to Pinecone...")
    upsert_to_pinecone(vectors, index)

    stats = index.describe_index_stats()
    total_vectors = stats["total_vector_count"]
    logger.info(f"=== Ingestion complete. Total vectors: {total_vectors} ===")

    expected = len(records)
    if total_vectors == expected:
        logger.info(f"Vector count matches expected: {expected}")
    else:
        logger.warning(f"Vector count mismatch. Expected {expected}, got {total_vectors}")


if __name__ == "__main__":
    main()