import os
import json
import re
import logging
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Config
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hydra-regulations")
HUGGINGFACE_MODEL   = os.getenv("HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_PATH           = Path("data/hydra_regulations_cleaned.json")
EMBEDDING_DIM       = 384

# Article boundary pattern
ARTICLE_PATTERN = re.compile(
    r'(?=(?:Article|Section|Clause|Provision|Rule)\s+\d+[\.\d]*)',
    re.IGNORECASE
)


def load_regulations(path: Path) -> list:
    """Load the cleaned regulations JSON file."""
    with open(path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} regulations from {path}")
    return data


def extract_article_ref(chunk: str) -> str:
    """Extract the article reference from the start of a chunk."""
    match = re.match(
        r'^(Article|Section|Clause|Provision|Rule)\s+[\d\.]+',
        chunk.strip(),
        re.IGNORECASE
    )
    return match.group(0).strip() if match else "General provision"


def chunk_regulation(full_text: str) -> list:
    """
    Split regulation text at article boundaries.
    Source file is pre-cleaned so no boilerplate filtering needed here.
    Each chunk represents exactly one legal obligation.
    """
    raw_chunks = re.split(ARTICLE_PATTERN, full_text)
    chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        # Skip empty or very short fragments
        if len(chunk.split()) < 5:
            continue
        chunks.append(chunk)
    return chunks


def generate_embeddings(texts: list, model: SentenceTransformer) -> list:
    """
    Generate normalized embeddings for a list of texts.
    normalize_embeddings=True ensures consistent cosine similarity scores.
    Uses embed_documents approach: encode all chunks together.
    """
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32
    )
    return embeddings.tolist()


def build_vectors(regulations: list, model: SentenceTransformer) -> list:
    """
    Build Pinecone vector records from all regulations.
    Each vector = one article chunk with full metadata.
    """
    vectors = []

    for reg in regulations:
        reg_id     = reg["regulation_id"]
        title      = reg["title"]
        jurisdiction = reg["jurisdiction"]
        category   = reg["category"]
        full_text  = reg["full_text"]

        chunks = chunk_regulation(full_text)

        if not chunks:
            logger.warning(f"Regulation {reg_id} produced no chunks. Check source file.")
            continue

        chunk_texts = [c for c in chunks]
        embeddings  = generate_embeddings(chunk_texts, model)

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            article_ref = extract_article_ref(chunk)
            vector_id   = f"reg-{reg_id}-chunk-{i}"

            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "regulation_id": float(reg_id),
                    "title":         title,
                    "jurisdiction":  jurisdiction,
                    "category":      category,
                    "article_ref":   article_ref,
                    "chunk_text":    chunk[:500],   # store first 500 chars
                    "chunk_index":   i
                }
            })

        logger.info(f"Regulation {reg_id} ({jurisdiction}): {len(chunks)} chunks embedded")

    logger.info(f"Total vectors built: {len(vectors)}")
    return vectors


def upsert_to_pinecone(vectors: list, index) -> None:
    """Upsert vectors to Pinecone in batches of 100."""
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

    # 1. Load embedding model
    logger.info(f"Loading embedding model: {HUGGINGFACE_MODEL}")
    model = SentenceTransformer(HUGGINGFACE_MODEL)

    # 2. Load cleaned regulations
    regulations = load_regulations(DATA_PATH)

    # 3. Connect to Pinecone
    logger.info("Connecting to Pinecone...")
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(f"Creating index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        logger.info(f"Index exists: {PINECONE_INDEX_NAME}")

    index = pc.Index(PINECONE_INDEX_NAME)

    # 4. Clear existing vectors before re-ingestion
    logger.info("Clearing existing vectors from index...")
    index.delete(delete_all=True)
    logger.info("Index cleared.")

    # 5. Build and upsert vectors
    logger.info("Building vectors...")
    vectors = build_vectors(regulations, model)

    logger.info("Upserting to Pinecone...")
    upsert_to_pinecone(vectors, index)

    # 6. Verify
    stats = index.describe_index_stats()
    total_vectors = stats["total_vector_count"]
    logger.info(f"=== Ingestion complete. Total vectors in index: {total_vectors} ===")

    # Sanity check
    expected = sum(
        len(chunk_regulation(r["full_text"])) for r in regulations
    )
    if total_vectors == expected:
        logger.info(f"Vector count matches expected: {expected}")
    else:
        logger.warning(f"Vector count mismatch. Expected {expected}, got {total_vectors}")


if __name__ == "__main__":
    main()