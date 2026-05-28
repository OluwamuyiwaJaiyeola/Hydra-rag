import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hydra-regulations")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in .env")