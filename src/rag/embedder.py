# =============================================================
# FILE: src/rag/embedder.py
# PURPOSE: Convert text into numbers (vectors/embeddings)
#          so we can search by meaning, not just keywords
#
# SIMPLE ANALOGY:
# Think of embeddings like GPS coordinates for words.
# "Termination" and "End of Contract" would have very
# similar coordinates because they mean the same thing.
# This lets us find relevant clauses even when the user
# uses different words than the contract.
# =============================================================

# SentenceTransformer converts text into numerical vectors
from sentence_transformers import SentenceTransformer

# numpy helps us work with numerical arrays
import numpy as np

# Type hints for cleaner, more readable code
from typing import List

# -------------------------------------------------------
# EMBEDDING MODEL
# We use "all-MiniLM-L6-v2" because:
# - It's free and runs locally (no API cost)
# - It's fast and lightweight
# - It's specifically trained for semantic similarity
# - Perfect for finding related legal clauses
# -------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"

# Load the model once when this file is imported
# (Loading once is more efficient than loading every time)
print(f"Loading embedding model: {MODEL_NAME}...")
embedding_model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded successfully!")


def get_embedding(text: str) -> List[float]:
    """
    WHAT THIS DOES:
    Converts a single piece of text into a list of numbers
    that represent its meaning mathematically.
    
    INPUT:
        text → any string of text (a clause, a question, etc.)
    
    OUTPUT:
        A list of 384 numbers representing the text's meaning
    
    EXAMPLE:
        "termination clause" → [0.23, -0.45, 0.12, ...]
        "end of agreement"   → [0.21, -0.43, 0.11, ...]
        (similar meaning = similar numbers!)
    """
    # encode() converts text to a numpy array of numbers
    # tolist() converts it to a regular Python list
    embedding = embedding_model.encode(text)
    return embedding.tolist()


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    WHAT THIS DOES:
    Converts multiple pieces of text into embeddings at once.
    Much faster than calling get_embedding() one by one.
    
    INPUT:
        texts → list of text strings
    
    OUTPUT:
        List of embeddings, one per input text
    
    WHY BATCH?
    If a contract has 20 clauses, processing them all at once
    is much faster than processing them one at a time.
    """
    # batch_size=32 means process 32 texts at a time
    # show_progress_bar=True shows a progress bar for large batches
    embeddings = embedding_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )
    return embeddings.tolist()