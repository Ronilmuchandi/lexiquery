# =============================================================
# FILE: src/rag/retriever.py
# PURPOSE: Store clause embeddings in ChromaDB and retrieve
#          the most relevant clauses for any given question
#
# SIMPLE ANALOGY:
# ChromaDB is like a super smart filing cabinet.
# Instead of filing by name or date, it files by MEANING.
# When you ask a question, it finds the most semantically
# similar clauses instantly.
# =============================================================

# ChromaDB is our vector database
import chromadb

# Our embedding function from embedder.py
from src.rag.embedder import get_embedding, get_embeddings_batch

# Type hints
from typing import List, Dict, Optional

# -------------------------------------------------------
# INITIALIZE CHROMADB
# We use persistent storage so embeddings are saved to disk
# and don't need to be recomputed every time we restart
# -------------------------------------------------------
chroma_client = chromadb.PersistentClient(
    path="data/vectorstore"  # Saved here on disk
)

# Get or create our contracts collection
# Think of a collection like a table in a database
collection = chroma_client.get_or_create_collection(
    name="legal_contracts",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity for matching
)


def index_contract(clauses: List[Dict]) -> None:
    """
    WHAT THIS DOES:
    Takes all the clauses from a contract and stores them
    in ChromaDB with their embeddings for future searching.
    
    Think of it like indexing a book — we're creating a
    searchable index of every clause in the contract.
    
    INPUT:
        clauses → list of clause dictionaries from chunker.py
    
    OUTPUT:
        Nothing — but ChromaDB now has all clauses stored
    """
    if not clauses:
        print("No clauses to index!")
        return
    
    print(f"Indexing {len(clauses)} clauses into ChromaDB...")
    
    # Extract just the text from each clause for batch embedding
    texts = [clause["text"] for clause in clauses]
    
    # Convert all clause texts to embeddings at once (faster!)
    print("Generating embeddings for all clauses...")
    embeddings = get_embeddings_batch(texts)
    
    # Prepare data for ChromaDB
    documents = []      # The actual clause texts
    ids = []            # Unique ID for each clause
    metadatas = []      # Extra info about each clause
    embeds = []         # The numerical embeddings
    
    for i, clause in enumerate(clauses):
        # Create a unique ID for this clause
        clause_id = f"{clause['contract']}_clause_{clause['clause_number']}"
        
        # Skip if this clause is already indexed
        # (prevents duplicates if we re-run)
        existing = collection.get(ids=[clause_id])
        if existing['ids']:
            continue
            
        documents.append(clause["text"])
        ids.append(clause_id)
        metadatas.append({
            "contract": clause["contract"],
            "clause_number": str(clause["clause_number"]),
            "char_count": str(clause["char_count"])
        })
        embeds.append(embeddings[i])
    
    # Only add if we have new clauses to add
    if documents:
        collection.add(
            documents=documents,
            embeddings=embeds,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Successfully indexed {len(documents)} clauses!")
    else:
        print("All clauses already indexed!")


def retrieve_relevant_clauses(
    question: str,
    contract_name: Optional[str] = None,
    n_results: int = 4
) -> List[Dict]:
    """
    WHAT THIS DOES:
    Takes a user's question and finds the most relevant
    clauses from the indexed contracts.
    
    Think of it like a super smart CTRL+F that understands
    meaning, not just exact words.
    
    INPUT:
        question      → the user's question in plain English
        contract_name → optional, filter by specific contract
        n_results     → how many relevant clauses to return (default 4)
    
    OUTPUT:
        List of the most relevant clauses with their metadata
    
    EXAMPLE:
        Question: "What happens if I share confidential info?"
        Returns: The confidentiality obligation clause
    """
    # Convert the question to an embedding
    print(f"Searching for: '{question}'")
    question_embedding = get_embedding(question)
    
    # Build filter if a specific contract is requested
    where_filter = None
    if contract_name:
        where_filter = {"contract": contract_name}
    
    # Search ChromaDB for most similar clauses
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results into a clean list of dictionaries
    relevant_clauses = []
    
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            relevant_clauses.append({
                # The actual clause text
                "text": doc,
                
                # Where this clause came from
                "contract": results["metadatas"][0][i]["contract"],
                "clause_number": results["metadatas"][0][i]["clause_number"],
                
                # How similar is this clause to the question?
                # Lower distance = more relevant (0 = perfect match)
                "relevance_score": round(1 - results["distances"][0][i], 3)
            })
    
    print(f"Found {len(relevant_clauses)} relevant clauses")
    return relevant_clauses