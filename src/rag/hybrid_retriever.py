# =============================================================
# FILE: src/rag/hybrid_retriever.py
# PURPOSE: Hybrid search combining BM25 (keyword) + ChromaDB
#          (semantic) for better legal document retrieval
#
# WHY HYBRID SEARCH?
# - Vector search is great for meaning/context
# - BM25 is great for exact legal keywords like
#   "indemnification", "force majeure", "termination"
# - Combining both = best of both worlds
#
# METHOD: Reciprocal Rank Fusion (RRF)
# Each result gets a score from both BM25 and vector search.
# RRF combines these scores into a single ranking.
# No duplicates — each unique clause appears only once.
# =============================================================

from rank_bm25 import BM25Okapi
from src.rag.retriever import retrieve_relevant_clauses, collection
from typing import List, Dict, Optional
import re


def tokenize(text: str) -> List[str]:
    """
    WHAT THIS DOES:
    Converts text into tokens for BM25 indexing.
    Preserves hyphenated legal terms like non-compete.
    """
    text = text.lower()
    tokens = re.findall(r'\b[\w-]+\b', text)
    return tokens


def make_unique_key(clause: Dict) -> str:
    """
    WHAT THIS DOES:
    Creates a truly unique key for each clause using
    contract name + clause number + first 30 chars of text.
    This prevents duplicate results across multiple contracts.
    """
    return f"{clause['contract']}|{str(clause['clause_number'])}|{clause['text'][:30]}"


def get_all_clauses(contract_name: Optional[str] = None) -> List[Dict]:
    """
    WHAT THIS DOES:
    Retrieves ALL clauses from ChromaDB for BM25 indexing.
    BM25 needs all documents to calculate term frequency correctly.

    INPUT:
        contract_name → optional filter by contract

    OUTPUT:
        List of all clause dictionaries
    """
    where_filter = None
    if contract_name:
        where_filter = {"contract": contract_name}

    all_items = collection.get(
        where=where_filter,
        include=["documents", "metadatas"]
    )

    if not all_items["documents"]:
        return []

    clauses = []
    for i, doc in enumerate(all_items["documents"]):
        clauses.append({
            "text": doc,
            "contract": all_items["metadatas"][i]["contract"],
            "clause_number": str(all_items["metadatas"][i]["clause_number"]),
        })

    return clauses


def bm25_search(
    query: str,
    clauses: List[Dict],
    n_results: int = 4
) -> List[Dict]:
    """
    WHAT THIS DOES:
    Performs BM25 keyword search across all clauses.
    BM25 is great at finding exact legal terms that
    semantic search might miss.

    INPUT:
        query     → user's question
        clauses   → all clauses to search through
        n_results → number of top results to return

    OUTPUT:
        List of top clauses ranked by BM25 score
    """
    if not clauses:
        return []

    # Tokenize all clause texts
    tokenized_clauses = [tokenize(clause["text"]) for clause in clauses]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_clauses)

    # Tokenize the query
    tokenized_query = tokenize(query)

    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)

    # Rank by score
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    # Return top n results
    results = []
    for idx in ranked_indices[:n_results]:
        clause = clauses[idx].copy()
        clause["bm25_score"] = float(scores[idx])
        results.append(clause)

    return results


def reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60
) -> List[Dict]:
    """
    WHAT THIS DOES:
    Combines vector search and BM25 results using
    Reciprocal Rank Fusion (RRF).

    RRF Formula: score = 1 / (k + rank)
    Results appearing in BOTH searches rank higher overall.
    Uses full text fingerprint to prevent ANY duplicates.

    INPUT:
        vector_results → results from ChromaDB semantic search
        bm25_results   → results from BM25 keyword search
        k              → smoothing constant (default 60)

    OUTPUT:
        Combined, deduplicated, re-ranked list of clauses
    """
    rrf_scores = {}

    # Score vector search results
    for rank, clause in enumerate(vector_results):
        # Use unique key including text fingerprint
        key = make_unique_key(clause)

        if key not in rrf_scores:
            rrf_scores[key] = {
                "clause": clause,
                "score": 0.0,
                "in_vector": False,
                "in_bm25": False
            }
        rrf_scores[key]["score"] += 1.0 / (k + rank + 1)
        rrf_scores[key]["in_vector"] = True

    # Score BM25 results
    for rank, clause in enumerate(bm25_results):
        key = make_unique_key(clause)

        if key not in rrf_scores:
            rrf_scores[key] = {
                "clause": clause,
                "score": 0.0,
                "in_vector": False,
                "in_bm25": False
            }
        rrf_scores[key]["score"] += 1.0 / (k + rank + 1)
        rrf_scores[key]["in_bm25"] = True

    # Sort by combined RRF score
    sorted_results = sorted(
        rrf_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    # Format final results
    final_results = []
    for item in sorted_results:
        clause = item["clause"].copy()
        clause["relevance_score"] = round(item["score"], 4)

        # Tag which search methods found this clause
        methods = []
        if item["in_vector"]:
            methods.append("vector")
        if item["in_bm25"]:
            methods.append("BM25")
        clause["search_method"] = " + ".join(methods)

        final_results.append(clause)

    return final_results


def hybrid_retrieve(
    question: str,
    contract_name: Optional[str] = None,
    n_results: int = 4
) -> List[Dict]:
    """
    WHAT THIS DOES:
    Main hybrid retrieval function — combines vector search
    and BM25 using Reciprocal Rank Fusion.

    Use this instead of retrieve_relevant_clauses() for
    better accuracy on legal documents.

    INPUT:
        question      → user's question in plain English
        contract_name → optional filter by contract
        n_results     → number of results to return

    OUTPUT:
        List of most relevant unique clauses using hybrid scoring
    """
    print(f"\nHybrid search for: '{question}'")

    # STEP 1: Get all clauses for BM25
    all_clauses = get_all_clauses(contract_name)

    if not all_clauses:
        print("No clauses found!")
        return []

    print(f"BM25 indexing {len(all_clauses)} clauses...")

    # STEP 2: BM25 keyword search
    bm25_results = bm25_search(question, all_clauses, n_results)
    print(f"BM25 found {len(bm25_results)} results")

    # STEP 3: Semantic vector search
    vector_results = retrieve_relevant_clauses(
        question=question,
        contract_name=contract_name,
        n_results=n_results
    )
    print(f"Vector search found {len(vector_results)} results")

    # STEP 4: Combine with RRF — guaranteed no duplicates!
    hybrid_results = reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results
    )

    final = hybrid_results[:n_results]
    print(f"Hybrid returning {len(final)} unique results")

    return final