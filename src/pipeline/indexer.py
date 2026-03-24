# =============================================================
# FILE: src/pipeline/indexer.py
# PURPOSE: Handles the full indexing pipeline —
#          parse PDF → chunk → embed → store in ChromaDB
#
# THINK OF THIS AS:
# The "intake" process when a new contract is uploaded.
# Like a law firm's intake process when they receive
# a new document — they read it, organize it, and
# file it properly before any analysis begins.
# =============================================================

# Our parser reads the PDF
from src.rag.parser import extract_text_from_pdf

# Our chunker splits it into clauses
from src.rag.chunker import chunk_by_clauses

# Our retriever stores it in ChromaDB
from src.rag.retriever import index_contract, collection

# Type hints
from typing import Dict


def process_and_index_contract(
    pdf_input,
    contract_name: str
) -> Dict:
    """
    WHAT THIS DOES:
    Full pipeline to take a raw PDF and make it
    searchable in ChromaDB.

    Steps:
    1. Extract text from PDF
    2. Split into clauses
    3. Generate embeddings
    4. Store in ChromaDB

    INPUT:
        pdf_input     → file path or bytes of the PDF
        contract_name → name to identify this contract

    OUTPUT:
        Dictionary with indexing results and stats
    """

    print(f"\nProcessing contract: {contract_name}")

    # -------------------------------------------------------
    # STEP 1: EXTRACT TEXT FROM PDF
    # -------------------------------------------------------
    print("Step 1: Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_input)

    if not text:
        return {
            "success": False,
            "error": "Could not extract text from PDF. File may be scanned or corrupted.",
            "contract_name": contract_name
        }

    print(f"Extracted {len(text)} characters")

    # -------------------------------------------------------
    # STEP 2: CHUNK INTO CLAUSES
    # -------------------------------------------------------
    print("Step 2: Splitting into clauses...")
    clauses = chunk_by_clauses(text, contract_name)

    if not clauses:
        return {
            "success": False,
            "error": "Could not identify clauses in this document.",
            "contract_name": contract_name
        }

    # -------------------------------------------------------
    # STEP 3: INDEX INTO CHROMADB
    # -------------------------------------------------------
    print("Step 3: Indexing into ChromaDB...")
    index_contract(clauses)

    print(f"Contract '{contract_name}' successfully processed!")

    return {
        "success": True,
        "contract_name": contract_name,
        "total_clauses": len(clauses),
        "total_characters": len(text),
        "clauses": clauses
    }


def get_indexed_contracts() -> list:
    """
    WHAT THIS DOES:
    Returns a list of all contracts currently
    stored in ChromaDB.

    Like checking what files are in the filing cabinet.

    OUTPUT:
        List of unique contract names
    """
    # Get all items from ChromaDB
    all_items = collection.get()

    if not all_items["metadatas"]:
        return []

    # Extract unique contract names
    contracts = list(set(
        meta["contract"]
        for meta in all_items["metadatas"]
    ))

    return contracts