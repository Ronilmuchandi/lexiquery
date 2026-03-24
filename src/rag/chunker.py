# =============================================================
# FILE: src/rag/chunker.py
# PURPOSE: Split a large contract into smaller meaningful pieces
#          called "chunks" or "clauses" for better searching
# 
# WHY WE CHUNK:
# A contract can be 50 pages long. If someone asks about 
# "termination policy", we don't want to search all 50 pages.
# We split it into clauses so we can find just the relevant 
# clause in seconds.
# =============================================================

# re is Python's regular expression library
# We use it to detect patterns like "Section 1" or "Article 2"
import re

# Type hints make code more readable and professional
from typing import List, Dict


def chunk_by_clauses(text: str, contract_name: str) -> List[Dict]:
    """
    WHAT THIS DOES:
    Takes a full contract text and splits it into individual clauses.
    
    Think of a contract like a book with chapters.
    This function finds where each "chapter" (clause) starts
    and splits the contract there.
    
    INPUT:
        text          → the full text of the contract
        contract_name → name of the contract (e.g. "NDA_2024")
    
    OUTPUT:
        A list of clause dictionaries. Example:
        [
            {
                "text": "Section 1. This agreement is between...",
                "clause_number": 1,
                "contract": "NDA_2024",
                "char_count": 450
            },
            ...
        ]
    """
    # This list will store all the clauses we find
    clauses = []
    
    # -------------------------------------------------------
    # CLAUSE DETECTION PATTERN
    # We look for common legal clause markers like:
    # "1."  "1.1"  "Section 1"  "Article 1"  "CLAUSE 1"
    # (?=...) means "split AT this point but keep the marker"
    # -------------------------------------------------------
    clause_pattern = r'(?=(?:\d+\.\d*|\bSection\s+\d+|\bArticle\s+\d+|\bCLAUSE\s+\d+)[\s\.])'
    
    # Split the full text wherever a clause marker appears
    sections = re.split(clause_pattern, text, flags=re.IGNORECASE)
    
    # Go through each section we found
    for i, section in enumerate(sections):
        
        # Remove extra whitespace from start and end
        section = section.strip()
        
        # Skip very tiny fragments (under 100 characters)
        # These are usually just headings or page numbers, not real clauses
        if len(section) > 100:
            clauses.append({
                # The actual text of this clause
                "text": section,
                
                # Which clause number is this? (for reference)
                "clause_number": i,
                
                # Which contract does this clause belong to?
                "contract": contract_name,
                
                # How long is this clause? (useful for debugging)
                "char_count": len(section)
            })
    
    # -------------------------------------------------------
    # FALLBACK: If no clause markers were found
    # (some contracts use different formatting)
    # we fall back to splitting by paragraphs instead
    # -------------------------------------------------------
    if len(clauses) == 0:
        print(f"No clause markers found in {contract_name}. Using paragraph chunking.")
        clauses = chunk_by_paragraphs(text, contract_name)
    
    # Tell us how many clauses were found
    print(f"Found {len(clauses)} clauses in {contract_name}")
    
    return clauses


def chunk_by_paragraphs(text: str, contract_name: str) -> List[Dict]:
    """
    WHAT THIS DOES:
    Fallback method. If we can't find standard clause markers,
    we split the contract by paragraphs (blank lines) instead.
    
    Think of it like splitting a document by double line breaks.
    
    INPUT:
        text          → full contract text
        contract_name → name of the contract
    
    OUTPUT:
        Same format as chunk_by_clauses — list of chunk dictionaries
    """
    # Split by double newlines (paragraph breaks)
    # Filter out any paragraphs shorter than 100 characters
    paragraphs = [
        p.strip() 
        for p in text.split('\n\n') 
        if len(p.strip()) > 100
    ]
    
    # Build the same dictionary format as chunk_by_clauses
    return [
        {
            "text": para,
            "clause_number": i,
            "contract": contract_name,
            "char_count": len(para)
        }
        for i, para in enumerate(paragraphs)
    ]