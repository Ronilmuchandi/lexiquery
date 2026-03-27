# =============================================================
# FILE: src/llm/utils.py
# PURPOSE: Helper utilities for LLM operations
# =============================================================

def truncate_text(text: str, max_chars: int = 4000) -> str:
    """
    Truncate text to avoid exceeding LLM token limits.
    
    INPUT:
        text      → text to truncate
        max_chars → maximum characters allowed
    
    OUTPUT:
        Truncated text with notice if cut
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated for length]"


def format_context(clauses: list) -> str:
    """
    Format retrieved clauses into clean context for LLM.
    
    INPUT:
        clauses → list of clause dictionaries
    
    OUTPUT:
        Formatted string ready to send to LLM
    """
    context = ""
    for clause in clauses:
        context += f"""
--- Clause {clause['clause_number']} from {clause['contract']} ---
{clause['text']}
"""
    return context