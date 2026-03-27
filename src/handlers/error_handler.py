# =============================================================
# FILE: src/handlers/error_handler.py
# PURPOSE: Centralized error handling for LexiQuery
# =============================================================


def handle_pdf_error(error: Exception) -> dict:
    """
    Handle PDF parsing errors gracefully.
    
    INPUT:
        error → the exception that was raised
    
    OUTPUT:
        Dictionary with error details
    """
    return {
        "success": False,
        "error_type": "PDF Error",
        "message": str(error),
        "suggestion": "Please ensure the file is a valid PDF and not password protected."
    }


def handle_llm_error(error: Exception) -> dict:
    """
    Handle LLM API errors gracefully.
    
    INPUT:
        error → the exception that was raised
    
    OUTPUT:
        Dictionary with error details
    """
    return {
        "success": False,
        "error_type": "LLM Error",
        "message": str(error),
        "suggestion": "Please check your GROQ_API_KEY and try again."
    }


def handle_retrieval_error(error: Exception) -> dict:
    """
    Handle ChromaDB retrieval errors gracefully.
    
    INPUT:
        error → the exception that was raised
    
    OUTPUT:
        Dictionary with error details
    """
    return {
        "success": False,
        "error_type": "Retrieval Error",
        "message": str(error),
        "suggestion": "Please ensure a contract has been indexed before querying."
    }