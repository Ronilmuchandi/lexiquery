# =============================================================
# FILE: src/rag/parser.py
# PURPOSE: Read PDF files and extract all the text from them
# PART OF: LexiQuery - Intelligent Legal Contract Analysis System
# =============================================================

# pdfplumber is a library that can open and read PDF files
import pdfplumber

# io lets us handle PDF files that come as raw bytes (from uploads)
import io

# Union lets us say "this input can be either a file path OR bytes"
from typing import Union


def extract_text_from_pdf(pdf_input: Union[str, bytes]) -> str:
    """
    WHAT THIS DOES:
    Opens a PDF file and reads all the text from every page.
    Think of it like a human reading through a contract 
    page by page and writing down everything they see.
    
    INPUT:
        pdf_input → either a file path like "contracts/nda.pdf"
                    or raw bytes (when user uploads via the app)
    
    OUTPUT:
        One big string containing all the text from the PDF
    """
    try:
        # This variable will collect all text from every page
        full_text = ""
        
        # CASE 1: Input is raw bytes (coming from Streamlit file uploader)
        if isinstance(pdf_input, bytes):
            # BytesIO converts raw bytes into a file-like object
            # so pdfplumber can read it as if it were a normal file
            with pdfplumber.open(io.BytesIO(pdf_input)) as pdf:
                
                # Go through every single page in the PDF
                for page in pdf.pages:
                    
                    # Extract the text from this page
                    text = page.extract_text()
                    
                    # Only add if the page actually has text
                    # (some pages might just have images)
                    if text:
                        full_text += text + "\n"
        
        # CASE 2: Input is a file path (like "data/contracts/nda.pdf")
        else:
            with pdfplumber.open(pdf_input) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
        
        # .strip() removes any extra blank spaces at the start/end
        return full_text.strip()
    
    except Exception as e:
        # If anything goes wrong, tell us exactly what the error was
        raise Exception(f"PDF parsing error: {str(e)}")


def get_pdf_metadata(pdf_path: str) -> dict:
    """
    WHAT THIS DOES:
    Reads basic information ABOUT the PDF without reading its content.
    Like checking the cover of a book before reading it.
    
    INPUT:
        pdf_path → file path to the PDF
    
    OUTPUT:
        Dictionary with page count and document metadata
    
    EXAMPLE OUTPUT:
        {
            "page_count": 12,
            "metadata": {"Author": "Legal Corp", "Title": "NDA Agreement"}
        }
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return {
                # How many pages does this contract have?
                "page_count": len(pdf.pages),
                
                # Metadata = info embedded in the PDF like author, title, date
                "metadata": pdf.metadata
            }
    
    except Exception as e:
        raise Exception(f"Metadata extraction error: {str(e)}")