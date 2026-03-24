# =============================================================
# FILE: src/rag/parser.py
# PURPOSE: Read PDF files and extract all the text from them
# PART OF: LexiQuery - Intelligent Legal Contract Analysis System
# NOTE: Using pypdfium2 instead of pdfplumber for better
#       compatibility across Python versions
# =============================================================

import pypdfium2 as pdfium
import io
from typing import Union


def extract_text_from_pdf(pdf_input: Union[str, bytes]) -> str:
    """
    WHAT THIS DOES:
    Opens a PDF file and reads all the text from every page.
    
    INPUT:
        pdf_input → either a file path or raw bytes
    
    OUTPUT:
        One big string containing all the text from the PDF
    """
    try:
        full_text = ""

        # Handle both file path and bytes input
        if isinstance(pdf_input, bytes):
            pdf = pdfium.PdfDocument(pdf_input)
        else:
            pdf = pdfium.PdfDocument(pdf_input)

        # Go through every page
        for page in pdf:
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            if text:
                full_text += text + "\n"

        return full_text.strip()

    except Exception as e:
        raise Exception(f"PDF parsing error: {str(e)}")


def get_pdf_metadata(pdf_path: str) -> dict:
    """
    WHAT THIS DOES:
    Reads basic information about the PDF.
    
    INPUT:
        pdf_path → file path to the PDF
    
    OUTPUT:
        Dictionary with page count
    """
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        return {
            "page_count": len(pdf),
        }
    except Exception as e:
        raise Exception(f"Metadata extraction error: {str(e)}")