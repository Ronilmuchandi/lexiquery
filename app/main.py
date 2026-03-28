# =============================================================
# FILE: app/main.py
# PURPOSE: FastAPI backend for LexiQuery
#          Provides REST API endpoints for contract analysis
#
# HOW TO RUN:
#   uvicorn app.main:app --reload
#
# API DOCS:
#   http://localhost:8000/docs  (Swagger UI)
# =============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile

from src.pipeline.indexer import process_and_index_contract, get_indexed_contracts
from src.pipeline.analyzer import answer_legal_question, flag_risky_clauses
from src.pipeline.comparator import score_contract_risk, compare_contracts

# =============================================================
# INITIALIZE FASTAPI APP
# =============================================================
app = FastAPI(
    title="LexiQuery API",
    description="AI-powered legal contract analysis REST API",
    version="1.0.0"
)

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# =============================================================
# REQUEST/RESPONSE MODELS
# =============================================================

class QuestionRequest(BaseModel):
    """Request model for contract Q&A."""
    question: str
    contract_name: Optional[str] = None


class CompareRequest(BaseModel):
    """Request model for contract comparison."""
    contract_names: list[str]


# =============================================================
# HEALTH CHECK
# =============================================================

@app.get("/")
def root():
    """
    Health check endpoint.
    Returns API status and version.
    """
    return {
        "status": "healthy",
        "app": "LexiQuery",
        "version": "1.0.0",
        "description": "AI-powered legal contract analyzer"
    }


@app.get("/health")
def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "components": {
            "api": "running",
            "vector_db": "chromadb",
            "llm": "groq/llama-3.3-70b"
        }
    }


# =============================================================
# CONTRACT ENDPOINTS
# =============================================================

@app.post("/contracts/upload")
async def upload_contract(file: UploadFile = File(...)):
    """
    WHAT THIS DOES:
    Upload and index a PDF contract for analysis.

    INPUT:
        file → PDF file (multipart form upload)

    OUTPUT:
        Success message with contract stats
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Get contract name from filename
    contract_name = file.filename.replace(".pdf", "").replace(" ", "_")

    # Save to temp file and process
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    result = process_and_index_contract(tmp_path, contract_name)
    os.unlink(tmp_path)

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return {
        "success": True,
        "contract_name": result["contract_name"],
        "total_clauses": result["total_clauses"],
        "total_characters": result["total_characters"],
        "message": f"Contract '{contract_name}' indexed successfully"
    }


@app.get("/contracts")
def list_contracts():
    """
    WHAT THIS DOES:
    List all indexed contracts.

    OUTPUT:
        List of contract names
    """
    contracts = get_indexed_contracts()
    return {
        "contracts": contracts,
        "total": len(contracts)
    }


# =============================================================
# ANALYSIS ENDPOINTS
# =============================================================

@app.post("/analyze/question")
def ask_question(request: QuestionRequest):
    """
    WHAT THIS DOES:
    Ask a question about an indexed contract.

    INPUT:
        question      → plain English question
        contract_name → optional filter by contract

    OUTPUT:
        AI answer with clause citations
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    result = answer_legal_question(
        question=request.question,
        contract_name=request.contract_name
    )

    return {
        "success": True,
        "question": request.question,
        "answer": result["answer"],
        "clauses_used": result["clauses_used"],
        "relevant_clauses": result["relevant_clauses"]
    }


@app.post("/analyze/risk/{contract_name}")
def analyze_risk(contract_name: str):
    """
    WHAT THIS DOES:
    Get risk score for a specific contract.

    INPUT:
        contract_name → name of the indexed contract

    OUTPUT:
        Risk score (1-10) with red flags and recommendations
    """
    result = score_contract_risk(contract_name)

    if not result["success"]:
        raise HTTPException(
            status_code=404,
            detail=result["error"]
        )

    return result


@app.post("/analyze/flags/{contract_name}")
def flag_risks(contract_name: str):
    """
    WHAT THIS DOES:
    Auto-scan contract for risky or unusual clauses.

    INPUT:
        contract_name → name of the indexed contract

    OUTPUT:
        List of flagged clauses with explanations
    """
    result = flag_risky_clauses(contract_name)
    return {
        "success": True,
        "contract_name": contract_name,
        "analysis": result["answer"],
        "sources": result["clauses_used"]
    }


@app.post("/analyze/compare")
def compare(request: CompareRequest):
    """
    WHAT THIS DOES:
    Compare multiple contracts side by side.

    INPUT:
        contract_names → list of contract names to compare

    OUTPUT:
        Comparison analysis with individual risk scores
    """
    if len(request.contract_names) < 2:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least 2 contracts to compare"
        )

    result = compare_contracts(request.contract_names)

    if not result["success"]:
        raise HTTPException(
            status_code=400,
            detail=result["error"]
        )

    return result