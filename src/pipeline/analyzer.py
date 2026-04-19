# =============================================================
# FILE: src/pipeline/analyzer.py
# PURPOSE: The brain of LexiQuery — connects ChromaDB retrieval
#          with Groq LLM to answer legal questions in plain English
#
# THE RAG FLOW:
# User Question → Find Relevant Clauses → Send to Groq → Get Answer
#
# SIMPLE ANALOGY:
# Imagine you have a lawyer (Groq) and a filing cabinet (ChromaDB)
# When you ask a question:
# 1. Assistant finds relevant files (ChromaDB retrieval)
# 2. Hands them to the lawyer (Groq)
# 3. Lawyer reads them and gives you a plain English answer
# =============================================================

# Our Groq LLM connection
from src.llm.groq_client import get_llm_response

# Our ChromaDB retrieval function
from src.rag.hybrid_retriever import hybrid_retrieve as retrieve_relevant_clauses

# Type hints
from typing import Dict, Optional

# -------------------------------------------------------
# LEGAL SYSTEM PROMPT
# This tells Groq exactly how to behave —
# like a legal assistant helping non-lawyers
# understand contracts
# -------------------------------------------------------
LEGAL_SYSTEM_PROMPT = """
You are LexiQuery, an expert legal assistant helping 
non-lawyers understand contracts clearly and safely.

Your job is to:
- Explain legal terms in plain simple English
- Highlight potentially risky or unusual clauses
- Flag anything that seems one-sided or unusual
- Always cite the specific clause your answer comes from
- Always recommend consulting a qualified lawyer 
  for important legal decisions

Your tone should be:
- Clear and simple (explain like the user is not a lawyer)
- Honest (flag risks even if the user didn't ask)
- Helpful (give actionable insights, not just summaries)

IMPORTANT: Never make up information. Only answer based 
on the contract clauses provided to you.
"""


def answer_legal_question(
    question: str,
    contract_name: Optional[str] = None
) -> Dict:
    """
    WHAT THIS DOES:
    Takes a user's question about a contract and returns
    a plain English answer with sources cited.
    
    THIS IS THE CORE RAG FUNCTION:
    Retrieve → Augment → Generate
    
    INPUT:
        question      → user's question in plain English
        contract_name → optional, filter to specific contract
    
    OUTPUT:
        Dictionary with answer, sources, and relevant clauses
    
    EXAMPLE:
        Input:  "What happens if I break this agreement?"
        Output: {
            "answer": "If you break this agreement...",
            "clauses_used": ["Clause 3", "Clause 4"],
            "contract": "NDA_Basic"
        }
    """
    
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"{'='*50}")
    
    # -------------------------------------------------------
    # STEP 1: RETRIEVE
    # Find the most relevant clauses from ChromaDB
    # -------------------------------------------------------
    print("Step 1: Retrieving relevant clauses...")
    relevant_clauses = retrieve_relevant_clauses(
        question=question,
        contract_name=contract_name,
        n_results=4  # Get top 4 most relevant clauses
    )
    
    if not relevant_clauses:
        return {
            "answer": "I could not find relevant clauses to answer your question. Please make sure a contract has been uploaded and indexed.",
            "clauses_used": [],
            "relevant_clauses": []
        }
    
    # -------------------------------------------------------
    # STEP 2: AUGMENT
    # Build context from retrieved clauses to send to Groq
    # We format them clearly so Groq knows which clause
    # each piece of text comes from
    # -------------------------------------------------------
    print("Step 2: Building context from clauses...")
    context = ""
    for i, clause in enumerate(relevant_clauses):
        context += f"""
--- Clause {clause['clause_number']} from {clause['contract']} ---
(Relevance Score: {clause['relevance_score']})
{clause['text']}
"""
    
    # Build the full message to send to Groq
    user_message = f"""
Here are the most relevant contract clauses I found:

{context}

Based ONLY on these clauses, please answer this question:
{question}

Please:
1. Give a clear plain English answer
2. Cite which clause number(s) your answer comes from
3. Flag any risks or unusual terms you notice
4. Recommend consulting a lawyer if this is important
"""
    
    # -------------------------------------------------------
    # STEP 3: GENERATE
    # Send context + question to Groq and get the answer
    # -------------------------------------------------------
    print("Step 3: Generating answer with Groq LLM...")
    answer = get_llm_response(
        system_prompt=LEGAL_SYSTEM_PROMPT,
        user_message=user_message
    )
    
    # Format the sources for display
    clauses_used = [
        f"Clause {c['clause_number']} from {c['contract']} (relevance: {c['relevance_score']})"
        for c in relevant_clauses
    ]
    
    print("Answer generated successfully!")
    
    return {
        # The actual plain English answer from Groq
        "answer": answer,
        
        # Which clauses were used to generate the answer
        "clauses_used": clauses_used,
        
        # The full clause objects for display in the UI
        "relevant_clauses": relevant_clauses
    }


def flag_risky_clauses(contract_name: str) -> Dict:
    """
    WHAT THIS DOES:
    Automatically scans an entire contract and flags
    any clauses that seem risky, unusual, or one-sided.
    
    No question needed — this runs a full risk scan.
    
    INPUT:
        contract_name → name of the contract to scan
    
    OUTPUT:
        Dictionary with risk analysis and flagged clauses
    """
    
    # Ask Groq to scan for risks across the whole contract
    risk_question = """
    Scan this contract and identify:
    1. Any clauses that heavily favour one party
    2. Unusual termination conditions
    3. Liability limitations or waivers
    4. Automatic renewal clauses
    5. Any other red flags a non-lawyer should know about
    
    For each risk found, explain:
    - What the risk is
    - Why it matters
    - What the person should do about it
    """
    
    return answer_legal_question(
        question=risk_question,
        contract_name=contract_name
    )