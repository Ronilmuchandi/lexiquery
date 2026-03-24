# =============================================================
# FILE: src/pipeline/comparator.py
# PURPOSE: Compare multiple contracts side by side and
#          generate a risk score (1-10) for each contract
#
# THIS IS WHAT MAKES LEXIQUERY UNIQUE:
# Most RAG projects just answer questions.
# We go further — we score contracts like a credit rating
# agency scores financial instruments.
# =============================================================

from src.llm.groq_client import get_llm_response
from src.rag.retriever import retrieve_relevant_clauses, collection
from typing import Dict, List, Optional


# -------------------------------------------------------
# RISK SCORING PROMPT
# Carefully engineered to get consistent 1-10 scores
# -------------------------------------------------------
RISK_SCORING_PROMPT = """
You are LexiQuery, an expert legal risk analyst.
Your job is to analyze contract clauses and assign
a risk score from 1 to 10.

RISK SCORE SCALE:
1-2  = Very low risk. Standard, fair, balanced contract.
3-4  = Low risk. Minor concerns but generally acceptable.
5-6  = Medium risk. Several concerning clauses. Review carefully.
7-8  = High risk. Significant red flags. Seek legal advice.
9-10 = Very high risk. Extremely one-sided. Do not sign without lawyer.

You MUST respond in this exact format:
RISK_SCORE: [number 1-10]
RISK_LEVEL: [Very Low/Low/Medium/High/Very High]
SUMMARY: [2-3 sentence plain English summary]
RED_FLAGS: [bullet point list of specific risky clauses]
RECOMMENDATIONS: [bullet point list of what to do]
"""


def score_contract_risk(contract_name: str) -> Dict:
    """
    WHAT THIS DOES:
    Analyzes all clauses of a contract and generates
    an overall risk score from 1-10.

    Think of it like a credit score — but for contracts.
    Higher score = more risky to sign.

    INPUT:
        contract_name → name of the indexed contract

    OUTPUT:
        Dictionary with risk score and detailed analysis
    """

    print(f"\nScoring risk for contract: {contract_name}")

    # -------------------------------------------------------
    # GET ALL CLAUSES FOR THIS CONTRACT
    # We want to analyze the WHOLE contract, not just
    # the most relevant parts
    # -------------------------------------------------------
    all_items = collection.get(
        where={"contract": contract_name},
        include=["documents", "metadatas"]
    )

    if not all_items["documents"]:
        return {
            "success": False,
            "error": f"Contract '{contract_name}' not found in database."
        }

    # Build full contract context from all clauses
    contract_text = ""
    for i, doc in enumerate(all_items["documents"]):
        contract_text += f"\nClause {i+1}:\n{doc}\n"

    # -------------------------------------------------------
    # ASK GROQ TO SCORE THE RISK
    # -------------------------------------------------------
    user_message = f"""
Please analyze this contract and provide a risk score:

{contract_text}

Remember to follow the exact response format specified.
"""

    print("Analyzing contract risk with Groq...")
    raw_response = get_llm_response(
        system_prompt=RISK_SCORING_PROMPT,
        user_message=user_message
    )

    # Parse the structured response
    result = parse_risk_response(raw_response, contract_name)
    return result


def parse_risk_response(raw_response: str, contract_name: str) -> Dict:
    """
    WHAT THIS DOES:
    Parses Groq's structured risk response into a clean
    dictionary we can display in the UI.

    INPUT:
        raw_response  → raw text from Groq
        contract_name → name of the contract

    OUTPUT:
        Clean dictionary with all risk fields
    """
    lines = raw_response.strip().split('\n')

    # Default values in case parsing fails
    result = {
        "success": True,
        "contract_name": contract_name,
        "risk_score": 5,
        "risk_level": "Medium",
        "summary": "",
        "red_flags": [],
        "recommendations": [],
        "raw_response": raw_response
    }

    current_section = None

    for line in lines:
        line = line.strip()

        if line.startswith("RISK_SCORE:"):
            try:
                result["risk_score"] = int(
                    line.replace("RISK_SCORE:", "").strip()
                )
            except:
                pass

        elif line.startswith("RISK_LEVEL:"):
            result["risk_level"] = line.replace(
                "RISK_LEVEL:", ""
            ).strip()

        elif line.startswith("SUMMARY:"):
            result["summary"] = line.replace(
                "SUMMARY:", ""
            ).strip()
            current_section = "summary"

        elif line.startswith("RED_FLAGS:"):
            current_section = "red_flags"

        elif line.startswith("RECOMMENDATIONS:"):
            current_section = "recommendations"

        elif line.startswith("-") or line.startswith("•") or line.startswith("*"):
            # Add bullet points to the right section
            clean_line = line.lstrip("-•*").strip()
            if current_section == "red_flags":
                result["red_flags"].append(clean_line)
            elif current_section == "recommendations":
                result["recommendations"].append(clean_line)

        elif current_section == "summary" and line:
            # Multi-line summary
            result["summary"] += " " + line

    return result


def compare_contracts(contract_names: List[str]) -> Dict:
    """
    WHAT THIS DOES:
    Compares multiple contracts side by side and
    highlights key differences between them.

    Perfect for comparing two versions of a contract
    or multiple vendor agreements.

    INPUT:
        contract_names → list of contract names to compare

    OUTPUT:
        Dictionary with comparison analysis
    """

    if len(contract_names) < 2:
        return {
            "success": False,
            "error": "Please provide at least 2 contracts to compare."
        }

    print(f"\nComparing contracts: {contract_names}")

    # Get risk scores for each contract
    scores = {}
    for name in contract_names:
        scores[name] = score_contract_risk(name)

    # Ask Groq to compare them
    comparison_prompt = """
    You are comparing multiple legal contracts.
    Identify:
    1. Key differences in obligations
    2. Which contract is more favorable and why
    3. Clauses that appear in one but not the other
    4. Overall recommendation on which to choose
    """

    # Build context with clauses from all contracts
    context = ""
    for name in contract_names:
        clauses = retrieve_relevant_clauses(
            "obligations termination confidentiality",
            contract_name=name,
            n_results=3
        )
        context += f"\n\n=== {name} ===\n"
        for c in clauses:
            context += f"\nClause {c['clause_number']}: {c['text']}\n"

    comparison = get_llm_response(
        system_prompt=comparison_prompt,
        user_message=f"Compare these contracts:\n{context}"
    )

    return {
        "success": True,
        "contracts_compared": contract_names,
        "individual_scores": scores,
        "comparison_analysis": comparison
    }