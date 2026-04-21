# =============================================================
# FILE: src/utils/faithfulness.py
# PURPOSE: Check if LLM answers are grounded in retrieved
#          clauses — detects hallucination
#
# WHY FAITHFULNESS MATTERS:
# In legal contexts, a hallucinated answer could be dangerous.
# Faithfulness scoring checks if the answer ONLY uses
# information from the retrieved clauses, not made-up facts.
#
# HOW IT WORKS:
# 1. Take the LLM answer
# 2. Take the retrieved clauses (context)
# 3. Ask another LLM call: "Is this answer supported by
#    this context? Score 0-1"
# 4. Score close to 1.0 = faithful, close to 0 = hallucinated
# =============================================================

from src.llm.groq_client import get_llm_response
from typing import Dict, List


FAITHFULNESS_PROMPT = """
You are a strict legal document auditor checking if an AI answer
is faithful to the source contract clauses provided.

Your job is to score how well the answer is supported by the
retrieved contract clauses — not by general legal knowledge.

SCORING RULES:
- Score 1.0: Every claim in the answer is directly supported
  by the provided clauses
- Score 0.7-0.9: Most claims are supported, minor elaboration
- Score 0.4-0.6: Some claims are supported, some are general
- Score 0.1-0.3: Few claims supported, mostly general knowledge
- Score 0.0: Answer contradicts or ignores the clauses entirely

You MUST respond in this EXACT format and nothing else:
FAITHFULNESS_SCORE: [number between 0.0 and 1.0]
REASONING: [one sentence explaining the score]
UNSUPPORTED_CLAIMS: [list any claims not found in clauses, or "None"]
"""


def score_faithfulness(
    question: str,
    answer: str,
    retrieved_clauses: List[Dict]
) -> Dict:
    """
    WHAT THIS DOES:
    Scores how faithful the LLM answer is to the retrieved
    contract clauses. Detects hallucination.

    INPUT:
        question          → the user's original question
        answer            → the LLM generated answer
        retrieved_clauses → the clauses used to generate answer

    OUTPUT:
        Dictionary with faithfulness score and reasoning
    """

    # Build context from retrieved clauses
    context = ""
    for clause in retrieved_clauses:
        context += f"\nClause {clause['clause_number']}:\n{clause['text']}\n"

    # Build the faithfulness check prompt
    user_message = f"""
Please evaluate the faithfulness of this AI answer:

ORIGINAL QUESTION:
{question}

RETRIEVED CONTRACT CLAUSES (the only allowed source):
{context}

AI ANSWER TO EVALUATE:
{answer}

Score the faithfulness of this answer based on the clauses provided.
"""

    print("Checking faithfulness...")

    # Get faithfulness score from LLM
    raw_response = get_llm_response(
        system_prompt=FAITHFULNESS_PROMPT,
        user_message=user_message
    )

    # Parse the response
    result = parse_faithfulness_response(raw_response, question)
    return result


def parse_faithfulness_response(raw_response: str, question: str) -> Dict:
    """
    WHAT THIS DOES:
    Parses the structured faithfulness score response
    into a clean dictionary.

    INPUT:
        raw_response → raw text from Groq
        question     → original question for reference

    OUTPUT:
        Clean dictionary with score and details
    """
    lines = raw_response.strip().split('\n')

    result = {
        "score": 0.5,
        "reasoning": "Could not parse response",
        "unsupported_claims": [],
        "raw_response": raw_response,
        "interpretation": "Unknown"
    }

    for line in lines:
        line = line.strip()

        if line.startswith("FAITHFULNESS_SCORE:"):
            try:
                score_str = line.replace("FAITHFULNESS_SCORE:", "").strip()
                result["score"] = float(score_str)
            except:
                pass

        elif line.startswith("REASONING:"):
            result["reasoning"] = line.replace("REASONING:", "").strip()

        elif line.startswith("UNSUPPORTED_CLAIMS:"):
            claims = line.replace("UNSUPPORTED_CLAIMS:", "").strip()
            if claims.lower() != "none":
                result["unsupported_claims"] = [
                    c.strip() for c in claims.split(",")
                    if c.strip()
                ]

    # Add human-readable interpretation
    score = result["score"]
    if score >= 0.8:
        result["interpretation"] = "Highly faithful — answer well grounded in clauses ✅"
    elif score >= 0.6:
        result["interpretation"] = "Mostly faithful — minor elaboration detected ⚠️"
    elif score >= 0.4:
        result["interpretation"] = "Partially faithful — some general knowledge used ⚠️"
    else:
        result["interpretation"] = "Low faithfulness — possible hallucination detected ❌"

    return result


def evaluate_answer_faithfulness(
    question: str,
    contract_name: str = None
) -> Dict:
    """
    WHAT THIS DOES:
    Full pipeline — generates an answer AND scores its
    faithfulness in one function call.

    INPUT:
        question      → user question
        contract_name → optional contract filter

    OUTPUT:
        Dictionary with answer, sources, and faithfulness score
    """
    from src.pipeline.analyzer import answer_legal_question

    print(f"\nQuestion: {question}")
    print("Generating answer...")

    # Generate answer using RAG pipeline
    qa_result = answer_legal_question(
        question=question,
        contract_name=contract_name
    )

    # Score faithfulness
    faithfulness = score_faithfulness(
        question=question,
        answer=qa_result["answer"],
        retrieved_clauses=qa_result["relevant_clauses"]
    )

    print(f"\nFaithfulness Score: {faithfulness['score']}")
    print(f"Interpretation: {faithfulness['interpretation']}")
    print(f"Reasoning: {faithfulness['reasoning']}")

    return {
        "question": question,
        "answer": qa_result["answer"],
        "clauses_used": qa_result["clauses_used"],
        "faithfulness_score": faithfulness["score"],
        "faithfulness_interpretation": faithfulness["interpretation"],
        "faithfulness_reasoning": faithfulness["reasoning"],
        "unsupported_claims": faithfulness["unsupported_claims"]
    }