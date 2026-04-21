# =============================================================
# FILE: src/utils/evaluator.py
# PURPOSE: Evaluate RAG pipeline quality using Hit Rate and MRR
#
# WHY EVALUATION MATTERS:
# Without metrics, you can't prove your RAG works well.
# Hit Rate and MRR are standard IR (Information Retrieval)
# metrics used in academic and industry RAG systems.
#
# HIT RATE: What % of questions retrieved the right clause?
# MRR: How high up in the results was the right clause?
#      MRR=1.0 means always first, MRR=0.25 means always 4th
# =============================================================

from src.rag.hybrid_retriever import hybrid_retrieve
from src.rag.retriever import retrieve_relevant_clauses
from typing import List, Dict
import json


# =============================================================
# TEST DATASET
# 10 question-answer pairs manually created from the NDA
# Each entry has:
#   question      → what the user asks
#   relevant_clause → which clause number has the answer
#   contract      → which contract to search
# =============================================================
NDA_TEST_DATASET = [
    {
        "question": "What is considered confidential information?",
        "relevant_clause": "1",
        "contract": "NDA_Basic"
    },
    {
        "question": "What information is excluded from confidentiality?",
        "relevant_clause": "2",
        "contract": "NDA_Basic"
    },
    {
        "question": "What are my obligations as the receiving party?",
        "relevant_clause": "3",
        "contract": "NDA_Basic"
    },
    {
        "question": "How long do I need to keep information confidential?",
        "relevant_clause": "4",
        "contract": "NDA_Basic"
    },
    {
        "question": "Does this agreement make us business partners?",
        "relevant_clause": "5",
        "contract": "NDA_Basic"
    },
    {
        "question": "What happens if part of this agreement is invalid?",
        "relevant_clause": "6",
        "contract": "NDA_Basic"
    },
    {
        "question": "Can this agreement be changed verbally?",
        "relevant_clause": "7",
        "contract": "NDA_Basic"
    },
    {
        "question": "What if I don't enforce my rights under this agreement?",
        "relevant_clause": "8",
        "contract": "NDA_Basic"
    },
    {
        "question": "Can I report suspected illegal activity without liability?",
        "relevant_clause": "9",
        "contract": "NDA_Basic"
    },
    {
        "question": "What are the obligations when terminating this agreement?",
        "relevant_clause": "4",
        "contract": "NDA_Basic"
    }
]


def evaluate_retrieval(
    test_dataset: List[Dict] = NDA_TEST_DATASET,
    n_results: int = 4,
    use_hybrid: bool = True
) -> Dict:
    """
    WHAT THIS DOES:
    Runs all test questions through the retrieval pipeline
    and measures how well it finds the right clauses.

    INPUT:
        test_dataset → list of question/answer pairs
        n_results    → how many results to retrieve per question
        use_hybrid   → True = hybrid search, False = vector only

    OUTPUT:
        Dictionary with Hit Rate, MRR, and per-question results
    """

    print(f"\n{'='*60}")
    print(f"RAG EVALUATION REPORT")
    print(f"Method: {'Hybrid (BM25 + Vector)' if use_hybrid else 'Vector Only'}")
    print(f"Test questions: {len(test_dataset)}")
    print(f"{'='*60}\n")

    results = []
    hits = 0
    reciprocal_ranks = []

    for i, test in enumerate(test_dataset):
        question = test["question"]
        relevant_clause = str(test["relevant_clause"])
        contract = test["contract"]

        print(f"Q{i+1}: {question}")

        # Run retrieval
        if use_hybrid:
            retrieved = hybrid_retrieve(
                question=question,
                contract_name=contract,
                n_results=n_results
            )
        else:
            retrieved = retrieve_relevant_clauses(
                question=question,
                contract_name=contract,
                n_results=n_results
            )

        # Check if relevant clause was retrieved
        retrieved_clauses = [
            str(r["clause_number"]) for r in retrieved
        ]

        hit = relevant_clause in retrieved_clauses
        hits += 1 if hit else 0

        # Calculate reciprocal rank
        # If clause found at position 1 → RR = 1.0
        # If found at position 2 → RR = 0.5
        # If not found → RR = 0.0
        rr = 0.0
        if hit:
            rank = retrieved_clauses.index(relevant_clause) + 1
            rr = 1.0 / rank
        reciprocal_ranks.append(rr)

        # Store result
        result = {
            "question": question,
            "relevant_clause": relevant_clause,
            "retrieved_clauses": retrieved_clauses,
            "hit": hit,
            "reciprocal_rank": round(rr, 3)
        }
        results.append(result)

        status = "✅ HIT" if hit else "❌ MISS"
        print(f"   Expected: Clause {relevant_clause}")
        print(f"   Retrieved: {retrieved_clauses}")
        print(f"   Status: {status} | RR: {round(rr, 3)}\n")

    # Calculate final metrics
    hit_rate = hits / len(test_dataset)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print(f"{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Hit Rate: {hit_rate:.1%} ({hits}/{len(test_dataset)} questions)")
    print(f"MRR:      {mrr:.3f} (higher is better, max=1.0)")
    print(f"{'='*60}\n")

    # Interpretation
    if hit_rate >= 0.8:
        print("Retrieval quality: EXCELLENT ✅")
    elif hit_rate >= 0.6:
        print("Retrieval quality: GOOD ✅")
    elif hit_rate >= 0.4:
        print("Retrieval quality: FAIR ⚠️")
    else:
        print("Retrieval quality: POOR ❌")

    return {
        "hit_rate": round(hit_rate, 3),
        "mrr": round(mrr, 3),
        "total_questions": len(test_dataset),
        "total_hits": hits,
        "method": "hybrid" if use_hybrid else "vector",
        "per_question_results": results
    }


def compare_retrieval_methods() -> Dict:
    """
    WHAT THIS DOES:
    Runs evaluation on BOTH hybrid and vector-only search
    and compares the results side by side.

    This is the A/B test that proves hybrid search is better!

    OUTPUT:
        Comparison dictionary with both method results
    """
    print("\n" + "="*60)
    print("COMPARING RETRIEVAL METHODS")
    print("="*60)

    print("\n--- Running Vector-Only Search ---")
    vector_results = evaluate_retrieval(use_hybrid=False)

    print("\n--- Running Hybrid Search (BM25 + Vector) ---")
    hybrid_results = evaluate_retrieval(use_hybrid=True)

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Method':<25} {'Hit Rate':>10} {'MRR':>10}")
    print("-"*45)
    print(f"{'Vector Only':<25} {vector_results['hit_rate']:>10.1%} {vector_results['mrr']:>10.3f}")
    print(f"{'Hybrid (BM25+Vector)':<25} {hybrid_results['hit_rate']:>10.1%} {hybrid_results['mrr']:>10.3f}")
    print("="*60)

    improvement = hybrid_results["hit_rate"] - vector_results["hit_rate"]
    if improvement > 0:
        print(f"\nHybrid search improved Hit Rate by {improvement:.1%} ✅")
    elif improvement == 0:
        print("\nBoth methods performed equally well")
    else:
        print(f"\nVector search performed better by {abs(improvement):.1%}")

    return {
        "vector_only": vector_results,
        "hybrid": hybrid_results,
        "hit_rate_improvement": round(improvement, 3)
    }