# =============================================================
# FILE: src/llm/groq_client.py
# PURPOSE: Connect LexiQuery to Groq API (Llama 3.3 70B)
#          with rate limiting to prevent quota exhaustion
#
# SIMPLE ANALOGY:
# Think of this as the phone line to our AI lawyer.
# Every time we need an answer, we call through this file.
# Rate limiting ensures we don't call too fast and get blocked.
# =============================================================

import os
from groq import Groq
from dotenv import load_dotenv
from src.utils.rate_limiter import groq_rate_limiter

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client with API key from .env
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Model we are using
# Llama 3.3 70B is one of the most capable open source models
MODEL = "llama-3.3-70b-versatile"


def get_llm_response(system_prompt: str, user_message: str) -> str:
    """
    WHAT THIS DOES:
    Sends a message to Groq LLM and gets a response.
    Applies rate limiting before every API call to prevent
    hitting Groq's quota limits.

    INPUT:
        system_prompt → instructions for how LLM should behave
        user_message  → the actual question or input

    OUTPUT:
        The LLM response as a string
    """
    try:
        # Apply rate limiting before every API call
        # This ensures we never exceed 30 calls per minute
        groq_rate_limiter.wait_if_needed()

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            # Low temperature = more precise, consistent answers
            # Important for legal accuracy
            temperature=0.1,
            max_tokens=2048
        )
        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"LLM Error: {str(e)}")


def test_connection():
    """
    WHAT THIS DOES:
    Tests that the Groq API connection is working correctly.
    Run this to verify your API key is valid.

    OUTPUT:
        A confirmation message from the LLM
    """
    response = get_llm_response(
        system_prompt="You are a helpful assistant.",
        user_message="Say 'LexiQuery is connected!' in one sentence."
    )
    return response