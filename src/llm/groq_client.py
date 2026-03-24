import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Model we are using
MODEL = "llama-3.3-70b-versatile"

def get_llm_response(system_prompt: str, user_message: str) -> str:
    """
    Send a message to Groq LLM and get a response.
    
    Args:
        system_prompt: Instructions for how the LLM should behave
        user_message: The actual question or input
    
    Returns:
        The LLM's response as a string
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,  # Low temperature = more precise legal answers
            max_tokens=2048
        )
        return response.choices[0].message.content
    
    except Exception as e:
        raise Exception(f"LLM Error: {str(e)}")


def test_connection():
    """Test that Groq API is working."""
    response = get_llm_response(
        system_prompt="You are a helpful assistant.",
        user_message="Say 'LexiQuery is connected!' in one sentence."
    )
    return response