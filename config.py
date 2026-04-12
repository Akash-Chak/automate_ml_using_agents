# config.py

import os
from functools import lru_cache
from typing import Tuple

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI, AuthenticationError, APIConnectionError

load_dotenv(find_dotenv(), override=False)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set or could not be loaded from the environment.")
    return OpenAI(api_key=api_key)


def validate_api_key() -> Tuple[bool, str]:
    """
    Validate the OpenAI API key at startup.
    Returns (ok: bool, message: str).
    Makes a lightweight models.list() call — no tokens consumed.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False, "OPENAI_API_KEY is not set. Add it to your .env file."
    if not api_key.startswith("sk-"):
        return False, "OPENAI_API_KEY looks malformed (should start with 'sk-')."
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()          # free call — just verifies the key is accepted
        return True, f"API key valid. Model: {MODEL}"
    except AuthenticationError:
        return False, "OPENAI_API_KEY is invalid or revoked (401 Unauthorized)."
    except APIConnectionError:
        return False, "Could not reach OpenAI API. Check your internet connection."
    except Exception as e:
        return False, f"API key check failed: {e}"


def call_llm(prompt: str) -> str:
    client = get_openai_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content
