# config.py

import json
import os
from datetime import datetime
from functools import lru_cache
from typing import Tuple

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI, AuthenticationError, APIConnectionError

load_dotenv(find_dotenv(), override=False)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── LLM call logger ──────────────────────────────────────────────────────────
# All calls to call_llm() are appended to logs/llm_calls_<date>.jsonl
# Each line is a self-contained JSON object with: timestamp, agent, prompt,
# raw_response, tokens_used.  Use a JSON-lines file so it stays appendable
# across multiple runs in the same day.

_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


def _llm_log_path() -> str:
    os.makedirs(_LOG_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(_LOG_DIR, f"llm_calls_{date_str}.jsonl")


def _log_llm_call(agent: str, prompt: str, response: str, usage: dict | None):
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "agent":     agent,
        "model":     MODEL,
        "prompt":    prompt,
        "response":  response,
        "usage":     usage,
    }
    try:
        with open(_llm_log_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[LLM logger] Could not write log: {exc}")


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


def call_llm(prompt: str, agent: str = "unknown") -> str:
    """
    Call the LLM and return the response text.

    Every call is logged to logs/llm_calls_<date>.jsonl so you can inspect
    exactly what was sent and received for each agent decision.

    Args:
        prompt: The full prompt string sent to the model.
        agent:  Name of the calling agent (e.g. "decision_preprocessing").
                Used as a label in the log file.
    """
    client = get_openai_client()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    response_text = completion.choices[0].message.content
    usage = None
    if completion.usage:
        usage = {
            "prompt_tokens":     completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens":      completion.usage.total_tokens,
        }
    _log_llm_call(agent, prompt, response_text, usage)
    return response_text
