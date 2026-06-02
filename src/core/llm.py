from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_GOOGLE_RATE_LIMITER: Any | None = None
_GOOGLE_RATE_LIMITER_RPM: float | None = None


def get_google_rate_limiter():
    global _GOOGLE_RATE_LIMITER, _GOOGLE_RATE_LIMITER_RPM

    requests_per_minute = float(os.getenv("GOOGLE_REQUESTS_PER_MINUTE", "4"))
    if requests_per_minute <= 0:
        return None
    if _GOOGLE_RATE_LIMITER is not None and _GOOGLE_RATE_LIMITER_RPM == requests_per_minute:
        return _GOOGLE_RATE_LIMITER

    from langchain_core.rate_limiters import InMemoryRateLimiter

    _GOOGLE_RATE_LIMITER_RPM = requests_per_minute
    _GOOGLE_RATE_LIMITER = InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60,
        check_every_n_seconds=0.5,
        max_bucket_size=1,
    )
    return _GOOGLE_RATE_LIMITER


def normalize_content(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        text = raw.get("text")
        return str(text).strip() if text is not None else str(raw).strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            text = normalize_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(raw).strip()


def build_chat_model(
    *,
    provider: str = "google",
    model_name: str | None = None,
    temperature: float = 0.0,
):
    normalized_provider = provider.lower().strip()
    if normalized_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            rate_limiter=get_google_rate_limiter(),
        )
    if normalized_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL", "qwen3.5:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )
    if normalized_provider in {"endpoint", "llm_endpoint", "openai_compatible"}:
        from langchain_openai import ChatOpenAI

        api_key = (os.getenv("LLM_API_KEY") or "").strip()
        endpoint = (os.getenv("LLM_ENDPOINT") or "").strip().rstrip("/")
        provider_name = (os.getenv("LLM_PROVIDER_NAME", "llm_endpoint") or "llm_endpoint").strip()
        selected_model = (model_name or os.getenv("DEFAULT_MODEL", "gpt-4o-mini")).strip()
        if not api_key:
            raise ValueError("LLM_API_KEY is required for the endpoint provider.")
        if not endpoint:
            raise ValueError("LLM_ENDPOINT is required for the endpoint provider.")

        return ChatOpenAI(
            model=selected_model,
            api_key=api_key,
            base_url=endpoint,
            temperature=temperature,
            metadata={"provider": provider_name},
        )
    raise ValueError("This lab supports `google`, `ollama`, and `endpoint` providers.")


def extract_json_object(raw: Any) -> dict[str, Any]:
    text = normalize_content(raw)
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if blocks:
            text = blocks[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def judge_answer_with_llm(
    *,
    query: str,
    answer: str,
    rubric: str,
    provider: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    prompt = f"""
You are grading a student order-agent answer.
Return JSON only with:
- score: integer from 0 to 10
- verdict: short string
- feedback: short list of strings

Rubric:
{rubric}

User query:
{query}

Student answer:
{answer}
""".strip()
    payload = extract_json_object(model.invoke(prompt).content)
    score = max(0, min(10, int(payload.get("score", 0))))
    return {
        "score": score,
        "verdict": str(payload.get("verdict", "")).strip(),
        "feedback": [str(item).strip() for item in payload.get("feedback", []) if str(item).strip()],
    }
