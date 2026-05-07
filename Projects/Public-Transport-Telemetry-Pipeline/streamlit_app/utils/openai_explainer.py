from __future__ import annotations

import os
import textwrap
from collections.abc import Iterable

from openai import OpenAI

DEFAULT_MODEL = os.getenv("OPENAI_EXPLANATION_MODEL", "gpt-4.1-mini")
DEFAULT_TIMEOUT_SECONDS = 20


def has_openai_config() -> bool:
    """
    Check whether OpenAI API configuration is available.

    The API key must be provided through environment variables or platform secrets.
    It must never be committed to the repository.
    """
    return bool(os.getenv("OPENAI_API_KEY"))


def clean_facts(facts: Iterable[str]) -> list[str]:
    """
    Normalize rule-based facts before sending them to the model.

    The model only receives these facts. It does not receive raw dataframes
    or access to source systems.
    """
    cleaned: list[str] = []

    for fact in facts:
        text = str(fact).strip()
        if text:
            cleaned.append(text)

    return cleaned


def build_explanation_prompt(
    facts: list[str],
    page_context: str,
) -> str:
    """
    Build a constrained prompt for dashboard explanation.

    The prompt explicitly prevents unsupported operational claims.
    """
    facts_text = "\n".join(f"- {fact}" for fact in facts)

    return textwrap.dedent(f"""
        You are writing a short explanation for a portfolio data engineering dashboard.

        Page context:
        {page_context}

        Precomputed dashboard facts:
        {facts_text}

        Write a concise plain-English explanation for a technical reviewer.

        Rules:
        - Use only the facts provided.
        - Do not calculate new metrics.
        - Do not inspect, assume, or invent raw data.
        - Do not infer root causes.
        - Do not claim live monitoring.
        - Do not claim prediction.
        - Do not claim weather impact or causal analysis.
        - Do not mention these rules.
        - Do not sound promotional or sales-like.
        - Use simple, professional English.
        - Write one compact paragraph under 120 words.
        - Prefer wording such as "scheduled snapshot", "exported Gold-layer metrics", and "descriptive summary".
        - If relevant, clarify that the metrics should be read as a scheduled snapshot, not as live monitoring.
        """).strip()


def generate_ai_explanation(
    facts: Iterable[str],
    page_context: str,
    model: str = DEFAULT_MODEL,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str | None:
    """
    Generate a short explanation from precomputed dashboard facts only.

    This helper intentionally keeps OpenAI outside the metric calculation path.
    If the API key is missing or the request fails, return None so the dashboard
    can fall back to deterministic rule-based insights.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    fact_list = clean_facts(facts)
    if not fact_list:
        return None

    client = OpenAI(
        api_key=api_key,
        timeout=timeout_seconds,
    )

    prompt = build_explanation_prompt(
        facts=fact_list,
        page_context=page_context,
    )

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
        )

        output_text = response.output_text.strip()

        if not output_text:
            return None

        # Dashboard display safety cap.
        # The prompt asks for under 120 words, but this prevents unexpectedly long output.
        words = output_text.split()
        if len(words) > 140:
            output_text = " ".join(words[:140]).rstrip() + "..."

        return output_text

    except Exception as exc:
        print(f"OpenAI explanation failed: {exc}")
        return None
