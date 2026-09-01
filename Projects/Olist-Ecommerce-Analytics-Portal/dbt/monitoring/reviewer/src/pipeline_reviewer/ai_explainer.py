from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types

EXPLANATION_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "pipeline_summary": {
            "type": "STRING",
            "description": (
                "A concise summary of the deterministic pipeline findings."
            ),
        },
        "findings": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "finding_id": {
                        "type": "STRING",
                        "description": (
                            "The identifier of an existing deterministic finding."
                        ),
                    },
                    "explanation": {
                        "type": "STRING",
                    },
                    "impact": {
                        "type": "STRING",
                    },
                    "recommended_actions": {
                        "type": "ARRAY",
                        "items": {
                            "type": "STRING",
                        },
                    },
                },
                "required": [
                    "finding_id",
                    "explanation",
                    "impact",
                    "recommended_actions",
                ],
            },
        },
    },
    "required": [
        "pipeline_summary",
        "findings",
    ],
}


def explain_finding_package(
    finding_package: dict[str, Any],
    project_id: str,
    location: str = "us-central1",
) -> dict[str, Any]:
    client = genai.Client(
        enterprise=True,
        project=project_id,
        location=location,
        http_options=types.HttpOptions(api_version="v1"),
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=json.dumps(
                finding_package,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a pipeline reliability explanation assistant. "
                    "The deterministic reviewer is the source of truth. "
                    "You must not change rule results, severity, evidence, "
                    "thresholds, entity identifiers, or invent new findings. "
                    "Explain only the findings provided in the input. "
                    "For each finding, explain what happened, the likely impact, "
                    "and practical next investigation steps. "
                    "If evidence is insufficient, say so explicitly."
                ),
                response_mime_type="application/json",
                response_schema=EXPLANATION_RESPONSE_SCHEMA,
            ),
        )

        return _validate_explanation_response(
            finding_package,
            response.text,
        )
    finally:
        client.close()


def _validate_explanation_response(
    finding_package: dict[str, Any],
    response_text: str,
) -> dict[str, Any]:
    parsed = json.loads(response_text)

    if not isinstance(parsed, dict):
        raise ValueError("AI explanation response must be a JSON object")

    explanations = parsed.get("findings")

    if not isinstance(explanations, list):
        raise ValueError("AI explanation response must contain a findings list")

    expected_ids = {finding["finding_id"] for finding in finding_package["findings"]}

    returned_ids = [explanation["finding_id"] for explanation in explanations]

    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError("AI explanation response contains duplicate finding_id values")

    returned_id_set = set(returned_ids)

    if returned_id_set != expected_ids:
        raise ValueError(
            "AI explanation finding_id values do not match "
            "the deterministic findings"
        )

    return parsed


def build_explanation_report(
    finding_package: dict[str, Any],
    project_id: str,
    location: str = "us-central1",
) -> dict[str, Any]:
    findings = finding_package.get("findings", [])

    if not findings:
        return {
            "ai_status": "SKIPPED",
            "pipeline_summary": ("No deterministic findings were triggered."),
            "findings": [],
        }

    try:
        explanation = explain_finding_package(
            finding_package=finding_package,
            project_id=project_id,
            location=location,
        )

        return {
            "ai_status": "SUCCESS",
            **explanation,
        }

    except Exception as exc:
        return {
            "ai_status": "UNAVAILABLE",
            "pipeline_summary": (
                "Deterministic findings are available, "
                "but AI explanation could not be generated."
            ),
            "findings": [],
            "error": str(exc),
        }