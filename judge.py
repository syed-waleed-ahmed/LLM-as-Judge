import json
from typing import Dict, Any

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL
from prompts import CRITERIA_PRESETS, Criterion, build_judge_system_prompt


# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


class JudgeError(Exception):
    """Custom exception for judge-related errors."""
    pass


def judge_pair(
    task_description: str,
    output_a: str,
    output_b: str,
    criterion_name: str,
) -> Dict[str, Any]:
    """
    Call the LLM judge (via Groq API) and return a dict with scores & explanations.

    Raises JudgeError on any failure.
    """
    if criterion_name in CRITERIA_PRESETS:
        criterion = CRITERIA_PRESETS[criterion_name]
    else:
        # Treat as custom criterion name with generic description
        criterion = Criterion(
            name=criterion_name,
            description="Custom user-defined evaluation criterion.",
        )

    system_prompt = build_judge_system_prompt(criterion)

    user_content = f"""
TASK:
{task_description.strip()}

OUTPUT A:
{output_a.strip()}

OUTPUT B:
{output_b.strip()}
    """.strip()

    try:
        # Groq's API is OpenAI-compatible for chat completions
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,  # deterministic judgment
        )
    except Exception as e:
        raise JudgeError(f"Error calling Groq API: {e}")

    try:
        content = response.choices[0].message.content.strip()
    except Exception as e:
        raise JudgeError(f"Malformed Groq response: {e}")

    # LLM should output pure JSON, but be defensive
    try:
        # If the model wrapped JSON in ```json ... ``` fences, strip them
        if content.startswith("```"):
            # Strip triple backticks on both sides
            content = content.strip("`")
            # Strip optional 'json\n'
            if content.lower().startswith("json"):
                content = content.split("\n", 1)[1]

        data = json.loads(content)
    except Exception as e:
        raise JudgeError(
            f"Failed to parse JSON from judge: {e}\n\nRaw content was:\n{content}"
        )

    # Basic validation
    required_keys = [
        "criterion",
        "output_a_score",
        "output_b_score",
        "output_a_explanation",
        "output_b_explanation",
        "winner",
        "overall_comment",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise JudgeError(f"Judge JSON missing keys: {missing}")

    return data
