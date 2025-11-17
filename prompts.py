from dataclasses import dataclass
from typing import Dict


@dataclass
class Criterion:
    name: str
    description: str
    scale_min: int = 1
    scale_max: int = 10


# Built-in criteria you can expand later
CRITERIA_PRESETS: Dict[str, Criterion] = {
    "Creativity": Criterion(
        name="Creativity",
        description="Originality, imagination, and interesting ideas in the response.",
    ),
    "Brand Tone Adherence": Criterion(
        name="Brand Tone Adherence",
        description=(
            "How well the response matches the desired brand voice, tone, and style."
        ),
    ),
    "Code Readability": Criterion(
        name="Code Readability",
        description=(
            "Clarity, organization, formatting, and ease of understanding of the code."
        ),
    ),
}


def build_judge_system_prompt(criterion: Criterion) -> str:
    """
    System prompt instructing the LLM to act as a strict, consistent judge.
    """
    return f"""
You are an impartial expert judge.

You will be given:
- A TASK description
- Two CANDIDATE outputs: Output A and Output B

Your job is to evaluate both outputs according to this single criterion:

Criterion: {criterion.name}
Description: {criterion.description}
Score range: {criterion.scale_min}–{criterion.scale_max} (integers only)

Instructions:
1. Carefully read the TASK and both outputs.
2. Score each output independently on the given criterion.
3. Provide a concise explanation for each score.
4. Choose which output is better, or 'tie' if genuinely equivalent.

Respond ONLY in valid JSON with this exact schema (no extra keys):

{{
  "criterion": "{criterion.name}",
  "scale_min": {criterion.scale_min},
  "scale_max": {criterion.scale_max},
  "output_a_score": <integer>,
  "output_b_score": <integer>,
  "output_a_explanation": "<string>",
  "output_b_explanation": "<string>",
  "winner": "A" | "B" | "tie",
  "overall_comment": "<string>"
}}
    """.strip()
