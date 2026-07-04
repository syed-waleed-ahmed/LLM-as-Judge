"""Prompt construction for the judge.

The prompts are deliberately explicit about the JSON contract so the model's
output validates against :class:`~llm_judge.models.JudgeResult`. We also pair
this with the provider's JSON output mode for belt-and-braces reliability.
"""

from __future__ import annotations

import json

from .models import Criterion


def build_system_prompt(criteria: list[Criterion]) -> str:
    """Build the system prompt describing the judging task and output schema."""
    criteria_block = "\n".join(
        f"- {c.name}: {c.description} (score {c.scale_min}-{c.scale_max}, integers)"
        for c in criteria
    )

    example = {
        "criteria": [
            {
                "criterion": c.name,
                "output_a_score": c.scale_min,
                "output_b_score": c.scale_max,
                "output_a_explanation": "Concise reason for A's score.",
                "output_b_explanation": "Concise reason for B's score.",
                "winner": "B",
            }
            for c in criteria
        ],
        "overall_winner": "B",
        "overall_comment": "One or two sentences summarising the overall judgment.",
    }

    return (
        "You are an impartial, rigorous evaluation judge. You compare two "
        "candidate outputs (A and B) for a given task and score them on each "
        "of the provided criteria.\n\n"
        "Criteria to evaluate:\n"
        f"{criteria_block}\n\n"
        "Rules:\n"
        "1. Judge each output only on how well it accomplishes the TASK.\n"
        "2. Score each output independently on every criterion; do not let one "
        "criterion's score influence another.\n"
        "3. Ignore the order in which the outputs are presented — position must "
        "not affect your scores.\n"
        "4. Ignore output length unless the criterion is explicitly about it.\n"
        "5. Keep explanations concise (one to three sentences) and specific.\n"
        "6. 'winner' per criterion and 'overall_winner' must be exactly \"A\", "
        "\"B\", or \"tie\".\n\n"
        "Respond with a SINGLE valid JSON object and nothing else — no prose, no "
        "markdown fences. It must match this schema exactly:\n"
        f"{json.dumps(example, indent=2)}"
    )


def build_user_prompt(task_description: str, output_a: str, output_b: str) -> str:
    """Build the user message containing the task and the two outputs."""
    return (
        "TASK:\n"
        f"{task_description.strip()}\n\n"
        "=== OUTPUT A ===\n"
        f"{output_a.strip()}\n\n"
        "=== OUTPUT B ===\n"
        f"{output_b.strip()}\n"
    )
