from __future__ import annotations

from llm_judge.criteria import resolve_criterion
from llm_judge.prompts import build_system_prompt, build_user_prompt


def test_system_prompt_lists_every_criterion():
    criteria = [resolve_criterion("Creativity"), resolve_criterion("Conciseness")]
    prompt = build_system_prompt(criteria)
    assert "Creativity" in prompt
    assert "Conciseness" in prompt
    # It should describe the JSON contract.
    assert "overall_winner" in prompt
    assert "json" in prompt.lower()


def test_user_prompt_contains_task_and_outputs():
    prompt = build_user_prompt("  do the thing  ", "  alpha  ", "  beta  ")
    assert "do the thing" in prompt
    assert "alpha" in prompt
    assert "beta" in prompt
    assert "OUTPUT A" in prompt
    assert "OUTPUT B" in prompt
    # Inputs are stripped, not left with surrounding whitespace.
    assert "  alpha  " not in prompt
