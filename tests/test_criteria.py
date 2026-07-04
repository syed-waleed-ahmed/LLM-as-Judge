from __future__ import annotations

from llm_judge.criteria import CRITERIA_PRESETS, resolve_criterion
from llm_judge.models import Criterion


def test_resolve_preset_returns_registered_criterion():
    crit = resolve_criterion("Creativity")
    assert crit is CRITERIA_PRESETS["Creativity"]


def test_resolve_custom_builds_new_criterion():
    crit = resolve_criterion("Politeness")
    assert isinstance(crit, Criterion)
    assert crit.name == "Politeness"
    assert "Politeness" in crit.description


def test_resolve_custom_uses_supplied_description():
    crit = resolve_criterion("Snark", description="How witty the response is.")
    assert crit.description == "How witty the response is."


def test_all_presets_are_valid_criteria():
    for name, crit in CRITERIA_PRESETS.items():
        assert crit.name == name
        assert crit.scale_max > crit.scale_min
