from __future__ import annotations

from pathlib import Path

from snks.agent.capability_state import extract_capability_state
from snks.agent.crafter_textbook import CrafterTextbook


TEXTBOOK_PATH = Path(__file__).parent.parent / "configs" / "crafter_textbook.yaml"


def test_extract_capability_state_marks_weapon_inventory_as_armed_melee():
    textbook = CrafterTextbook(str(TEXTBOOK_PATH))

    state = extract_capability_state({"wood_sword": 1}, textbook)

    assert state.armed_melee is True
    assert state.weapons == ("wood_sword",)
    assert state.to_trace() == {
        "armed_melee": True,
        "weapons": ["wood_sword"],
    }


def test_extract_capability_state_unarmed_without_weapon():
    textbook = CrafterTextbook(str(TEXTBOOK_PATH))

    state = extract_capability_state({"wood": 5}, textbook)

    assert state.armed_melee is False
    assert state.weapons == ()
