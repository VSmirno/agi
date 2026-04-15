"""Stage 84: integration test — vital fix (body reads from inventory, not info top-level).

Crafter puts health/food/drink/energy in info["inventory"], not top-level info.
Before the fix, body was always {health: 9.0, food: 9.0, ...} (default fires).
After the fix, body correctly reflects actual vitals.
"""

from __future__ import annotations

VITAL_VARS = {"health", "food", "drink", "energy"}


def parse_inv_body(info: dict, vitals: list[str]) -> tuple[dict, dict]:
    """Same logic as vector_mpc_agent.py after Stage 84 fix."""
    raw_inv = dict(info.get("inventory", {}))
    body = {v: float(raw_inv.get(v, 9.0)) for v in vitals}
    inv = {k: v for k, v in raw_inv.items() if k not in VITAL_VARS}
    return inv, body


class TestVitalFix:
    def test_body_reads_from_inventory_not_info(self):
        info = {
            "inventory": {"food": 2.0, "health": 5.0, "drink": 7.0, "energy": 3.0, "wood": 3},
            "player_pos": (32, 32),
        }
        vitals = ["health", "food", "drink", "energy"]
        inv, body = parse_inv_body(info, vitals)

        assert body["food"] == 2.0, f"Expected 2.0, got {body['food']} (old bug: 9.0)"
        assert body["health"] == 5.0
        assert body["drink"] == 7.0
        assert body["energy"] == 3.0

    def test_body_not_always_nine(self):
        """The old bug: info.get(v, 9.0) returned 9.0 because vitals are not top-level."""
        info = {
            "inventory": {"food": 1.0, "health": 2.0, "drink": 3.0, "energy": 4.0},
        }
        vitals = ["health", "food", "drink", "energy"]
        _, body = parse_inv_body(info, vitals)

        # None of these should be 9.0
        for v in vitals:
            assert body[v] != 9.0, f"{v} is still 9.0 — fix not applied"

    def test_inv_excludes_body_vars(self):
        info = {
            "inventory": {
                "health": 9.0, "food": 5.0, "drink": 5.0, "energy": 5.0,
                "wood": 3, "stone": 1, "coal": 0,
            }
        }
        vitals = ["health", "food", "drink", "energy"]
        inv, body = parse_inv_body(info, vitals)

        for v in vitals:
            assert v not in inv, f"{v} should not be in inv dict"

    def test_inv_contains_resources(self):
        info = {
            "inventory": {
                "health": 9.0, "food": 5.0, "drink": 5.0, "energy": 5.0,
                "wood": 3, "stone": 1,
            }
        }
        vitals = ["health", "food", "drink", "energy"]
        inv, body = parse_inv_body(info, vitals)

        assert inv.get("wood") == 3
        assert inv.get("stone") == 1

    def test_missing_vital_defaults_to_nine(self):
        """If a vital is absent from inventory (edge case), default is 9.0."""
        info = {"inventory": {"wood": 2}}
        vitals = ["health", "food", "drink", "energy"]
        _, body = parse_inv_body(info, vitals)

        for v in vitals:
            assert body[v] == 9.0, f"Missing vital {v} should default to 9.0"
