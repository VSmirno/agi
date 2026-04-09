"""Stage 76 Gate 5: automated no-hardcode lint.

Scans the Stage 76 new files for forbidden patterns:
- Hardcoded drive/variable lists ["health", "food", ...] outside of test
  harnesses and docstrings
- Magic number thresholds like `HP < 3`, `health > 5`
- `most_urgent_drive`, `dominant_drive`, or argmax over drives
- `if inv.get("X", 0) < N:` patterns (derived-feature branch)

Whitelisted names (documented in spec Open Questions #2):
- `bootstrap_k` (magic number)
- `similarity_threshold` (magic number)
- `temperature` (magic number)

The lint matches patterns defensively — any violation fails CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


STAGE76_FILES = [
    "src/snks/memory/state_encoder.py",
    "src/snks/memory/sdr_encoder.py",
    "src/snks/memory/episodic_sdm.py",
    "src/snks/agent/continuous_agent.py",
]


def _strip_comments_and_strings(content: str) -> str:
    """Remove line comments and string literals so we don't false-flag
    docstrings and comments. Simple regex; good enough for a lint test."""
    # Remove triple-quoted strings (docstrings)
    content = re.sub(r'"""[\s\S]*?"""', "", content)
    content = re.sub(r"'''[\s\S]*?'''", "", content)
    # Remove single-line strings
    content = re.sub(r'"[^"\n]*"', '""', content)
    content = re.sub(r"'[^'\n]*'", "''", content)
    # Remove # comments
    content = re.sub(r"#.*", "", content)
    return content


def _read_code(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        pytest.fail(f"Stage 76 file missing: {path}")
    return _strip_comments_and_strings(file_path.read_text())


# ---------------------------------------------------------------------------
# Forbidden patterns
# ---------------------------------------------------------------------------


class TestNoHardcodedDriveList:
    """Should not hardcode ["health", "food", "drink", "energy"] as a set/list."""

    FORBIDDEN_PATTERNS = [
        # list/set literals mentioning all four drives
        r'\[\s*""\s*,\s*""\s*,\s*""\s*,\s*""\s*\]',  # after string stripping
    ]

    def test_no_four_drive_literal(self):
        """No file should contain a 4-element list that was the body-drive list.

        After string stripping, a literal like ["health","food","drink","energy"]
        becomes ["","","",""] which is detectable.
        """
        for path in STAGE76_FILES:
            code = _read_code(path)
            # Matches list of 4+ empty strings (was hardcoded drive list)
            four_empty_lists = re.findall(
                r'\[\s*""(\s*,\s*"")+\s*\]', code
            )
            # Check if any list had 3+ commas (4+ elements)
            for match in four_empty_lists:
                comma_count = match.count(",")
                assert comma_count < 3, (
                    f"{path}: suspicious list of {comma_count+1} string literals "
                    f"(possible hardcoded drive list): {match}"
                )


class TestNoHardcodedDriveNames:
    """Should not use dominant_drive / most_urgent_drive / argmax over drives."""

    FORBIDDEN_NAMES = [
        r"\bmost_urgent_drive\b",
        r"\bdominant_drive\b",
        r"\bdrive_priority\b",
        r"\bdrive_argmax\b",
    ]

    def test_no_drive_argmax_patterns(self):
        for path in STAGE76_FILES:
            code = _read_code(path)
            for pattern in self.FORBIDDEN_NAMES:
                matches = re.findall(pattern, code)
                assert not matches, (
                    f"{path}: forbidden drive-argmax pattern: {pattern}"
                )


class TestNoDerivedFeatureThresholds:
    """Forbid if inv.get("X", 0) < N: patterns — these are derived features."""

    def test_no_inv_get_threshold_branch(self):
        # Matches:  if inv.get("something", 0) < 3
        #           if inv.get("X") > 5
        forbidden_re = re.compile(
            r'if\s+\w+\.get\(""\s*(,\s*\d+)?\s*\)\s*[<>]=?\s*\d'
        )
        for path in STAGE76_FILES:
            code = _read_code(path)
            matches = forbidden_re.findall(code)
            assert not matches, (
                f"{path}: forbidden derived-feature branch (if inv.get('X') < N): "
                f"{matches}"
            )


class TestNoHardcodedHealthBranches:
    """Forbid magic-number health/food/drink/energy comparisons."""

    def test_no_magic_body_comparisons(self):
        # After stripping, "health" → "", so the dangerous pattern is:
        #   if ""X < 3  → we match  [a-z]+ \s* [<>] \s* \d
        # But we can't distinguish from legit code; stricter: look for the
        # specific variable-name literals BEFORE stripping.
        for path in STAGE76_FILES:
            file_path = Path(path)
            raw = file_path.read_text()
            # Before stripping: look for patterns like `if health < 3`
            # or `inv["health"] < 3` etc.
            magic_patterns = [
                r'if\s+\w*health\w*\s*[<>]=?\s*\d',
                r'if\s+\w*food\w*\s*[<>]=?\s*\d',
                r'if\s+\w*drink\w*\s*[<>]=?\s*\d',
                r'if\s+\w*energy\w*\s*[<>]=?\s*\d',
            ]
            for pattern in magic_patterns:
                matches = re.findall(pattern, raw)
                # Exclude matches inside docstrings — we check if the match
                # line is inside a triple-quoted block. Simpler heuristic:
                # after strip-comments-strings, the pattern should be absent.
                stripped = _strip_comments_and_strings(raw)
                stripped_matches = re.findall(pattern, stripped)
                assert not stripped_matches, (
                    f"{path}: magic-number body comparison: "
                    f"{stripped_matches} (pattern={pattern})"
                )


class TestNoHardcodedDecisionThresholds:
    """Only whitelisted magic numbers allowed in decision paths.

    Whitelist: bootstrap_k, similarity_threshold, temperature, top_k,
    max_steps, capacity — these are tunable parameters, not embedded constants.
    """

    def test_no_inline_numeric_comparison_outside_whitelist(self):
        # This is loose: we only fail on obvious inline threshold comparisons
        # that reference hardcoded inventory variable names.
        for path in STAGE76_FILES:
            file_path = Path(path)
            raw = file_path.read_text()
            # Forbidden: `if inv["health"] < 3:` or `if inv.get("health", 0) < 3:`
            patterns = [
                r'inv\[""\]\s*[<>]=?\s*\d',
                r'inventory\[""\]\s*[<>]=?\s*\d',
            ]
            stripped = _strip_comments_and_strings(raw)
            for pattern in patterns:
                matches = re.findall(pattern, stripped)
                assert not matches, (
                    f"{path}: hardcoded inventory threshold check: {matches}"
                )


# ---------------------------------------------------------------------------
# Positive sanity check: tracker.observed_variables IS used
# ---------------------------------------------------------------------------


class TestUsesObservedVariables:
    """Stage 76 code MUST use tracker.observed_variables / observed_max, not
    hardcoded lists."""

    def test_score_actions_uses_observed_variables(self):
        code = Path("src/snks/memory/episodic_sdm.py").read_text()
        assert "observed_variables" in code or "observed_max" in code, (
            "episodic_sdm.py must reference tracker.observed_variables "
            "or observed_max for ideology compliance"
        )

    def test_continuous_agent_uses_observed_variables(self):
        code = Path("src/snks/agent/continuous_agent.py").read_text()
        assert "observed_variables" in code or "observed_max" in code, (
            "continuous_agent.py must reference tracker.observed_variables "
            "or observed_max for ideology compliance"
        )
