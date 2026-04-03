from __future__ import annotations

from typing import List, Optional, Tuple

from state import RepoState


def validate_commands(
    commands: List[str],
    state: Optional[RepoState],
) -> Tuple[bool, str]:
    """
    Validate a list of shell command strings against simple safety rules.

    Returns:
      - (True, "Safe") when allowed
      - (True, "WARNING: <reason>") when risky but allowed with confirmation
      - (False, "<reason>") when blocked
    """
    if state is not None and state.has_conflicts:
        return False, "Repository has merge conflicts (has_conflicts=True)."

    if any("ERROR" in c for c in commands):
        return False, "Commands contain an ERROR entry."

    risky_substrings = [
        "reset --hard",
        "push --force",
    ]

    warnings: List[str] = []
    for c in commands:
        lower = c.lower()
        for risky in risky_substrings:
            if risky in lower:
                warnings.append(f"Command '{c}' can permanently rewrite history or discard changes.")

    if warnings:
        return True, "WARNING: " + " ".join(warnings)
    return True, "Safe"

