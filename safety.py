from __future__ import annotations

from typing import List, Tuple

from state import RepoState


def validate_commands(commands: List[str], state: RepoState) -> Tuple[bool, str]:
    """
    Validate a list of shell command strings against simple safety rules.

    Returns:
      - (True, "Safe") when allowed
      - (False, "<reason>") when blocked
    """
    if state.has_conflicts:
        return False, "Repository has merge conflicts (has_conflicts=True)."

    if any("ERROR" in c for c in commands):
        return False, "Commands contain an ERROR entry."

    blocked_substrings = [
        "reset --hard",
        "push --force",
    ]

    for c in commands:
        lower = c.lower()
        for blocked in blocked_substrings:
            if blocked in lower:
                return False, f"Blocked command detected: contains '{blocked}'."

    return True, "Safe"

