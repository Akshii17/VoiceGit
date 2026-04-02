from __future__ import annotations

from typing import List

from state import RepoState


def generate_commands(intent: str, state: RepoState) -> List[str]:
    """
    Generate deterministic git commands (as strings) from an intent + RepoState.

    Notes:
    - Errors are represented as a single-element list starting with "ERROR:".
    - Logic is intentionally simple and driven only by RepoState.
    """
    normalized = (intent or "").strip().lower()

    if normalized == "status":
        return ["git status"]

    if normalized == "commit":
        if not state.has_changes:
            return ["ERROR: no changes"]
        return ["git add .", "git commit -m 'auto'"]

    if normalized == "push":
        if not state.has_commits:
            return ["ERROR: no commits"]

        cmds: List[str] = []

        # "changes not staged" heuristic: repo has changes but nothing staged yet.
        changes_not_staged = state.has_changes and state.staged_files == 0
        if changes_not_staged:
            cmds.append("git add .")

        # If there are changes, ensure they get committed before pushing.
        if state.has_changes:
            cmds.append("git commit -m 'auto'")

        cmds.append("git push origin main")
        return cmds

    return ["ERROR: unknown intent"]

