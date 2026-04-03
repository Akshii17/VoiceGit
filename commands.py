from __future__ import annotations

import json
from typing import List, Optional

from state import RepoState


def generate_commands(
    intent: str,
    state: Optional[RepoState],
) -> List[str]:
    """
    Build a list of git command strings from intent + optional RepoState.

    repo-only intents require ``state``; ``init`` and ``clone`` work with ``state is None``.
    Uses ``input()`` for clone URL, branch names, and commit message when needed.
    """
    normalized = (intent or "").strip().lower()

    if normalized == "init":
        return ["git init"]

    if normalized == "clone":
        url = input("Repository URL: ").strip()
        if not url:
            return ["ERROR: no repository URL provided"]
        return [f"git clone {url}"]

    if state is None:
        return ["ERROR: not inside a repository context for this command"]

    if normalized == "status":
        return ["git status"]

    if normalized == "add":
        return ["git add ."]

    if normalized == "commit":
        if not state.has_changes and state.staged_files == 0:
            return ["ERROR: no changes"]
        message = input("Commit message (empty = auto commit): ").strip()
        if not message:
            message = "auto commit"
        return [f"git commit -m {json.dumps(message)}"]

    if normalized == "push":
        if not state.has_commits:
            return ["ERROR: no commits"]
        return ["git push origin main"]

    if normalized == "pull":
        return ["git pull origin main"]

    if normalized == "branch":
        return ["git branch"]

    if normalized == "checkout":
        name = input("Branch name: ").strip()
        if not name:
            return ["ERROR: no branch name provided"]
        return [f"git checkout {name}"]

    if normalized == "merge":
        name = input("Branch to merge: ").strip()
        if not name:
            return ["ERROR: no branch name provided"]
        return [f"git merge {name}"]

    if normalized == "log":
        return ["git log"]

    if normalized == "diff":
        return ["git diff"]

    if normalized == "reset":
        return ["git reset --soft HEAD~1"]

    if normalized == "revert":
        return ["git revert HEAD"]

    if normalized == "stash":
        return ["git stash"]

    return ["ERROR: unknown intent"]
