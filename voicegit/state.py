from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
from typing import List


@dataclass(frozen=True)
class RepoState:
    has_changes: bool
    staged_files: int
    has_commits: bool
    ahead_of_remote: int
    behind_of_remote: int
    current_branch: str
    has_conflicts: bool


def _run(args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
    )


def _ensure_git_available() -> None:
    try:
        proc = _run(["git", "--version"])
    except FileNotFoundError as e:
        raise RuntimeError("`git` is not installed or not available on PATH.") from e

    if proc.returncode != 0:
        raise RuntimeError("`git` is not installed or not working correctly.")


def _is_git_repository() -> bool:
    try:
        proc = _run(["git", "rev-parse", "--is-inside-work-tree"])
    except FileNotFoundError:
        return False

    return proc.returncode == 0 and proc.stdout.strip().lower() == "true"


def _parse_ahead_behind(status_text: str) -> tuple[int, int]:
    ahead = 0
    behind = 0

    # Typical lines:
    # "Your branch is ahead of 'origin/main' by 1 commit."
    # "Your branch is behind 'origin/main' by 2 commits, and can be fast-forwarded."
    m_ahead = re.search(r"\bahead of\b.*?\bby\s+(\d+)\s+commit", status_text, flags=re.I)
    if m_ahead:
        ahead = int(m_ahead.group(1))

    m_behind = re.search(r"\bbehind\b.*?\bby\s+(\d+)\s+commit", status_text, flags=re.I)
    if m_behind:
        behind = int(m_behind.group(1))

    return ahead, behind


def _parse_has_changes(status_text: str) -> bool:
    # Prefer the "clean" indicator when present.
    if "nothing to commit, working tree clean" in status_text.lower():
        return False

    # Otherwise, presence of these sections usually indicates changes.
    indicators = [
        "Changes to be committed:",
        "Changes not staged for commit:",
        "Untracked files:",
        "Unmerged paths:",
    ]
    return any(s in status_text for s in indicators)


def _parse_has_commits(status_text: str) -> bool:
    # Typical when repo has no commits:
    # "No commits yet"
    return "no commits yet" not in status_text.lower()


def _parse_has_conflicts(status_text: str) -> bool:
    lower = status_text.lower()
    if "unmerged paths:" in lower:
        return True
    if "you have unmerged paths." in lower:
        return True
    # Conflict markers in status list lines can show up like:
    # "both modified:" / "both added:" / "deleted by us:" etc.
    conflict_phrases = [
        "both modified:",
        "both added:",
        "deleted by them:",
        "deleted by us:",
        "added by us:",
        "added by them:",
        "unmerged:",
    ]
    return any(p in lower for p in conflict_phrases)


def _parse_staged_files(status_text: str) -> int:
    # Count file lines under "Changes to be committed:" until the next blank line.
    lines = status_text.splitlines()
    in_staged = False
    count = 0

    for line in lines:
        if not in_staged:
            if line.strip() == "Changes to be committed:":
                in_staged = True
            continue

        # After the header, git prints a hint line and then staged entries like:
        # "\tmodified:   path"
        if not line.strip():
            break

        # Heuristic: staged entries are indented and contain ":" (e.g., "modified:").
        if (line.startswith("\t") or line.startswith("  ")) and ":" in line:
            count += 1

    return count


def get_repo_state() -> RepoState:
    _ensure_git_available()

    if not _is_git_repository():
        raise RuntimeError("Not inside a git repository. Run this tool from within a repo.")

    status_proc = _run(["git", "status"])
    if status_proc.returncode != 0:
        msg = (status_proc.stderr or status_proc.stdout or "").strip()
        raise RuntimeError(msg or f"`git status` failed (exit code {status_proc.returncode}).")
    status_text = status_proc.stdout or ""

    branch_proc = _run(["git", "branch", "--show-current"])
    if branch_proc.returncode != 0:
        msg = (branch_proc.stderr or branch_proc.stdout or "").strip()
        raise RuntimeError(
            msg or f"`git branch --show-current` failed (exit code {branch_proc.returncode})."
        )
    current_branch = (branch_proc.stdout or "").strip()

    # Detached HEAD: `--show-current` returns empty; best-effort fallback from status.
    if not current_branch:
        m = re.search(r"^HEAD detached (?:at|from)\s+(.+)$", status_text, flags=re.M)
        if m:
            current_branch = f"(detached: {m.group(1).strip()})"
        else:
            current_branch = "(detached)"

    ahead, behind = _parse_ahead_behind(status_text)

    return RepoState(
        has_changes=_parse_has_changes(status_text),
        staged_files=_parse_staged_files(status_text),
        has_commits=_parse_has_commits(status_text),
        ahead_of_remote=ahead,
        behind_of_remote=behind,
        current_branch=current_branch,
        has_conflicts=_parse_has_conflicts(status_text),
    )

