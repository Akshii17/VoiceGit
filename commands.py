from __future__ import annotations

import json
from typing import List, Optional

from intent import ALL_INTENTS
from state import RepoState


def _needs_stage_all(state: RepoState) -> bool:
    """True when there are changes but nothing is staged yet."""
    return state.has_changes and state.staged_files == 0


def _has_anything_to_commit(state: RepoState) -> bool:
    return state.has_changes or state.staged_files > 0


def _prompt_nonempty(prompt: str, what: str) -> str | None:
    value = input(prompt).strip()
    if not value:
        print(f"No {what} provided.")
        return None
    return value


def generate_commands(
    intent: str,
    state: Optional[RepoState],
) -> List[str]:
    """
    Return a list of git commands for the given intent.

    ``init``/``clone`` work without a RepoState; all others expect ``state``.
    Unknown or unimplemented intents always return an ``ERROR: …`` entry.
    """
    normalized = (intent or "").strip().lower()

    if normalized not in ALL_INTENTS:
        return ["ERROR: unknown intent"]

    # ── Setup / cloning ─────────────────────────────────────────────────────
    if normalized == "init":
        return ["git init"]

    if normalized in {"clone", "clone_branch", "clone_shallow"}:
        url = _prompt_nonempty("Repository URL: ", "repository URL")
        if url is None:
            return ["ERROR: no repository URL provided"]
        if normalized == "clone_shallow":
            return [f"git clone --depth 1 {url}"]
        if normalized == "clone_branch":
            branch = _prompt_nonempty("Branch name: ", "branch name")
            if branch is None:
                return ["ERROR: no branch name provided"]
            return [f"git clone -b {branch} {url}"]
        return [f"git clone {url}"]

    # All remaining intents require a repo state.
    if state is None:
        return ["ERROR: not inside a repository context for this command"]

    # ── Core workflow ───────────────────────────────────────────────────────
    if normalized == "status":
        return ["git status"]

    if normalized == "add":
        return ["git add ."]

    if normalized == "add_patch":
        return ["git add -p"]

    if normalized == "commit":
        if not _has_anything_to_commit(state):
            return ["ERROR: no changes"]
        cmds: List[str] = []
        if _needs_stage_all(state):
            cmds.append("git add .")
        message = input("Commit message (empty = auto commit): ").strip()
        if not message:
            message = "auto commit"
        cmds.append(f"git commit -m {json.dumps(message)}")
        return cmds

    if normalized == "commit_amend":
        return ["git commit --amend"]

    if normalized == "push":
        if not state.has_commits:
            return ["ERROR: no commits"]
        cmds: List[str] = []
        if state.has_changes:
            if _needs_stage_all(state):
                cmds.append("git add .")
            message = input("Commit message for local changes (empty = auto commit): ").strip()
            if not message:
                message = "auto commit"
            cmds.append(f"git commit -m {json.dumps(message)}")
        cmds.append("git push origin main")
        return cmds

    if normalized == "push_force":
        return ["git push --force"]

    if normalized == "push_upstream":
        branch = _prompt_nonempty("Branch name to push upstream: ", "branch name")
        if branch is None:
            return ["ERROR: no branch name provided"]
        return [f"git push -u origin {branch}"]

    if normalized == "push_tags":
        return ["git push --tags"]

    if normalized == "pull":
        return ["git pull origin main"]

    if normalized == "pull_rebase":
        return ["git pull --rebase origin main"]

    if normalized == "fetch":
        return ["git fetch"]

    # ── Branching ───────────────────────────────────────────────────────────
    if normalized == "list_branches":
        return ["git branch -a"]

    if normalized == "create_branch":
        name = _prompt_nonempty("New branch name: ", "branch name")
        if name is None:
            return ["ERROR: no branch name provided"]
        return [f"git checkout -b {name}"]

    if normalized == "checkout":
        name = _prompt_nonempty("Branch name to checkout: ", "branch name")
        if name is None:
            return ["ERROR: no branch name provided"]
        return [f"git checkout {name}"]

    if normalized == "delete_branch":
        name = _prompt_nonempty("Branch name to delete: ", "branch name")
        if name is None:
            return ["ERROR: no branch name provided"]
        return [f"git branch -d {name}"]

    if normalized == "merge_branch":
        name = _prompt_nonempty("Branch to merge: ", "branch name")
        if name is None:
            return ["ERROR: no branch name provided"]
        return [f"git merge {name}"]

    if normalized == "merge_abort":
        return ["git merge --abort"]

    if normalized == "merge_squash":
        name = _prompt_nonempty("Branch to squash-merge: ", "branch name")
        if name is None:
            return ["ERROR: no branch name provided"]
        return [f"git merge --squash {name}"]

    # ── Rebase ──────────────────────────────────────────────────────────────
    if normalized == "rebase":
        name = _prompt_nonempty("Branch to rebase onto: ", "branch name")
        if name is None:
            return ["ERROR: no branch name provided"]
        return [f"git rebase {name}"]

    if normalized == "rebase_interactive":
        return ["git rebase -i HEAD~3"]

    if normalized == "rebase_abort":
        return ["git rebase --abort"]

    if normalized == "rebase_continue":
        return ["git rebase --continue"]

    # ── Inspection / history ────────────────────────────────────────────────
    if normalized == "log":
        return ["git log"]

    if normalized == "log_oneline":
        return ["git log --oneline"]

    if normalized == "log_graph":
        return ["git log --graph --oneline --all"]

    if normalized == "diff":
        return ["git diff"]

    if normalized == "diff_staged":
        return ["git diff --staged"]

    if normalized == "show":
        return ["git show"]

    if normalized == "blame":
        path = _prompt_nonempty("File to blame: ", "file path")
        if path is None:
            return ["ERROR: no file path provided"]
        return [f"git blame {path}"]

    # ── Undo / restore / reset ──────────────────────────────────────────────
    if normalized == "restore":
        path = _prompt_nonempty("File to restore: ", "file path")
        if path is None:
            return ["ERROR: no file path provided"]
        return [f"git restore {path}"]

    if normalized == "revert":
        return ["git revert HEAD"]

    if normalized == "reset":
        return ["git reset"]

    if normalized == "reset_soft":
        return ["git reset --soft HEAD~1"]

    if normalized == "reset_hard":
        return ["git reset --hard HEAD~1"]

    if normalized == "reset_mixed":
        return ["git reset HEAD~1"]

    if normalized == "reset_unstage":
        path = _prompt_nonempty("File to unstage: ", "file path")
        if path is None:
            return ["ERROR: no file path provided"]
        return [f"git reset HEAD {path}"]

    # ── Stash ───────────────────────────────────────────────────────────────
    if normalized == "stash":
        return ["git stash"]

    if normalized == "stash_pop":
        return ["git stash pop"]

    if normalized == "stash_apply":
        return ["git stash apply"]

    if normalized == "stash_list":
        return ["git stash list"]

    if normalized == "stash_drop":
        return ["git stash drop"]

    # Any recognised but unimplemented intent
    return ["ERROR: Intent not implemented yet"]
