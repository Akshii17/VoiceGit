from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent
NOTES_FILE = BASE_DIR / "learning_notes.txt"
STREAM_FILE = BASE_DIR / "learning_stream.txt"
WINDOW_SCRIPT = BASE_DIR / "learning_window.py"

learning_mode = False
_learning_process: Optional[subprocess.Popen] = None


def explain_command(cmd: str) -> str:
    text = (cmd or "").strip().lower()
    if text.startswith("git clone"):
        return "Copies a remote repository into a new local folder"
    if text.startswith("git init"):
        return "Creates a new empty Git repository in the current directory"
    if text.startswith("git add"):
        return "Stages your changes (prepares files to be committed)"
    if text.startswith("git commit"):
        return "Saves a snapshot of your project locally"
    if text.startswith("git push"):
        return "Uploads your commits to the remote repository"
    if text.startswith("git pull"):
        return "Downloads and integrates changes from the remote repository"
    if text.startswith("git checkout -b"):
        return "Creates a new branch and switches to it"
    if text.startswith("git checkout"):
        return "Switches branches or restores working tree files"
    if text.startswith("git merge"):
        return "Joins another branch's history into the current branch"
    if text.startswith("git branch"):
        return "Lists, creates, or deletes branches"
    if text.startswith("git log"):
        return "Shows commit history"
    if text.startswith("git diff"):
        return "Shows changes between commits, branches, or working tree"
    if text.startswith("git reset"):
        return "Moves the current branch pointer (here: soft reset one commit)"
    if text.startswith("git revert"):
        return "Creates a new commit that undoes a previous commit"
    if text.startswith("git stash"):
        return "Temporarily saves uncommitted changes for later"
    if text.startswith("git status"):
        return "Shows working tree status (staged, modified, untracked)"
    return ""


def store_learning(cmd: str, explanation: str) -> None:
    if not cmd or not explanation:
        return

    def canonical_command_key(command: str) -> str:
        text = command.strip().lower()

        # Collapse variable argument variants (messages, branch names, file paths, URLs).
        prefix_to_key = [
            ("git commit --amend", "git commit --amend"),
            ("git commit", "git commit"),
            ("git checkout -b", "git checkout -b <branch>"),
            ("git checkout", "git checkout <branch>"),
            ("git merge --squash", "git merge --squash <branch>"),
            ("git merge", "git merge <branch>"),
            ("git push -u origin", "git push -u origin <branch>"),
            ("git push origin", "git push origin <branch>"),
            ("git clone -b", "git clone -b <branch> <url>"),
            ("git clone --depth", "git clone --depth <n> <url>"),
            ("git clone", "git clone <url>"),
            ("git blame", "git blame <file>"),
            ("git restore", "git restore <file>"),
            ("git reset head", "git reset HEAD <file>"),
            ("git branch -d", "git branch -d <branch>"),
            ("git rebase", "git rebase <branch>"),
        ]

        for prefix, key in prefix_to_key:
            if text.startswith(prefix):
                return key

        return command.strip()

    NOTES_FILE.touch(exist_ok=True)
    content = NOTES_FILE.read_text(encoding="utf-8")
    marker = f"Command: {canonical_command_key(cmd)}"
    if marker in content:
        return

    with NOTES_FILE.open("a", encoding="utf-8") as f:
        if content and not content.endswith("\n"):
            f.write("\n")
        f.write(f"Command: {canonical_command_key(cmd)}\n")
        f.write(f"Explanation: {explanation}\n\n")


def write_to_learning_stream(text: str) -> None:
    if not text:
        return
    STREAM_FILE.touch(exist_ok=True)
    with STREAM_FILE.open("a", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def start_learning_window() -> None:
    global learning_mode, _learning_process

    if _learning_process and _learning_process.poll() is None:
        learning_mode = True
        return

    STREAM_FILE.touch(exist_ok=True)

    _learning_process = subprocess.Popen(
        [sys.executable, str(WINDOW_SCRIPT)],
        cwd=str(BASE_DIR),
    )
    learning_mode = True


def stop_learning_window() -> None:
    global learning_mode, _learning_process

    learning_mode = False
    if _learning_process and _learning_process.poll() is None:
        _learning_process.terminate()
    _learning_process = None
