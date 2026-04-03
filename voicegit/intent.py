from __future__ import annotations

from typing import Optional

from dataset import TRAINING_DATA
from keywords import KEYWORD_RULES
from model import predict_intent

CONFIDENCE_THRESHOLD: float = 0.45

ALL_INTENTS: list[str] = [
    # Setup & config
    "init", "clone", "clone_branch", "clone_shallow",
    "remote", "remote_add", "remote_remove",
    # Core workflow
    "status", "add", "add_patch",
    "commit", "commit_amend",
    "push", "push_force", "push_upstream", "push_tags",
    "pull", "pull_rebase", "fetch",
    # Branching
    "list_branches", "create_branch", "checkout", "delete_branch",
    "merge_branch", "merge_squash", "merge_abort",
    "rebase", "rebase_interactive", "rebase_abort", "rebase_continue",
    # Inspection
    "log", "log_oneline", "log_graph", "log_file", "log_stat",
    "diff", "diff_staged", "diff_branches", "diff_commit",
    "show", "blame",
    # Undoing
    "restore", "revert",
    "reset", "reset_soft", "reset_hard", "reset_mixed", "reset_unstage",
    "stash", "stash_pop", "stash_apply", "stash_list", "stash_drop",
    # File ops
    "remove_file", "move_file",
]


def _normalise(text: str) -> str:
    import re

    text = text.lower().strip()
    text = re.sub(r"['\"]", "", text)
    text = re.sub(r"[^a-z0-9\s\-\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keyword_fallback(text: str) -> Optional[str]:
    normalized = _normalise(text)
    for rule in KEYWORD_RULES:
        if not rule:
            continue
        *keyword_parts, intent = rule
        keywords: list[str] = []
        for part in keyword_parts:
            if isinstance(part, list):
                keywords.extend(part)
            else:
                keywords.append(str(part))
        for kw in sorted(keywords, key=len, reverse=True):
            clean_kw = _normalise(kw)
            if clean_kw and clean_kw in normalized:
                return intent
    return None


def classify(user_input: str) -> tuple[str, float]:
    text = (user_input or "").strip()
    if not text:
        return "unknown", 0.0

    fallback = _keyword_fallback(text)
    if fallback:
        return fallback, 1.0

    intent, confidence = predict_intent(text)
    if confidence >= CONFIDENCE_THRESHOLD:
        return intent, confidence
    return "unknown", 0.0


def detect_intent(user_input: str) -> str:
    intent, _ = classify(user_input)
    return intent


__all__ = ["ALL_INTENTS", "classify", "detect_intent", "TRAINING_DATA", "KEYWORD_RULES"]
