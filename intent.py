from __future__ import annotations


def detect_intent(user_input: str) -> str:
    """
    Rule-based intent detection from free-form text.
    Returns an intent name or "unknown".
    """
    text = (user_input or "").strip().lower()
    if not text:
        return "unknown"

    if text.startswith("git "):
        rest = text[4:].strip().split()
        if rest:
            verb = rest[0]
            if verb in {
                "status",
                "add",
                "commit",
                "push",
                "pull",
                "init",
                "clone",
                "branch",
                "checkout",
                "merge",
                "log",
                "diff",
                "reset",
                "revert",
                "stash",
            }:
                return verb

    # Phrase-based (more specific first)
    if "check status" in text or "show status" in text:
        return "status"
    if "add files" in text or "stage files" in text:
        return "add"
    if "commit changes" in text:
        return "commit"
    if "push code" in text or "upload code" in text:
        return "push"
    if "pull latest" in text:
        return "pull"
    if "switch branch" in text:
        return "checkout"
    if "create branch" in text or "new branch" in text:
        return "checkout"

    # Single-word / keyword (avoid matching substrings where cheap: order matters)
    if "clone" in text:
        return "clone"
    if "init" in text or "initialize" in text:
        return "init"
    if "stash" in text:
        return "stash"
    if "revert" in text:
        return "revert"
    if "reset" in text:
        return "reset"
    if "diff" in text:
        return "diff"
    if "merge" in text:
        return "merge"
    if "checkout" in text:
        return "checkout"
    if "pull" in text:
        return "pull"
    if "push" in text or "upload" in text:
        return "push"
    if "commit" in text or "save" in text:
        return "commit"
    if "add" in text or "stage" in text:
        return "add"
    if "branch" in text:
        return "branch"
    if "log" in text:
        return "log"
    if "status" in text:
        return "status"

    return "unknown"
